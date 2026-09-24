"""Run a single superadditivity experiment.

This is the main entry point for the pipeline:
  config -> graph -> partition -> N clients -> D-SGD -> drift eval -> save.

Usage:
    python scripts/run_experiment.py data=moderate_noniid graph=sbm_medium
    python scripts/run_experiment.py experiment=core_factorial
"""

from __future__ import annotations

import logging
import sys
import time
from pathlib import Path

import hydra
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from superadditivity.datasets.dataset_loader import DatasetLoader
from superadditivity.datasets.semantic_partitioner import SemanticPartitioner
from superadditivity.datasets.quantity_skew_partitioner import QuantitySkewPartitioner
from superadditivity.datasets.client_dataset import ClientDataset
from superadditivity.graphs.graph_manager import GraphManager
from superadditivity.models import (
    build_resnet18_cifar,
    build_resnet34_cifar,
    build_convnet,
    build_wide_resnet,
    build_vit_tiny,
)
from superadditivity.models.model_utils import init_weights, clone_model
from superadditivity.training.decentralized_client import DecentralizedClient
from superadditivity.training.dsgd_coordinator import DSGDCoordinator
from superadditivity.training.fedavg_coordinator import FedAvgCoordinator
from superadditivity.training.local_only_coordinator import LocalOnlyCoordinator
from superadditivity.training.lr_schedule import CosineDecaySchedule
from superadditivity.evaluation.experiment_evaluator import ExperimentEvaluator
from superadditivity.evaluation.representation_extractor import RepresentationExtractor
from superadditivity.evaluation.cka_analyzer import CKAAnalyzer
from superadditivity.evaluation.rsa_analyzer import RSAAnalyzer
from superadditivity.evaluation.mmd_analyzer import MMDAnalyzer
from superadditivity.evaluation.fisher_analyzer import FisherAnalyzer
from superadditivity.evaluation.centroid_analyzer import CentroidAnalyzer
from superadditivity.evaluation.drift_tracker import DriftTracker
from superadditivity.logging.csv_logger import CSVLogger
from superadditivity.logging.metadata_store import MetadataStore
from superadditivity.logging.checkpoint_manager import CheckpointManager
from superadditivity.utils.seed import set_all_seeds, derived_seed, PROBE_SEED, seed_worker
from superadditivity.utils.device import select_device
from superadditivity.utils.io import ensure_dir, save_json

logger = logging.getLogger(__name__)


MODEL_REGISTRY = {
    "resnet18_cifar": build_resnet18_cifar,
    "resnet34_cifar": build_resnet34_cifar,
    "convnet4": build_convnet,
    "wide_resnet": build_wide_resnet,
    "vit_tiny": build_vit_tiny,
}


def build_model(
    cfg: DictConfig,
    n_classes: int,
    in_channels: int,
    image_size: int,
) -> torch.nn.Module:
    """Instantiate the model based on config, dataset channels, and resolution."""
    arch = cfg.model.architecture
    if arch not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown architecture: {arch!r}. "
            f"Choose from {sorted(MODEL_REGISTRY)}."
        )

    if arch in ("resnet18_cifar", "resnet34_cifar"):
        return MODEL_REGISTRY[arch](
            num_classes=n_classes, in_channels=in_channels,
        )
    elif arch == "convnet4":
        return MODEL_REGISTRY[arch](
            num_classes=n_classes, in_channels=in_channels,
        )
    elif arch == "wide_resnet":
        return MODEL_REGISTRY[arch](
            num_classes=n_classes,
            in_channels=in_channels,
            depth=cfg.model.get("depth", 28),
            widen_factor=cfg.model.get("widen_factor", 2),
            dropout=cfg.model.get("dropout", 0.0),
        )
    elif arch == "vit_tiny":
        return MODEL_REGISTRY[arch](
            num_classes=n_classes,
            in_channels=in_channels,
            image_size=image_size,
            patch_size=cfg.model.get("patch_size", 4),
        )
    else:
        raise ValueError(f"Unknown architecture: {arch}")


def _build_community_assignments_array(
    community_map: dict, n_clients: int
) -> np.ndarray:
    """Convert {comm_id: [client_ids]} to array[client_id] -> comm_id."""
    arr = np.zeros(n_clients, dtype=np.int64)
    for comm_id, client_ids in community_map.items():
        for cid in client_ids:
            arr[cid] = comm_id
    return arr


def _build_client_id_to_community(
    community_map: dict,
) -> dict:
    """Convert {comm_id: [client_ids]} to {client_id: comm_id}."""
    result = {}
    for comm_id, client_ids in community_map.items():
        for cid in client_ids:
            result[cid] = comm_id
    return result


def run(cfg: DictConfig) -> dict:
    """Execute a single experiment run."""
    t_start = time.time()
    run_seed = cfg.run_seed
    graph_seed = cfg.graph_seed
    output_dir = Path(cfg.output_dir) / f"seed_{run_seed}_graph_{graph_seed}"
    ensure_dir(output_dir)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(output_dir / "run.log"),
        ],
    )

    logger.info("=" * 60)
    logger.info("Starting experiment: %s", cfg.experiment_name)
    logger.info("Run seed: %d, Graph seed: %d", run_seed, graph_seed)
    logger.info("Config:\n%s", OmegaConf.to_yaml(cfg))
    logger.info("=" * 60)

    # ---- Validate n_clients ----
    n_clients = cfg.n_clients
    n_communities = cfg.n_communities
    if n_clients % n_communities != 0:
        raise ValueError(
            f"n_clients ({n_clients}) must be a multiple of "
            f"n_communities ({n_communities})."
        )

    set_all_seeds(run_seed)
    device = select_device()

    # ---- Dataset ----
    logger.info("Loading dataset: %s", cfg.data.dataset)
    loader = DatasetLoader(
        dataset_name=cfg.data.dataset,
        data_dir=cfg.data.data_dir,
        download=cfg.data.download,
    )
    train_dataset, test_dataset = loader.load()
    n_classes = loader.get_num_classes()
    in_channels = loader.get_in_channels()
    image_size = loader.get_image_size()

    # ---- Graph ----
    logger.info("Building graph: %s", cfg.graph.topology)
    graph_manager = GraphManager(cfg.graph, graph_seed=graph_seed)
    graph_manager.build()
    W = graph_manager.get_mixing_matrix()
    community_map = graph_manager.get_communities()
    graph_metrics = graph_manager.get_metrics()

    # ---- Data partition ----
    logger.info("Partitioning data: method=%s", cfg.data.partition_method)
    if cfg.data.partition_method == "quantity_skew":
        qty_partitioner = QuantitySkewPartitioner(
            n_clients=cfg.n_clients,
            alpha=cfg.data.dirichlet_alpha or 1.0,
        )
        targets = np.array(DatasetLoader._get_targets(train_dataset))
        qty_result = qty_partitioner.partition(
            targets,
            n_communities=cfg.n_communities,
            seed=derived_seed(run_seed, kind="dirichlet"),
        )
        client_indices = qty_result["client_indices"]
        community_arr = qty_result["community_assignments"]
    else:
        semantic_clusters = loader.get_semantic_clusters()
        partitioner = SemanticPartitioner(
            train_dataset=train_dataset,
            semantic_clusters=semantic_clusters,
            community_assignments=community_map,
            alpha=cfg.data.dirichlet_alpha or 1.0,
            mode=cfg.data.partition_method,
            run_seed=run_seed,
            num_clients=cfg.n_clients,
        )
        partition_result = partitioner.partition()
        client_indices = [partition_result[i] for i in range(cfg.n_clients)]
        community_arr = _build_community_assignments_array(
            community_map, cfg.n_clients
        )

    # ---- Model ----
    logger.info(
        "Initialising model: %s (in_channels=%d, image_size=%d)",
        cfg.model.architecture, in_channels, image_size,
    )
    init_seed = derived_seed(run_seed, kind="weight_init")
    set_all_seeds(init_seed)
    base_model = build_model(cfg, n_classes, in_channels, image_size)
    init_weights(base_model, seed=init_seed)
    layer_names = base_model.get_layer_names()

    # ---- Clients ----
    logger.info("Creating %d clients...", cfg.n_clients)
    clients = []
    for cid in range(cfg.n_clients):
        model_copy = clone_model(base_model)
        dataset = ClientDataset(
            train_dataset, client_indices[cid],
            client_id=cid, run_seed=run_seed,
        )
        client = DecentralizedClient(
            client_id=cid,
            model=model_copy,
            dataset=dataset,
            device=device,
            lr=cfg.training.optimizer.lr,
            momentum=cfg.training.optimizer.momentum,
            weight_decay=cfg.training.optimizer.weight_decay,
            batch_size=cfg.training.batch_size,
            local_steps=cfg.training.local_steps,
        )
        clients.append(client)

    lr_schedule = CosineDecaySchedule(
        lr_max=cfg.training.optimizer.lr,
        lr_min=cfg.training.lr_schedule.lr_min,
        total_rounds=cfg.training.total_rounds,
        warmup_rounds=cfg.training.lr_schedule.warmup_rounds,
    )

    csv_logger = CSVLogger(str(output_dir / "metrics.csv"))

    # ---- Evaluator ----
    probe_set = loader.get_probe_set()
    probe_loader = DataLoader(
        probe_set,
        batch_size=loader.batch_size,
        shuffle=False,
        num_workers=0,
        worker_init_fn=seed_worker,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=loader.batch_size,
        shuffle=False,
        num_workers=0,
    )

    extractor = RepresentationExtractor(layer_names=layer_names, device=str(device))
    cka = CKAAnalyzer()
    rsa = RSAAnalyzer()
    mmd = MMDAnalyzer(n_permutations=200, seed=0)
    fisher = FisherAnalyzer()
    centroid = CentroidAnalyzer()
    drift_tracker = DriftTracker(
        output_path=str(output_dir / "drift_metrics.h5"),
        layer_names=layer_names,
    )

    client_id_to_community = _build_client_id_to_community(community_map)
    evaluator = ExperimentEvaluator(
        extractor=extractor,
        cka=cka,
        rsa=rsa,
        mmd=mmd,
        fisher=fisher,
        centroid=centroid,
        drift_tracker=drift_tracker,
        probe_loader=probe_loader,
        community_map=client_id_to_community,
        test_loader=test_loader,
        primary_layer=layer_names[-2],
        device=str(device),
    )

    checkpoint_mgr = CheckpointManager(
        output_dir=str(output_dir / "checkpoints"),
        community_assignments=community_arr,
    )

    metadata = MetadataStore(str(output_dir / "metadata.json"))
    metadata.collect(run_seed=run_seed, graph_seed=graph_seed,
                     config=OmegaConf.to_container(cfg, resolve=True))
    metadata.save()

    # ---- Coordinator ----
    algorithm = cfg.training.get("algorithm", "dsgd")
    if algorithm == "fedavg":
        coordinator = FedAvgCoordinator(
            clients=clients, lr_schedule=lr_schedule,
            total_rounds=cfg.training.total_rounds,
            eval_every=cfg.training.eval_every,
            checkpoint_every=cfg.training.checkpoint_every,
            output_dir=str(output_dir),
            evaluator=evaluator,
            loggers=[csv_logger],
            checkpoint_manager=checkpoint_mgr,
        )
    elif algorithm == "local_only":
        coordinator = LocalOnlyCoordinator(
            clients=clients, lr_schedule=lr_schedule,
            total_rounds=cfg.training.total_rounds,
            eval_every=cfg.training.eval_every,
            checkpoint_every=cfg.training.checkpoint_every,
            output_dir=str(output_dir),
            evaluator=evaluator,
            loggers=[csv_logger],
            checkpoint_manager=checkpoint_mgr,
        )
    else:
        coordinator = DSGDCoordinator(
            clients=clients, mixing_matrix=W,
            lr_schedule=lr_schedule,
            total_rounds=cfg.training.total_rounds,
            eval_every=cfg.training.eval_every,
            checkpoint_every=cfg.training.checkpoint_every,
            output_dir=str(output_dir),
            evaluator=evaluator,
            loggers=[csv_logger],
            checkpoint_manager=checkpoint_mgr,
        )

    history = coordinator.run()
    csv_logger.close()
    drift_tracker.save()

    # Extract final-round drift metrics from the tracker for the summary.
    primary_layer = layer_names[-2]
    final_drift_metrics = {}
    for key in sorted(drift_tracker._recorded_keys):
        trajectory = drift_tracker.get_trajectory(key)
        if trajectory.ndim == 1 and len(trajectory) > 0:
            final_drift_metrics[f"final_{key}"] = float(trajectory[-1])
    # Alias the primary-layer CKA mean as the canonical analysis column.
    cka_key = f"cka_{primary_layer}_mean"
    if cka_key in drift_tracker._recorded_keys:
        traj = drift_tracker.get_trajectory(cka_key)
        if len(traj) > 0:
            final_drift_metrics["final_cka_cross_community"] = float(traj[-1])

    summary = {
        "experiment_name": cfg.experiment_name,
        "run_seed": run_seed,
        "graph_seed": graph_seed,
        "data_regime": cfg.data.partition_method,
        "dirichlet_alpha": cfg.data.dirichlet_alpha,
        "network_regime": cfg.graph.topology,
        "dataset": cfg.data.dataset,
        "model": cfg.model.architecture,
        "algorithm": algorithm,
        "total_rounds": cfg.training.total_rounds,
        "final_loss": history["mean_loss"][-1] if history["mean_loss"] else None,
        "wall_time": time.time() - t_start,
        **graph_metrics,
        **final_drift_metrics,
    }
    save_json(summary, output_dir / "summary.json")

    logger.info("Experiment complete. Wall time: %.1fs", summary["wall_time"])
    logger.info("Output: %s", output_dir)

    return summary


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    run(cfg)


if __name__ == "__main__":
    main()
