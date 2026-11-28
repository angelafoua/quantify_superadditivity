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
from superadditivity.evaluation.cka_analyzer import CKAAnalyzer, ClientCKAAnalyzer
from superadditivity.evaluation.rsa_analyzer import RSAAnalyzer
from superadditivity.evaluation.mmd_analyzer import MMDAnalyzer
from superadditivity.evaluation.paired_mmd_analyzer import PairedMMDAnalyzer
from superadditivity.evaluation.fisher_analyzer import FisherAnalyzer
from superadditivity.evaluation.centroid_analyzer import CentroidAnalyzer
from superadditivity.evaluation.drift_tracker import DriftTracker
from superadditivity.analysis.drift_metrics import PRIMARY_ALIASES
from superadditivity.logging.csv_logger import CSVLogger
from superadditivity.logging.metadata_store import MetadataStore
from superadditivity.logging.checkpoint_manager import (
    CheckpointManager,
    TrainingCheckpointManager,
    _config_hash,
)
from superadditivity.utils.seed import set_all_seeds, derived_seed, seed_worker
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
    """Execute a single experiment run.

    If a resume checkpoint exists and is compatible with the current config,
    training resumes from the saved round instead of restarting.
    """
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
    partition_stats: dict = {}
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
            between_community=cfg.data.get("between_community", "dirichlet"),
            alpha_within=cfg.data.get("alpha_within", None),
        )
        partition_result = partitioner.partition()
        partition_stats["community_label_tv"] = partitioner.community_label_distance(
            partition_result
        )
        partition_stats["within_community_label_tv"] = (
            partitioner.within_community_label_distance(partition_result)
        )
        partition_stats["alpha_within"] = cfg.data.get("alpha_within", None)
        partition_stats["between_community"] = (
            cfg.data.get("between_community", "dirichlet")
            if cfg.data.partition_method == "dirichlet_semantic"
            else None
        )
        logger.info(
            "Mean total-variation distance between community label "
            "distributions: %.3f", partition_stats["community_label_tv"],
        )
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
    eval_cfg = cfg.get("evaluation", {}) or {}
    legacy_cfg = eval_cfg.get("legacy_community_mmd", {}) or {}
    mmd = (
        MMDAnalyzer(n_permutations=int(legacy_cfg.get("n_permutations", 200)), seed=0)
        if legacy_cfg.get("enabled", False)
        else None
    )
    pmmd_cfg = eval_cfg.get("paired_mmd", {}) or {}
    paired_mmd = None
    paired_mmd_layers = None
    if pmmd_cfg.get("enabled", True):
        paired_mmd = PairedMMDAnalyzer(
            bandwidth_multipliers=list(
                pmmd_cfg.get("bandwidth_multipliers", [0.25, 0.5, 1.0, 2.0, 4.0])
            ),
            bandwidth_subsample=int(pmmd_cfg.get("bandwidth_subsample", 2000)),
            bandwidth_seed=int(pmmd_cfg.get("bandwidth_seed", 0)),
            n_permutations=int(pmmd_cfg.get("n_permutations", 10000)),
            permutation_seed=int(pmmd_cfg.get("permutation_seed", 0)),
            device=str(device),
            bandwidth_reference=str(pmmd_cfg.get("bandwidth_reference", "within_client")),
        )
        layers_cfg = pmmd_cfg.get("layers", None)
        paired_mmd_layers = list(layers_cfg) if layers_cfg is not None else None
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
        paired_mmd=paired_mmd,
        paired_mmd_layers=paired_mmd_layers,
        accuracy_every=(
            int(eval_cfg["accuracy_every"])
            if eval_cfg.get("accuracy_every") is not None else None
        ),
        final_round=cfg.training.total_rounds - 1,
        client_cka=(
            ClientCKAAnalyzer(device=str(device))
            if (eval_cfg.get("client_cka", {}) or {}).get("enabled", True)
            else None
        ),
        client_cka_layers=(
            list((eval_cfg.get("client_cka", {}) or {})["layers"])
            if (eval_cfg.get("client_cka", {}) or {}).get("layers") is not None
            else None
        ),
    )

    checkpoint_mgr = CheckpointManager(
        output_dir=str(output_dir / "checkpoints"),
        community_assignments=community_arr,
    )

    # ---- Resume manager ----
    config_dict = OmegaConf.to_container(cfg, resolve=True)
    resume_mgr = TrainingCheckpointManager(
        output_dir=str(output_dir / "resume_checkpoints"),
        config_dict=config_dict,
    )

    metadata = MetadataStore(str(output_dir / "metadata.json"))
    metadata.collect(run_seed=run_seed, graph_seed=graph_seed,
                     config=config_dict)
    metadata.save()

    # ---- Coordinator ----
    algorithm = cfg.training.get("algorithm", "dsgd")
    coordinator_kwargs = dict(
        clients=clients, lr_schedule=lr_schedule,
        total_rounds=cfg.training.total_rounds,
        eval_every=cfg.training.eval_every,
        checkpoint_every=cfg.training.checkpoint_every,
        output_dir=str(output_dir),
        evaluator=evaluator,
        loggers=[csv_logger],
        checkpoint_manager=checkpoint_mgr,
        resume_manager=resume_mgr,
        drift_tracker=drift_tracker,
    )

    if algorithm == "fedavg":
        coordinator = FedAvgCoordinator(**coordinator_kwargs)
    elif algorithm == "local_only":
        coordinator = LocalOnlyCoordinator(**coordinator_kwargs)
    else:
        coordinator = DSGDCoordinator(
            mixing_matrix=W,
            mix_buffers=bool(cfg.training.get("mix_bn_buffers", False)),
            **coordinator_kwargs,
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
    # Mean over the evaluations in the last `summary_window_rounds` rounds:
    # less noisy than the final evaluation alone, and the same time span
    # for every metric whatever its evaluation frequency.
    window_rounds = int(
        (cfg.get("evaluation", {}) or {}).get("summary_window_rounds", 50)
    )
    for key in sorted(drift_tracker._recorded_keys):
        trajectory = drift_tracker.get_trajectory(key)
        if trajectory.ndim == 1 and len(trajectory) > 0:
            rounds = drift_tracker.get_rounds(key)
            in_window = rounds >= rounds[-1] - window_rounds
            final_drift_metrics[f"window_{key}"] = float(np.mean(trajectory[in_window]))
    # Canonical primary-layer columns used by the analysis.
    for alias, pattern in PRIMARY_ALIASES.items():
        key = pattern.format(layer=primary_layer)
        if f"final_{key}" in final_drift_metrics:
            final_drift_metrics[f"final_{alias}"] = final_drift_metrics[f"final_{key}"]
            final_drift_metrics[f"window_{alias}"] = final_drift_metrics[f"window_{key}"]

    summary = {
        "experiment_name": cfg.experiment_name,
        "config_hash": _config_hash(config_dict),
        "run_seed": run_seed,
        "graph_seed": graph_seed,
        "data_regime": cfg.data.partition_method,
        "dirichlet_alpha": cfg.data.dirichlet_alpha,
        "network_regime": cfg.graph.topology,
        "mix_bn_buffers": bool(cfg.training.get("mix_bn_buffers", False)),
        # Graph-level parameters, so analyses can tell SBM levels apart
        # (network_regime is only the topology name).
        "p_in": cfg.graph.get("p_in", None),
        "p_out": cfg.graph.get("p_out", None),
        "primary_layer": primary_layer,
        "dataset": cfg.data.dataset,
        "model": cfg.model.architecture,
        "algorithm": algorithm,
        "total_rounds": cfg.training.total_rounds,
        "final_loss": history["mean_loss"][-1] if history["mean_loss"] else None,
        "wall_time": time.time() - t_start,
        **graph_metrics,
        **partition_stats,
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
