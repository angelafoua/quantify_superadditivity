"""Run only the data-partition step for one seed, one data regime, one network.

Mirrors the dataset -> graph -> partition part of ``scripts/run_experiment.py``
and skips the model, training and evaluation. Prints a per-client report and
writes it to ``outputs/partition_only/<data>_<graph>_rs<run>_gs<graph>/``.

Usage (from the repository root):
    python scripts/partition_only.py --data iid --graph erdos_renyi --run-seed 42 --graph-seed 100
    python scripts/partition_only.py --data moderate_noniid --graph sbm_medium --run-seed 42 --graph-seed 100
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from omegaconf import OmegaConf  # noqa: E402

from scripts.run_sweep import _build_config  # noqa: E402
from superadditivity.datasets.dataset_loader import DatasetLoader  # noqa: E402
from superadditivity.datasets.quantity_skew_partitioner import QuantitySkewPartitioner  # noqa: E402
from superadditivity.datasets.semantic_partitioner import SemanticPartitioner  # noqa: E402
from superadditivity.graphs.graph_manager import GraphManager  # noqa: E402
from superadditivity.utils.io import ensure_dir, save_json  # noqa: E402
from superadditivity.utils.seed import derived_seed, set_all_seeds  # noqa: E402

logger = logging.getLogger("partition_only")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", required=True, help="configs/data/<name>.yaml, e.g. iid, moderate_noniid, quantity_skew")
    parser.add_argument("--graph", required=True, help="configs/graph/<name>.yaml, e.g. erdos_renyi, sbm_medium")
    parser.add_argument("--run-seed", type=int, default=42)
    parser.add_argument("--graph-seed", type=int, default=100)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    # Same config composition the sweep uses, so values match a real run.
    cfg = _build_config([
        f"data={args.data}",
        f"graph={args.graph}",
        f"run_seed={args.run_seed}",
        f"graph_seed={args.graph_seed}",
    ])
    run_seed, graph_seed = cfg.run_seed, cfg.graph_seed
    n_clients, n_communities = cfg.n_clients, cfg.n_communities
    if n_clients % n_communities != 0:
        raise SystemExit(f"n_clients ({n_clients}) must be a multiple of n_communities ({n_communities}).")

    out_dir = ensure_dir(
        Path("outputs/partition_only")
        / f"{args.data}_{args.graph}_rs{run_seed}_gs{graph_seed}"
    )

    # 1. Dataset (same call as run(); downloads CIFAR-100 if missing and download=true).
    set_all_seeds(run_seed)
    loader = DatasetLoader(
        dataset_name=cfg.data.dataset,
        data_dir=cfg.data.data_dir,
        download=cfg.data.download,
    )
    train_dataset, _ = loader.load()
    targets = np.array(DatasetLoader._get_targets(train_dataset))
    n_classes = loader.get_num_classes()

    # 2. Graph (only needed for the communities used by the semantic partitioner).
    graph_manager = GraphManager(cfg.graph, graph_seed=graph_seed)
    graph_manager.build()
    community_map = graph_manager.get_communities()
    graph_metrics = graph_manager.get_metrics()

    # 3. Partition (same branching as run(), lines 197-227).
    method = cfg.data.partition_method
    if method == "quantity_skew":
        qty = QuantitySkewPartitioner(n_clients=n_clients, alpha=cfg.data.dirichlet_alpha or 1.0)
        result = qty.partition(targets, n_communities=n_communities,
                               seed=derived_seed(run_seed, kind="dirichlet"))
        client_indices = result["client_indices"]
        community_arr = result["community_assignments"]
    else:
        partitioner = SemanticPartitioner(
            train_dataset=train_dataset,
            semantic_clusters=loader.get_semantic_clusters(),
            community_assignments=community_map,
            alpha=cfg.data.dirichlet_alpha or 1.0,
            mode=method,
            run_seed=run_seed,
            num_clients=n_clients,
        )
        partition = partitioner.partition()
        client_indices = [partition[i] for i in range(n_clients)]
        community_arr = np.zeros(n_clients, dtype=np.int64)
        for comm_id, cids in community_map.items():
            for cid in cids:
                community_arr[cid] = comm_id

    # 4. Report: per-client size, class coverage and skew.
    clients = []
    for cid, idx in enumerate(client_indices):
        idx = np.asarray(idx, dtype=np.int64)
        counts = np.bincount(targets[idx], minlength=n_classes) if len(idx) else np.zeros(n_classes, dtype=np.int64)
        n = int(len(idx))
        clients.append({
            "client": cid,
            "community": int(community_arr[cid]),
            "n_samples": n,
            "n_distinct_classes": int((counts > 0).sum()),
            "top_class_share": float(counts.max() / n) if n else 0.0,
        })

    sizes = np.array([c["n_samples"] for c in clients])
    all_idx = np.concatenate([np.asarray(i) for i in client_indices])
    summary = {
        "data": args.data,
        "graph": args.graph,
        "partition_method": method,
        "dirichlet_alpha": cfg.data.dirichlet_alpha,
        "run_seed": run_seed,
        "graph_seed": graph_seed,
        "n_clients": n_clients,
        "n_communities": n_communities,
        "total_samples_assigned": int(len(all_idx)),
        "unique_samples_assigned": int(len(np.unique(all_idx))),
        "samples_per_client_min": int(sizes.min()),
        "samples_per_client_max": int(sizes.max()),
        "samples_per_client_mean": float(sizes.mean()),
        "mean_distinct_classes_per_client": float(np.mean([c["n_distinct_classes"] for c in clients])),
        "mean_top_class_share": float(np.mean([c["top_class_share"] for c in clients])),
        **{f"graph_{k}": float(v) for k, v in graph_metrics.items()},
        "clients": clients,
    }
    save_json(summary, out_dir / "partition_summary.json")

    print()
    print(f"Data: {args.data} ({method}, alpha={cfg.data.dirichlet_alpha}) | Graph: {args.graph} | run_seed={run_seed} graph_seed={graph_seed}")
    print(f"Samples assigned: {summary['total_samples_assigned']} (unique {summary['unique_samples_assigned']}) of {len(targets)}")
    print(f"Samples per client: min {summary['samples_per_client_min']}, max {summary['samples_per_client_max']}, mean {summary['samples_per_client_mean']:.1f}")
    print(f"Mean distinct classes per client: {summary['mean_distinct_classes_per_client']:.1f} of {n_classes}")
    print(f"Mean share of a client's top class: {summary['mean_top_class_share']:.3f} (IID is about 0.01 to 0.02)")
    print()
    print(f"{'client':>6} {'comm':>4} {'samples':>8} {'classes':>8} {'top_share':>10}")
    for c in clients:
        print(f"{c['client']:>6} {c['community']:>4} {c['n_samples']:>8} {c['n_distinct_classes']:>8} {c['top_class_share']:>10.3f}")
    print()
    print(f"Report written to: {(out_dir / 'partition_summary.json').resolve()}")


if __name__ == "__main__":
    main()
