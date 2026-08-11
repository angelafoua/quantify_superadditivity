"""Run a full sweep of experiments for a given experiment config.

Usage:
    python scripts/run_sweep.py --experiment core_factorial
    python scripts/run_sweep.py --experiment extended_grid
    python scripts/run_sweep.py --experiment pout_sweep
"""

from __future__ import annotations

import argparse
import itertools
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from omegaconf import OmegaConf

from superadditivity.utils.io import load_json, save_json, ensure_dir

logger = logging.getLogger(__name__)


def run_core_factorial(cfg: dict) -> None:
    """Run all cells of the core 2×2 factorial."""
    from scripts.run_experiment import run

    for cell_name, cell_cfg in cfg["cells"].items():
        for run_seed in cfg["run_seeds"]:
            for graph_seed in cfg["graph_seeds"]:
                logger.info(
                    "Running cell=%s, run_seed=%d, graph_seed=%d",
                    cell_name, run_seed, graph_seed,
                )

                overrides = [
                    f"data={cell_cfg['data']}",
                    f"graph={cell_cfg['graph']}",
                    f"model={cfg['model']}",
                    f"training={cfg['training']}",
                    f"run_seed={run_seed}",
                    f"graph_seed={graph_seed}",
                    f"experiment_name={cfg['experiment_name']}_{cell_name}",
                ]
                run_cfg = _build_config(overrides)
                try:
                    run(run_cfg)
                except Exception as e:
                    logger.error("Failed: %s (cell=%s, rs=%d, gs=%d)", e, cell_name, run_seed, graph_seed)


def run_extended_grid(cfg: dict) -> None:
    """Run all cells of the extended 4×4 factorial."""
    from scripts.run_experiment import run

    for data_level in cfg["data_levels"]:
        for network_level in cfg["network_levels"]:
            for run_seed in cfg["run_seeds"]:
                for graph_seed in cfg["graph_seeds"]:
                    logger.info(
                        "Running data=%s, network=%s, rs=%d, gs=%d",
                        data_level, network_level, run_seed, graph_seed,
                    )
                    overrides = [
                        f"data={data_level}",
                        f"graph={network_level}",
                        f"model={cfg['model']}",
                        f"training={cfg['training']}",
                        f"run_seed={run_seed}",
                        f"graph_seed={graph_seed}",
                        f"experiment_name={cfg['experiment_name']}_{data_level}_{network_level}",
                    ]
                    run_cfg = _build_config(overrides)
                    try:
                        run(run_cfg)
                    except Exception as e:
                        logger.error("Failed: %s", e)


def run_pout_sweep(cfg: dict) -> None:
    """Run parametric p_out sweep."""
    from scripts.run_experiment import run

    for data_level in cfg["data_levels"]:
        for p_out in cfg["p_out_values"]:
            for run_seed in cfg["run_seeds"]:
                for graph_seed in cfg["graph_seeds"]:
                    logger.info(
                        "Running data=%s, p_out=%.3f, rs=%d, gs=%d",
                        data_level, p_out, run_seed, graph_seed,
                    )
                    overrides = [
                        f"data={data_level}",
                        "graph=sbm_medium",
                        f"graph.p_out={p_out}",
                        f"graph.p_in={cfg['p_in']}",
                        f"model={cfg['model']}",
                        f"training={cfg['training']}",
                        f"run_seed={run_seed}",
                        f"graph_seed={graph_seed}",
                        f"experiment_name={cfg['experiment_name']}_pout{p_out}_{data_level}",
                    ]
                    run_cfg = _build_config(overrides)
                    try:
                        run(run_cfg)
                    except Exception as e:
                        logger.error("Failed: %s", e)


def _build_config(overrides: list) -> "DictConfig":
    """Build a Hydra config from overrides."""
    from hydra import compose, initialize_config_dir

    config_dir = str(Path(__file__).resolve().parent.parent / "configs")
    with initialize_config_dir(version_base=None, config_dir=config_dir):
        cfg = compose(config_name="config", overrides=overrides)
    return cfg


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    parser = argparse.ArgumentParser(description="Run experiment sweep")
    parser.add_argument(
        "--experiment", required=True,
        choices=["core_factorial", "extended_grid", "pout_sweep", "robustness_check"],
        help="Experiment configuration to run.",
    )
    args = parser.parse_args()

    config_path = (
        Path(__file__).resolve().parent.parent
        / "configs" / "experiment" / f"{args.experiment}.yaml"
    )
    cfg = OmegaConf.to_container(OmegaConf.load(config_path), resolve=True)

    if args.experiment == "core_factorial":
        run_core_factorial(cfg)
    elif args.experiment == "extended_grid":
        run_extended_grid(cfg)
    elif args.experiment == "pout_sweep":
        run_pout_sweep(cfg)
    elif args.experiment == "robustness_check":
        logger.info("Robustness check: running each sub-experiment...")
        for check_name, check_cfg in cfg["checks"].items():
            logger.info("Sub-experiment: %s", check_name)
            for data_level in check_cfg["data_levels"]:
                for network_level in check_cfg["network_levels"]:
                    for run_seed in cfg["run_seeds"]:
                        for graph_seed in cfg["graph_seeds"]:
                            overrides = [
                                f"data={data_level}",
                                f"graph={network_level}",
                                f"model={check_cfg['model']}",
                                f"training={cfg['training']}",
                                f"run_seed={run_seed}",
                                f"graph_seed={graph_seed}",
                                f"experiment_name=robustness_{check_name}",
                            ]
                            run_cfg = _build_config(overrides)
                            try:
                                from scripts.run_experiment import run
                                run(run_cfg)
                            except Exception as e:
                                logger.error("Failed: %s", e)

    logger.info("Sweep complete for experiment: %s", args.experiment)


if __name__ == "__main__":
    main()
