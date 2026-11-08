"""Run a full sweep of experiments for a given experiment config.

A failed experiment **stops the entire sweep** and exits with a nonzero
status code.  Completed runs are skipped, interrupted runs are resumed
from checkpoint, but no failure is swallowed.

Usage:
    python scripts/run_sweep.py --experiment core_factorial
    python scripts/run_sweep.py --experiment extended_grid
    python scripts/run_sweep.py --experiment pout_sweep
    python scripts/run_sweep.py --experiment pout_sweep_domainnet --resume
"""

from __future__ import annotations

import argparse
import logging
import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from omegaconf import OmegaConf

from superadditivity.logging.checkpoint_manager import _config_hash
from superadditivity.utils.io import load_json, save_json, ensure_dir

logger = logging.getLogger(__name__)


def _run_dir(run_cfg: "DictConfig") -> Path:
    """Return the run-specific output directory for a config."""
    return (
        Path(run_cfg.output_dir)
        / f"seed_{run_cfg.run_seed}_graph_{run_cfg.graph_seed}"
    )


def _is_complete(run_cfg: "DictConfig") -> bool:
    """Return True if this run already finished with a compatible config.

    Validates that ``summary.json`` exists, is parseable, and its config hash
    and ``total_rounds`` match the current configuration.  Also checks that
    ``drift_metrics.h5`` was written (guards against interrupted final saves).
    """
    run_dir = _run_dir(run_cfg)
    summary = run_dir / "summary.json"
    if not summary.exists():
        return False
    try:
        data = load_json(summary)
        if not isinstance(data, dict):
            return False
        if data.get("total_rounds") != run_cfg.training.total_rounds:
            return False
        saved_hash = data.get("config_hash")
        if saved_hash:
            current_hash = _config_hash(
                OmegaConf.to_container(run_cfg, resolve=True)
            )
            if saved_hash != current_hash:
                logger.warning(
                    "Summary config hash mismatch in %s "
                    "(saved=%s, current=%s). Re-running.",
                    run_dir, saved_hash, current_hash,
                )
                return False
        if not (run_dir / "drift_metrics.h5").exists():
            logger.warning(
                "summary.json exists but drift_metrics.h5 is missing in %s. "
                "Treating as incomplete.",
                run_dir,
            )
            return False
        return True
    except Exception:
        return False


def _has_resume_checkpoint(run_cfg: "DictConfig") -> bool:
    """Return True if the run directory contains a resume checkpoint."""
    resume_dir = _run_dir(run_cfg) / "resume_checkpoints"
    if not resume_dir.is_dir():
        return False
    return any(resume_dir.glob("resume_ckpt_r*.pt"))


def _run_status(run_cfg: "DictConfig") -> str:
    """Classify a run as 'complete', 'interrupted', or 'new'."""
    if _is_complete(run_cfg):
        return "complete"
    if _has_resume_checkpoint(run_cfg):
        return "interrupted"
    return "new"


def _write_error_log(run_cfg: "DictConfig", label: str, exc: Exception) -> None:
    """Write a structured error log to the run directory and to the console."""
    run_dir = _run_dir(run_cfg)
    ensure_dir(run_dir)
    config_dict = OmegaConf.to_container(run_cfg, resolve=True)

    error_info = {
        "experiment": label,
        "config_hash": _config_hash(config_dict),
        "run_dir": str(run_dir),
        "error_type": type(exc).__name__,
        "error_message": str(exc),
        "traceback": traceback.format_exc(),
    }

    error_path = run_dir / "error.json"
    save_json(error_info, error_path)

    logger.error(
        "SWEEP STOPPED — experiment failed.\n"
        "  Experiment: %s\n"
        "  Config hash: %s\n"
        "  Run dir: %s\n"
        "  Error: %s: %s\n"
        "  Error log: %s\n"
        "  Next step: fix the issue, then re-run with --resume.",
        label, error_info["config_hash"], run_dir,
        error_info["error_type"], error_info["error_message"],
        error_path,
    )


class SweepFailedError(Exception):
    """Raised when an experiment in a sweep fails, stopping the sweep."""


def _write_manifest(experiment_name: str, run_dirs: list) -> None:
    """Write a manifest listing all run directories for a sweep.

    This lets the analysis aggregator locate results by direct path
    instead of relying on recursive directory traversal, which can
    fail on FUSE-mounted filesystems like Google Drive in Colab.
    """
    manifest_path = Path("outputs") / experiment_name / "sweep_manifest.json"
    ensure_dir(manifest_path.parent)
    save_json(
        {"experiment_name": experiment_name, "run_dirs": [str(d) for d in run_dirs]},
        manifest_path,
    )
    logger.info("Wrote sweep manifest: %s (%d runs)", manifest_path, len(run_dirs))


def _run_single(run_cfg: "DictConfig", label: str) -> None:
    """Run a single experiment, converting any failure to a SweepFailedError."""
    from scripts.run_experiment import run

    try:
        run(run_cfg)
    except Exception as exc:
        _write_error_log(run_cfg, label, exc)
        raise SweepFailedError(
            f"Experiment failed: {label} — {type(exc).__name__}: {exc}"
        ) from exc


def run_core_factorial(cfg: dict, resume: bool = False) -> None:
    """Run all cells of the core 2x2 factorial."""
    all_run_dirs: list = []
    for cell_name, cell_cfg in cfg["cells"].items():
        for run_seed in cfg["run_seeds"]:
            for graph_seed in cfg["graph_seeds"]:
                label = f"cell={cell_name}, rs={run_seed}, gs={graph_seed}"
                logger.info("Running %s", label)

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
                all_run_dirs.append(_run_dir(run_cfg))
                status = _run_status(run_cfg) if resume else "new"
                if status == "complete":
                    logger.info("SKIP (complete): %s", _run_dir(run_cfg))
                    continue
                if status == "interrupted":
                    logger.info("RESUME (from checkpoint): %s", _run_dir(run_cfg))
                else:
                    logger.info("START (new): %s", _run_dir(run_cfg))
                _run_single(run_cfg, label)
    _write_manifest(cfg["experiment_name"], all_run_dirs)


def run_extended_grid(cfg: dict, resume: bool = False) -> None:
    """Run all cells of the extended 4x4 factorial."""
    all_run_dirs: list = []
    for data_level in cfg["data_levels"]:
        for network_level in cfg["network_levels"]:
            for run_seed in cfg["run_seeds"]:
                for graph_seed in cfg["graph_seeds"]:
                    label = (
                        f"data={data_level}, network={network_level}, "
                        f"rs={run_seed}, gs={graph_seed}"
                    )
                    logger.info("Running %s", label)
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
                    all_run_dirs.append(_run_dir(run_cfg))
                    status = _run_status(run_cfg) if resume else "new"
                    if status == "complete":
                        logger.info("SKIP (complete): %s", _run_dir(run_cfg))
                        continue
                    if status == "interrupted":
                        logger.info("RESUME (from checkpoint): %s", _run_dir(run_cfg))
                    else:
                        logger.info("START (new): %s", _run_dir(run_cfg))
                    _run_single(run_cfg, label)
    _write_manifest(cfg["experiment_name"], all_run_dirs)


def run_pout_sweep(cfg: dict, resume: bool = False) -> None:
    """Run parametric p_out sweep."""
    total = (
        len(cfg["data_levels"]) * len(cfg["p_out_values"])
        * len(cfg["run_seeds"]) * len(cfg["graph_seeds"])
    )
    done = 0
    skipped = 0
    all_run_dirs: list = []

    for data_level in cfg["data_levels"]:
        for p_out in cfg["p_out_values"]:
            for run_seed in cfg["run_seeds"]:
                for graph_seed in cfg["graph_seeds"]:
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
                    all_run_dirs.append(_run_dir(run_cfg))
                    status = _run_status(run_cfg) if resume else "new"

                    label = (
                        f"data={data_level}, p_out={p_out:.3f}, "
                        f"rs={run_seed}, gs={graph_seed}"
                    )

                    if status == "complete":
                        skipped += 1
                        logger.info(
                            "[%d/%d] SKIP (complete): %s",
                            skipped + done, total, label,
                        )
                        continue

                    if status == "interrupted":
                        logger.info(
                            "[%d/%d] RESUME (from checkpoint): %s",
                            skipped + done + 1, total, label,
                        )
                    else:
                        logger.info(
                            "[%d/%d] START (new): %s",
                            skipped + done + 1, total, label,
                        )

                    _run_single(run_cfg, label)
                    done += 1

    logger.info(
        "Sweep progress: %d skipped, %d completed, %d total of %d",
        skipped, done, skipped + done, total,
    )
    _write_manifest(cfg["experiment_name"], all_run_dirs)


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
        help="Experiment configuration to run (must match a YAML in configs/experiment/).",
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Skip completed runs and resume interrupted ones from checkpoint.",
    )
    args = parser.parse_args()

    config_path = (
        Path(__file__).resolve().parent.parent
        / "configs" / "experiment" / f"{args.experiment}.yaml"
    )
    if not config_path.exists():
        logger.error("Config not found: %s", config_path)
        sys.exit(1)
    cfg = OmegaConf.to_container(OmegaConf.load(config_path), resolve=True)

    try:
        if args.experiment.startswith("core_factorial"):
            run_core_factorial(cfg, resume=args.resume)
        elif args.experiment == "extended_grid":
            run_extended_grid(cfg, resume=args.resume)
        elif args.experiment.startswith("pout_sweep"):
            run_pout_sweep(cfg, resume=args.resume)
        elif args.experiment == "robustness_check":
            logger.info("Robustness check: running each sub-experiment...")
            from scripts.run_experiment import run as run_experiment

            for check_name, check_cfg in cfg["checks"].items():
                logger.info("Sub-experiment: %s", check_name)
                for data_level in check_cfg["data_levels"]:
                    for network_level in check_cfg["network_levels"]:
                        for run_seed in cfg["run_seeds"]:
                            for graph_seed in cfg["graph_seeds"]:
                                label = (
                                    f"robustness_{check_name}, data={data_level}, "
                                    f"net={network_level}, rs={run_seed}, gs={graph_seed}"
                                )
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
                                status = _run_status(run_cfg) if args.resume else "new"
                                if status == "complete":
                                    logger.info("SKIP (complete): %s", _run_dir(run_cfg))
                                    continue
                                if status == "interrupted":
                                    logger.info("RESUME (from checkpoint): %s", _run_dir(run_cfg))
                                else:
                                    logger.info("START (new): %s", _run_dir(run_cfg))
                                _run_single(run_cfg, label)
    except SweepFailedError:
        sys.exit(1)

    logger.info("Sweep complete for experiment: %s", args.experiment)


if __name__ == "__main__":
    main()
