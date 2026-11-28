"""Generate publication-quality figures from experiment results.

Usage:
    python scripts/run_visualization.py --results_dir outputs/core_factorial
    python scripts/run_visualization.py --results_dir outputs/core_factorial --metric mmd
    python scripts/run_visualization.py --results_dir outputs/core_factorial --force

Figures show *drift* (higher = more drift): CKA as ``1 - CKA``, MMD as is
(see superadditivity/analysis/drift_metrics.py). Output goes to
``<results_dir>/figures`` for CKA and ``<results_dir>/figures_<metric>``
otherwise. Statistics shown on the interaction bar chart (CI, p-value) are
read from the matching ``analysis[_<metric>]/interaction_test.json`` when
it exists.
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from superadditivity.analysis.aggregator import ResultAggregator
from superadditivity.analysis.drift_metrics import (
    CELL_LABELS,
    DEFAULT_METRIC,
    DRIFT_METRICS,
    ORIENTATIONS,
    DriftMetric,
    assign_factorial_cells,
    output_subdir,
    resolve_drift_metric,
)
from superadditivity.analysis.interaction_analyzer import InteractionAnalyzer
from superadditivity.utils.io import ensure_dir, load_json
from superadditivity.visualization.figure_style import set_publication_style

logger = logging.getLogger(__name__)


def _find_summaries(results_dir: Path) -> list:
    """Find summary.json files in nested or sibling-directory layouts."""
    found = sorted(results_dir.rglob("summary.json"))
    if found:
        return found
    parent = results_dir.parent
    prefix = results_dir.name + "_"
    if parent.is_dir():
        for sibling in sorted(parent.iterdir()):
            if sibling.is_dir() and sibling.name.startswith(prefix):
                found.extend(sorted(sibling.rglob("summary.json")))
    return found


def _results_fingerprint(
    results_dir: Path, metric: Optional[DriftMetric] = None, layer: str = "layer4"
) -> str:
    """Hash summaries, analysis JSON and the metric to detect stale figures."""
    metric = metric or resolve_drift_metric()
    md5 = hashlib.md5()
    md5.update(f"{metric.column}|{metric.orientation}|{layer}".encode())
    for p in _find_summaries(results_dir):
        try:
            md5.update(p.read_bytes())
        except OSError:
            pass
    analysis_dir = results_dir / output_subdir(metric, "analysis")
    if analysis_dir.is_dir():
        for p in sorted(analysis_dir.glob("*.json")):
            md5.update(p.read_bytes())
    return md5.hexdigest()


_EXPECTED_FIGURE_FILES = [
    "factorial_grid.pdf",
    "interaction_bar.pdf",
    "drift_trajectories.pdf",
]


def _is_viz_current(
    fig_dir: Path,
    results_dir: Path,
    metric: Optional[DriftMetric] = None,
    layer: str = "layer4",
) -> bool:
    """Return True if figures are up to date and at least one output exists."""
    stamp = fig_dir / ".viz_fingerprint"
    if not stamp.exists():
        return False
    try:
        if stamp.read_text().strip() != _results_fingerprint(results_dir, metric, layer):
            return False
    except Exception:
        return False
    if not any((fig_dir / f).exists() for f in _EXPECTED_FIGURE_FILES):
        logger.info(
            "Fingerprint matches but no figure files found in %s. "
            "Regenerating.", fig_dir,
        )
        return False
    return True


def _write_viz_fingerprint(
    fig_dir: Path,
    results_dir: Path,
    metric: Optional[DriftMetric] = None,
    layer: str = "layer4",
) -> None:
    stamp = fig_dir / ".viz_fingerprint"
    stamp.write_text(_results_fingerprint(results_dir, metric, layer))


def _invalidate_viz_fingerprint(fig_dir: Path) -> None:
    """Remove stale fingerprint so a partial re-run is never mistaken for current."""
    stamp = fig_dir / ".viz_fingerprint"
    if stamp.exists():
        stamp.unlink()


def plot_interaction_bar_figure(
    agg: ResultAggregator,
    metric: DriftMetric,
    results_dir: Path,
    save_path: Path,
) -> bool:
    """Draw the 2x2 interaction bar chart if the results form a 2x2 design.

    Returns
    -------
    bool
        ``True`` if the figure was written.
    """
    from superadditivity.visualization.interaction_plots import (
        factorial_cell_stats,
        plot_interaction_bar,
    )

    try:
        table = agg.build_factorial_table(drift_metric=metric)
    except ValueError as exc:
        logger.info("Skipping interaction bar chart (not a 2x2 design): %s", exc)
        return False
    if table.empty:
        return False

    means, cis, counts = factorial_cell_stats(table)
    I = InteractionAnalyzer.compute_interaction(
        *(table.loc[table["cell"] == c, "metric"].values for c in "ABCD")
    )

    ci = p_value = n_boot = None
    test_path = results_dir / output_subdir(metric, "analysis") / "interaction_test.json"
    if test_path.exists():
        test = load_json(test_path)
        if test.get("metric_column") == metric.column and test.get(
            "orientation"
        ) == metric.orientation and np.isclose(test.get("I_observed", np.nan), I):
            ci = (test["I_ci_lower"], test["I_ci_upper"])
            p_value = test["p_value"]
            n_boot = test.get("n_bootstrap")
        else:
            logger.warning(
                "%s does not match the current results/metric; run "
                "run_analysis.py --metric %s to show the bootstrap CI and p-value.",
                test_path, metric.name,
            )

    fig = plot_interaction_bar(
        means,
        cell_cis=cis,
        interaction_I=I,
        interaction_ci=ci,
        interaction_p=p_value,
        interaction_n_bootstrap=n_boot,
        ylabel=metric.label,
        title=(
            "2x2 Factorial: " + metric.label
            + " (n = " + ", ".join(f"{c}:{counts[c]}" for c in "ABCD") + ")"
        ),
        save_path=str(save_path),
    )
    plt.close(fig)
    return True


def plot_trajectory_figure(
    agg: ResultAggregator,
    metric: DriftMetric,
    layer: str,
    save_path: Path,
) -> bool:
    """Plot per-condition mean drift trajectories (drift orientation).

    Runs are grouped by their factorial cell when the results form a 2x2
    design, otherwise by ``data_regime + network_regime``.

    Returns
    -------
    bool
        ``True`` if the figure was written.
    """
    from superadditivity.visualization.drift_trajectories import plot_drift_trajectories

    key = metric.h5_key(layer)
    if key is None:
        logger.info("Metric %s has no per-round trajectory; skipping.", metric.name)
        return False
    runs = agg.load_trajectories(key)
    runs = [r for r in runs if r["data_regime"] is not None and r["network_regime"] is not None]
    if not runs:
        logger.info("No %s trajectories found; skipping trajectory plot.", key)
        return False

    import pandas as pd

    labels_df = pd.DataFrame(
        {"data_regime": [r["data_regime"] for r in runs],
         "network_regime": [r["network_regime"] for r in runs]}
    )
    try:
        cells = assign_factorial_cells(labels_df)
        labels = [f"{c}: {CELL_LABELS[c]}" for c in cells]
    except ValueError:
        labels = [f"{d} + {n}" for d, n in zip(labels_df["data_regime"], labels_df["network_regime"])]

    grouped: Dict[str, List[np.ndarray]] = defaultdict(list)
    rounds_by_label: Dict[str, np.ndarray] = {}
    for label, run in zip(labels, runs):
        grouped[label].append(metric.to_drift(run["values"]))
        prev = rounds_by_label.get(label)
        if prev is None or len(run["rounds"]) < len(prev):
            rounds_by_label[label] = run["rounds"]

    plot_data = {}
    for label in sorted(grouped):
        n = min(len(v) for v in grouped[label])
        plot_data[label] = np.stack([v[:n] for v in grouped[label]])
    # All conditions share the x-axis; use the shortest common round list.
    rounds = min(rounds_by_label.values(), key=len)
    n_common = len(rounds)
    plot_data = {k: v[:, :n_common] for k, v in plot_data.items()}

    fig = plot_drift_trajectories(
        plot_data,
        rounds=rounds,
        metric_name=metric.name,
        ylabel=f"{metric.label}, {layer}",
        title=f"{metric.label} over training ({layer})",
        save_path=str(save_path),
    )
    plt.close(fig)
    return True


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    parser = argparse.ArgumentParser(description="Generate figures")
    parser.add_argument("--results_dir", required=True)
    parser.add_argument("--output_dir", default=None)
    parser.add_argument(
        "--force", action="store_true",
        help="Regenerate figures even if inputs haven't changed.",
    )
    parser.add_argument(
        "--metric", default=DEFAULT_METRIC,
        help=(
            f"Drift metric: one of {sorted(DRIFT_METRICS)} or a raw summary "
            f"column (then --orientation is required). Default: {DEFAULT_METRIC}."
        ),
    )
    parser.add_argument(
        "--orientation", default=None, choices=list(ORIENTATIONS),
        help="For a raw --metric column: 'similarity' or 'distance'.",
    )
    parser.add_argument(
        "--layer", default="layer4",
        help="Layer for the drift trajectory plot (default: layer4).",
    )
    args = parser.parse_args()

    try:
        metric = resolve_drift_metric(args.metric, args.orientation)
    except ValueError as exc:
        parser.error(str(exc))

    results_dir = Path(args.results_dir)
    fig_dir = Path(args.output_dir or str(results_dir / output_subdir(metric, "figures")))
    ensure_dir(fig_dir)

    if not args.force and _is_viz_current(fig_dir, results_dir, metric, args.layer):
        logger.info(
            "Figures are up to date (inputs unchanged). Use --force to regenerate."
        )
        return

    _invalidate_viz_fingerprint(fig_dir)
    set_publication_style()

    agg = ResultAggregator(str(results_dir))
    df = agg.load_summaries()

    if df.empty:
        logger.error("No data found in %s", results_dir)
        return

    if "data_regime" in df.columns and "network_regime" in df.columns:
        if metric.column in df.columns:
            from superadditivity.visualization.interaction_plots import plot_factorial_grid
            fig = plot_factorial_grid(
                df,
                drift_metric=metric,
                save_path=str(fig_dir / "factorial_grid.pdf"),
            )
            plt.close(fig)
            logger.info("Generated factorial grid")

            if plot_interaction_bar_figure(
                agg, metric, results_dir, fig_dir / "interaction_bar.pdf"
            ):
                logger.info("Generated interaction bar chart")
        else:
            logger.warning(
                "Summaries have no %s column; skipping factorial figures.", metric.column
            )

    if plot_trajectory_figure(agg, metric, args.layer, fig_dir / "drift_trajectories.pdf"):
        logger.info("Generated drift trajectories")

    surface_path = (
        results_dir / output_subdir(metric, "analysis") / "interaction_surface.json"
    )
    if surface_path.exists():
        from superadditivity.visualization.interaction_surface import plot_interaction_surface

        surface_data = load_json(surface_path)
        # Older surface files carry no metric and were computed on raw CKA.
        if surface_data.get("metric_column", metric.column) != metric.column or (
            "orientation" not in surface_data
        ):
            logger.warning(
                "%s was not computed on %s in drift orientation; re-run "
                "run_analysis.py --analysis surface --metric %s. Skipping.",
                surface_path, metric.name, metric.name,
            )
        else:
            fig = plot_interaction_surface(
                np.array(surface_data["alpha_values"]),
                np.array(surface_data["beta_values"]),
                np.array(surface_data["I_values"]),
                title=f"Interaction Surface I(α, β), {metric.label}",
                save_path=str(fig_dir / "interaction_surface.pdf"),
            )
            plt.close(fig)
            logger.info("Generated interaction surface")

    _write_viz_fingerprint(fig_dir, results_dir, metric, args.layer)
    logger.info("All figures saved to %s", fig_dir)


if __name__ == "__main__":
    main()
