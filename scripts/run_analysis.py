"""Post-hoc statistical analysis of experiment results.

Usage:
    python scripts/run_analysis.py --results_dir outputs/core_factorial
    python scripts/run_analysis.py --results_dir outputs/core_factorial --metric mmd
    python scripts/run_analysis.py --results_dir outputs/pout_sweep --analysis surface
    python scripts/run_analysis.py --results_dir outputs/pout_sweep --analysis surface --force
    python scripts/run_analysis.py --results_dir outputs/core_factorial \
        --metric final_mmd_paired_layer4_drift_bw0p5 --orientation distance

All analyses run on *drift* (higher = more drift): CKA is converted with
``1 - CKA``; MMD is used as is. See superadditivity/analysis/drift_metrics.py.
Output goes to ``<results_dir>/analysis`` for CKA and
``<results_dir>/analysis_<metric>`` otherwise.
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import sys
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from superadditivity.analysis.aggregator import ResultAggregator
from superadditivity.analysis.drift_metrics import (
    DEFAULT_METRIC,
    DRIFT_METRICS,
    ORIENTATIONS,
    TRANSFORMS,
    DriftMetric,
    output_subdir,
    resolve_drift_metric,
)
from superadditivity.analysis.anova import two_way_anova, functional_anova
from superadditivity.analysis.interaction_analyzer import InteractionAnalyzer
from superadditivity.analysis.mixed_effects import fit_mixed_effects
from superadditivity.analysis.multiple_comparisons import pairwise_ttests
from superadditivity.analysis.statistical_report import StatisticalReport
from superadditivity.utils.io import ensure_dir, save_json

logger = logging.getLogger(__name__)


def _find_summaries(results_dir: str) -> list:
    """Find summary.json files in nested or sibling-directory layouts."""
    rd = Path(results_dir)
    found = sorted(rd.rglob("summary.json"))
    if found:
        return found
    parent = rd.parent
    prefix = rd.name + "_"
    if parent.is_dir():
        for sibling in sorted(parent.iterdir()):
            if sibling.is_dir() and sibling.name.startswith(prefix):
                found.extend(sorted(sibling.rglob("summary.json")))
    return found


def _summaries_fingerprint(
    results_dir: str, metric: Optional[DriftMetric] = None
) -> str:
    """Hash all summary.json files (and the metric) for change detection."""
    metric = metric or resolve_drift_metric()
    md5 = hashlib.md5()
    md5.update(f"{metric.column}|{metric.orientation}|{metric.transform}".encode())
    for p in _find_summaries(results_dir):
        md5.update(p.read_bytes())
    return md5.hexdigest()


def default_output_dir(results_dir: str, metric: DriftMetric) -> str:
    """``<results_dir>/analysis`` for CKA, ``analysis_<metric>`` otherwise."""
    return str(Path(results_dir) / output_subdir(metric, "analysis"))


def _metric_section(metric: DriftMetric) -> str:
    transform_note = (
        "" if metric.transform == "none"
        else f"\n- **Transform:** {metric.transform} (scale-robustness check)"
    )
    return (
        f"- **Metric:** `{metric.name}`: {metric.description}{transform_note}\n"
        f"- **Summary column:** `{metric.column}` ({metric.orientation})\n"
        f"- **{metric.drift_definition}** (higher = more drift)\n"
        f"- **Interaction:** I = Drift(D) - Drift(B) - Drift(C) + Drift(A); "
        f"I > 0 is superadditive drift"
    )


_EXPECTED_ANALYSIS_FILES = [
    "statistical_report.md",
    "interaction_test.json",
]


def _is_analysis_current(
    output_dir: str, results_dir: str, metric: Optional[DriftMetric] = None
) -> bool:
    """Return True if analysis output was produced from the current summaries.

    Also checks that expected output files still exist — a matching
    fingerprint alone is not enough if outputs were deleted.
    """
    out = Path(output_dir)
    stamp = out / ".analysis_fingerprint"
    if not stamp.exists():
        return False
    try:
        saved = stamp.read_text().strip()
        if saved != _summaries_fingerprint(results_dir, metric):
            return False
    except Exception:
        return False
    if not any((out / f).exists() for f in _EXPECTED_ANALYSIS_FILES):
        logger.info(
            "Fingerprint matches but no analysis outputs found in %s. "
            "Regenerating.", output_dir,
        )
        return False
    return True


def _write_fingerprint(
    output_dir: str, results_dir: str, metric: Optional[DriftMetric] = None
) -> None:
    stamp = Path(output_dir) / ".analysis_fingerprint"
    stamp.write_text(_summaries_fingerprint(results_dir, metric))


def _invalidate_fingerprint(output_dir: str) -> None:
    """Remove stale fingerprint so a partial re-run is never mistaken for current."""
    stamp = Path(output_dir) / ".analysis_fingerprint"
    if stamp.exists():
        stamp.unlink()


def run_factorial_analysis(
    results_dir: str,
    output_dir: str,
    metric: Optional[DriftMetric] = None,
) -> None:
    """Run the full 2×2 factorial analysis with interaction test.

    Parameters
    ----------
    results_dir:
        Directory containing run summaries.
    output_dir:
        Where the report and ``interaction_test.json`` are written.
    metric:
        Drift metric to analyse (default: CKA, as ``1 - CKA``).
    """
    metric = metric or resolve_drift_metric()
    ensure_dir(output_dir)
    _invalidate_fingerprint(output_dir)
    agg = ResultAggregator(results_dir)
    df = agg.build_factorial_table(drift_metric=metric)

    if df.empty:
        logger.error("No data found in %s", results_dir)
        return

    logger.info(
        "Loaded %d runs for factorial analysis of %s (%s)",
        len(df), metric.name, metric.drift_definition,
    )

    report = StatisticalReport(
        title=f"Superadditivity Analysis Report ({metric.name})",
        output_path=str(Path(output_dir) / "statistical_report.md"),
    )
    report.add_section("Drift Metric", _metric_section(metric))

    anova_result = two_way_anova(df)
    report.add_anova_results(anova_result)

    func_anova = functional_anova(df, n_permutations=10000, seed=42)
    report.add_section(
        "Permutation ANOVA",
        f"{func_anova['method']}: interaction F={func_anova['observed_F']:.4f}, "
        f"permutation p="
        f"{StatisticalReport.format_p(func_anova['perm_p_value'], func_anova['n_permutations'])}",
    )

    groups = {}
    for name, group in df.groupby(["data_regime", "network_regime"]):
        key = f"{name[0]}_{name[1]}"
        groups[key] = group["metric"].values

    pairs = pairwise_ttests(groups)
    report.add_pairwise_results(pairs)

    # Cells are assigned in build_factorial_table (A/B/C/D); a design
    # that is not 2x2 raises there instead of silently skipping the test.
    # Primary test pairs runs by run_seed (shared initial weights and data
    # partition across cells). The independent-cells bootstrap is kept as a
    # secondary analysis.
    cell_values = {c: df.loc[df["cell"] == c, "metric"].values for c in "ABCD"}
    independent = InteractionAnalyzer.bootstrap_interaction_test(
        cell_values["A"], cell_values["B"], cell_values["C"], cell_values["D"],
        n_bootstrap=10000, seed=42,
    )
    independent.pop("bootstrap_distribution", None)
    if "run_seed" in df.columns and df["run_seed"].notna().all():
        interaction_result = InteractionAnalyzer.paired_interaction_test(
            {c: df[df["cell"] == c] for c in "ABCD"}, unit_col="run_seed",
            n_bootstrap=10000, seed=42,
        )
        interaction_result["independent_bootstrap"] = independent
    else:
        logger.warning("No run_seed column: using the independent-cells bootstrap.")
        interaction_result = independent
    report.add_interaction_results(interaction_result)
    cell_info = {}
    for c in "ABCD":
        rows = df[df["cell"] == c]
        cell_info[c] = {
            "data_regime": str(rows["data_regime"].iloc[0]),
            "network_regime": str(rows["network_regime"].iloc[0]),
            "n_runs": int(len(rows)),
            "mean_drift": float(rows["metric"].mean()),
            "mean_raw": float(rows["metric_raw"].mean()),
        }
    save_json(
        {
            **{k: v for k, v in interaction_result.items()
               if k != "bootstrap_distribution"},
            "metric": metric.name,
            "metric_column": metric.column,
            "orientation": metric.orientation,
            "transform": metric.transform,
            "drift_definition": metric.drift_definition,
            "cells": cell_info,
        },
        Path(output_dir) / "interaction_test.json",
    )

    me_result = fit_mixed_effects(df)
    report.add_section("Mixed-Effects Model", me_result["summary"])

    report.save()
    _write_fingerprint(output_dir, results_dir, metric)
    logger.info("Analysis complete. Report: %s", report.output_path)


def run_surface_analysis(
    results_dir: str,
    output_dir: str,
    metric: Optional[DriftMetric] = None,
) -> None:
    """Compute and fit the I(alpha, beta) interaction surface.

    Parameters
    ----------
    results_dir:
        Directory containing run summaries.
    output_dir:
        Where ``interaction_surface.json`` and the report are written.
    metric:
        Drift metric to analyse (default: CKA, as ``1 - CKA``).
    """
    metric = metric or resolve_drift_metric()
    _invalidate_fingerprint(output_dir)
    ensure_dir(output_dir)
    agg = ResultAggregator(results_dir)
    df = agg.load_summaries()

    if df.empty:
        logger.error("No data found in %s", results_dir)
        return
    if metric.column not in df.columns:
        raise ValueError(f"Missing column in summaries: {metric.column}")

    df = df[df[metric.column].notna()].copy()
    df["_drift"] = metric.to_drift(df[metric.column])
    sweep_data = InteractionAnalyzer.compute_sweep_interactions(
        df, metric_col="_drift"
    )
    if len(sweep_data["I_values"]) == 0:
        logger.warning("No sweep points computed.")
        return

    surface_fit = InteractionAnalyzer.fit_interaction_surface(
        sweep_data["alpha_values"],
        sweep_data["beta_values"],
        sweep_data["I_values"],
    )

    save_json({
        "alpha_values": sweep_data["alpha_values"].tolist(),
        "beta_values": sweep_data["beta_values"].tolist(),
        "I_values": sweep_data["I_values"].tolist(),
        "I_ci_lower": sweep_data["I_ci_lower"].tolist(),
        "I_ci_upper": sweep_data["I_ci_upper"].tolist(),
        "p_values": sweep_data["p_values"].tolist(),
        "n_units": sweep_data["n_units"].tolist(),
        "baseline_alpha": sweep_data["baseline_alpha"],
        "baseline_beta": sweep_data["baseline_beta"],
        "fit_params": surface_fit["params"],
        "r_squared": surface_fit["r_squared"],
        "metric": metric.name,
        "metric_column": metric.column,
        "orientation": metric.orientation,
        "transform": metric.transform,
        "drift_definition": metric.drift_definition,
    }, Path(output_dir) / "interaction_surface.json")

    report = StatisticalReport(
        title=f"Interaction Surface Analysis ({metric.name})",
        output_path=str(Path(output_dir) / "surface_report.md"),
    )
    report.add_section("Drift Metric", _metric_section(metric))
    point_lines = [
        f"Reference cell: alpha = {sweep_data['baseline_alpha']} (IID if inf), "
        f"p_out = {sweep_data['baseline_beta']}. Points on the reference row or "
        "column are omitted (I = 0 by construction).",
        "",
        "| alpha | p_out | I | 95% CI | p (I <= 0) | seeds |",
        "|---|---|---|---|---|---|",
    ]
    for i in range(len(sweep_data["I_values"])):
        point_lines.append(
            f"| {sweep_data['alpha_values'][i]:g} | {sweep_data['beta_values'][i]:g} "
            f"| {sweep_data['I_values'][i]:.4g} "
            f"| [{sweep_data['I_ci_lower'][i]:.4g}, {sweep_data['I_ci_upper'][i]:.4g}] "
            f"| {sweep_data['p_values'][i]:.3g} | {sweep_data['n_units'][i]} |"
        )
    report.add_section("Interaction per sweep point", "\n".join(point_lines))
    report.add_surface_fit_results(surface_fit)
    report.save()

    _write_fingerprint(output_dir, results_dir, metric)
    logger.info("Surface analysis complete. Output: %s", output_dir)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    parser = argparse.ArgumentParser(description="Run statistical analysis")
    parser.add_argument("--results_dir", required=True, help="Directory with experiment outputs")
    parser.add_argument("--output_dir", default=None, help="Output directory (default: results_dir/analysis)")
    parser.add_argument("--analysis", default="factorial", choices=["factorial", "surface", "both"])
    parser.add_argument(
        "--force", action="store_true",
        help="Rerun analysis even if results haven't changed since last run.",
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
        help=(
            "For a raw --metric column: 'similarity' (drift = 1 - value) or "
            "'distance' (drift = value)."
        ),
    )
    parser.add_argument(
        "--transform", default="none", choices=list(TRANSFORMS),
        help=(
            "Monotone transform of drift before analysis (scale-robustness "
            "check): none, log (drift > 0) or logit (0 < drift < 1, e.g. 1 - CKA)."
        ),
    )
    args = parser.parse_args()

    try:
        metric = resolve_drift_metric(args.metric, args.orientation, args.transform)
    except ValueError as exc:
        parser.error(str(exc))

    output_dir = args.output_dir or default_output_dir(args.results_dir, metric)

    if not args.force and _is_analysis_current(output_dir, args.results_dir, metric):
        logger.info(
            "Analysis is up to date (summaries unchanged). "
            "Use --force to regenerate."
        )
        return

    if args.analysis in ("factorial", "both"):
        run_factorial_analysis(args.results_dir, output_dir, metric)
    if args.analysis in ("surface", "both"):
        run_surface_analysis(args.results_dir, output_dir, metric)


if __name__ == "__main__":
    main()
