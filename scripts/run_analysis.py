"""Post-hoc statistical analysis of experiment results.

Usage:
    python scripts/run_analysis.py --results_dir outputs/core_factorial
    python scripts/run_analysis.py --results_dir outputs/pout_sweep --analysis surface
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from superadditivity.analysis.aggregator import ResultAggregator
from superadditivity.analysis.anova import two_way_anova, functional_anova
from superadditivity.analysis.effect_sizes import bootstrap_ci
from superadditivity.analysis.interaction_analyzer import InteractionAnalyzer
from superadditivity.analysis.mixed_effects import fit_mixed_effects
from superadditivity.analysis.multiple_comparisons import pairwise_ttests
from superadditivity.analysis.statistical_report import StatisticalReport
from superadditivity.utils.io import ensure_dir, save_json

logger = logging.getLogger(__name__)


def run_factorial_analysis(results_dir: str, output_dir: str) -> None:
    """Run the full 2×2 factorial analysis with interaction test."""
    ensure_dir(output_dir)
    agg = ResultAggregator(results_dir)
    df = agg.build_factorial_table()

    if df.empty:
        logger.error("No data found in %s", results_dir)
        return

    logger.info("Loaded %d runs for factorial analysis", len(df))

    report = StatisticalReport(
        title="Superadditivity Analysis Report",
        output_path=str(Path(output_dir) / "statistical_report.md"),
    )

    anova_result = two_way_anova(df)
    report.add_anova_results(anova_result)

    func_anova = functional_anova(df, n_permutations=10000, seed=42)
    report.add_section(
        "Permutation ANOVA",
        f"Interaction F={func_anova['observed_F']:.4f}, "
        f"permutation p={func_anova['perm_p_value']:.4g}",
    )

    groups = {}
    for name, group in df.groupby(["data_regime", "network_regime"]):
        key = f"{name[0]}_{name[1]}"
        groups[key] = group["metric"].values

    pairs = pairwise_ttests(groups)
    report.add_pairwise_results(pairs)

    cell_keys = sorted(groups.keys())
    if len(cell_keys) >= 4:
        A_key = [k for k in cell_keys if "iid" in k.lower() and ("er" in k.lower() or "dense" in k.lower())]
        B_key = [k for k in cell_keys if "noniid" in k.lower() and ("er" in k.lower() or "dense" in k.lower())]
        C_key = [k for k in cell_keys if "iid" in k.lower() and ("sbm" in k.lower() or "community" in k.lower())]
        D_key = [k for k in cell_keys if "noniid" in k.lower() and ("sbm" in k.lower() or "community" in k.lower())]

        if A_key and B_key and C_key and D_key:
            interaction_result = InteractionAnalyzer.bootstrap_interaction_test(
                groups[A_key[0]], groups[B_key[0]],
                groups[C_key[0]], groups[D_key[0]],
                n_bootstrap=10000, seed=42,
            )
            report.add_interaction_results(interaction_result)
            save_json(
                {k: v for k, v in interaction_result.items() if k != "bootstrap_distribution"},
                Path(output_dir) / "interaction_test.json",
            )

    me_result = fit_mixed_effects(df)
    report.add_section("Mixed-Effects Model", me_result["summary"])

    report.save()
    logger.info("Analysis complete. Report: %s", report.output_path)


def run_surface_analysis(results_dir: str, output_dir: str) -> None:
    """Compute and fit the I(alpha, beta) interaction surface."""
    ensure_dir(output_dir)
    agg = ResultAggregator(results_dir)
    df = agg.load_summaries()

    if df.empty:
        logger.error("No data found in %s", results_dir)
        return

    sweep_data = InteractionAnalyzer.compute_sweep_interactions(df)
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
        "fit_params": surface_fit["params"],
        "r_squared": surface_fit["r_squared"],
    }, Path(output_dir) / "interaction_surface.json")

    report = StatisticalReport(
        title="Interaction Surface Analysis",
        output_path=str(Path(output_dir) / "surface_report.md"),
    )
    report.add_surface_fit_results(surface_fit)
    report.save()

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
    args = parser.parse_args()

    output_dir = args.output_dir or str(Path(args.results_dir) / "analysis")

    if args.analysis in ("factorial", "both"):
        run_factorial_analysis(args.results_dir, output_dir)
    if args.analysis in ("surface", "both"):
        run_surface_analysis(args.results_dir, output_dir)


if __name__ == "__main__":
    main()
