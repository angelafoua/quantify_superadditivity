"""Build a Markdown statistical report from analysis results."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from superadditivity.utils.io import ensure_dir

logger = logging.getLogger(__name__)


class StatisticalReport:
    """Accumulate sections and render a Markdown report.

    Parameters
    ----------
    title:
        Report title.
    output_path:
        Path to write the Markdown file.
    """

    def __init__(self, title: str, output_path: str) -> None:
        self.title = title
        self.output_path = Path(output_path)
        self._sections: List[str] = []

    def add_section(self, heading: str, body: str) -> None:
        """Append a section with a level-2 heading."""
        self._sections.append(f"## {heading}\n\n{body}\n")

    def add_anova_results(self, anova_result: Dict[str, Any]) -> None:
        """Format ANOVA table into a report section."""
        table = anova_result["table"]
        lines = ["| Source | SS | df | F | p |", "|--------|----|----|---|---|"]
        for source in table.index:
            row = table.loc[source]
            ss = f"{row.get('sum_sq', 0):.4f}"
            df_val = f"{row.get('df', 0):.0f}"
            f_val = f"{row.get('F', 0):.4f}" if "F" in row and not (row.get("F") != row.get("F")) else "—"
            p_val = f"{row.get('PR(>F)', 1):.4g}" if "PR(>F)" in row and not (row.get("PR(>F)") != row.get("PR(>F)")) else "—"
            lines.append(f"| {source} | {ss} | {df_val} | {f_val} | {p_val} |")

        eta_sq = anova_result.get("partial_eta_sq", {})
        if eta_sq:
            lines.append("")
            lines.append("**Partial eta-squared:**")
            for source, val in eta_sq.items():
                lines.append(f"- {source}: {val:.4f}")

        self.add_section("Two-Way ANOVA", "\n".join(lines))

    @staticmethod
    def format_p(p: float, n_resamples: Optional[int] = None) -> str:
        """Format a p-value; a resampling p at its floor is shown as a bound."""
        if p != p:  # nan
            return "n/a"
        if n_resamples and p <= 1.0 / (n_resamples + 1) + 1e-15:
            return f"< {1.0 / n_resamples:.1g}"
        return f"{p:.4g}"

    def add_interaction_results(self, interaction_result: Dict[str, Any]) -> None:
        """Format the interaction-term test.

        Uses ``method`` (if present) to describe the test. A nested
        ``independent_bootstrap`` result, if present, is reported as a
        secondary analysis that ignores the pairing of runs by seed.
        """
        r = interaction_result
        method = r.get("method", "bootstrap, cells resampled independently")
        lines = [
            f"- **Method:** {method}",
            f"- **I (observed):** {r['I_observed']:.4g}",
            f"- **95% CI:** [{r['I_ci_lower']:.4g}, {r['I_ci_upper']:.4g}]",
            f"- **p-value (H0: I <= 0):** "
            f"{self.format_p(r['p_value'], None if 'n_units' in r else r.get('n_bootstrap'))}",
        ]
        if "n_units" in r:
            lines.append(f"- **Units (seeds) in all cells:** {r['n_units']}")
            if r.get("bootstrap_ci_lower") == r.get("bootstrap_ci_lower"):
                lines.append(
                    f"- **Bootstrap 95% CI over units:** "
                    f"[{r['bootstrap_ci_lower']:.4g}, {r['bootstrap_ci_upper']:.4g}]"
                )
        else:
            lines.append(f"- **N bootstrap:** {r['n_bootstrap']}")

        p = r["p_value"]
        if p != p:
            lines.append("\n**Conclusion:** Not testable (fewer than 2 seeds in every cell).")
        elif p < 0.05:
            lines.append("\n**Conclusion:** Significant superadditive interaction detected (I > 0).")
        else:
            lines.append("\n**Conclusion:** No significant superadditive interaction detected.")

        ind = r.get("independent_bootstrap")
        if ind:
            lines += [
                "",
                "*Secondary (treats runs as independent, ignoring shared seeds):* "
                f"I = {ind['I_observed']:.4g}, 95% CI [{ind['I_ci_lower']:.4g}, "
                f"{ind['I_ci_upper']:.4g}], p = "
                f"{self.format_p(ind['p_value'], ind.get('n_bootstrap'))}",
            ]
        self.add_section("Superadditive Interaction Test", "\n".join(lines))

    def add_surface_fit_results(self, surface_result: Dict[str, Any]) -> None:
        """Format surface fit parameters."""
        params = surface_result["params"]
        if not surface_result.get("fitted", True):
            self.add_section(
                "Interaction Surface Fit",
                "Not fitted (too few sweep points or the fit failed).",
            )
            return
        lines = [
            "Model: I(α, β) = a · α^(-b) · β^c + d",
            "",
            f"- a = {params['a']:.4f}",
            f"- b = {params['b']:.4f}",
            f"- c = {params['c']:.4f}",
            f"- d = {params['d']:.4f}",
            f"- R² = {surface_result['r_squared']:.4f} (descriptive: points share "
            "reference cells, so they are not independent)",
        ]
        self.add_section("Interaction Surface Fit", "\n".join(lines))

    def add_pairwise_results(self, pairs: list) -> None:
        """Format pairwise t-test results."""
        lines = [
            "| Group 1 | Group 2 | t | p (raw) | p (corrected) | Cohen's d |",
            "|---------|---------|---|---------|---------------|-----------|",
        ]
        for p in pairs:
            lines.append(
                f"| {p['group1']} | {p['group2']} | "
                f"{p['t_stat']:.3f} | {p['p_raw']:.4g} | "
                f"{p['p_corrected']:.4g} | {p['cohens_d']:.3f} |"
            )
        self.add_section("Pairwise Comparisons", "\n".join(lines))

    def render(self) -> str:
        """Render the full report as a Markdown string."""
        parts = [f"# {self.title}\n"]
        parts.extend(self._sections)
        return "\n".join(parts)

    def save(self) -> None:
        """Write the report to disk."""
        ensure_dir(self.output_path.parent)
        text = self.render()
        self.output_path.write_text(text, encoding="utf-8")
        logger.info("Statistical report saved to %s", self.output_path)
