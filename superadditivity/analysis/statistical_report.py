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

    def add_interaction_results(self, interaction_result: Dict[str, Any]) -> None:
        """Format interaction-term analysis."""
        lines = [
            f"- **I (observed):** {interaction_result['I_observed']:.4f}",
            f"- **95% CI:** [{interaction_result['I_ci_lower']:.4f}, {interaction_result['I_ci_upper']:.4f}]",
            f"- **Bootstrap p-value (H0: I <= 0):** {interaction_result['p_value']:.4g}",
            f"- **N bootstrap:** {interaction_result['n_bootstrap']}",
        ]

        if interaction_result["p_value"] < 0.05:
            lines.append("\n**Conclusion:** Significant superadditive interaction detected (I > 0).")
        else:
            lines.append("\n**Conclusion:** No significant superadditive interaction detected.")

        self.add_section("Superadditive Interaction Test", "\n".join(lines))

    def add_surface_fit_results(self, surface_result: Dict[str, Any]) -> None:
        """Format surface fit parameters."""
        params = surface_result["params"]
        lines = [
            "Model: I(α, β) = a · α^(-b) · β^c + d",
            "",
            f"- a = {params['a']:.4f}",
            f"- b = {params['b']:.4f}",
            f"- c = {params['c']:.4f}",
            f"- d = {params['d']:.4f}",
            f"- R² = {surface_result['r_squared']:.4f}",
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
