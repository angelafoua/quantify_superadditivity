"""Post-hoc statistical analysis and superadditivity quantification."""

from __future__ import annotations

from superadditivity.analysis.aggregator import ResultAggregator
from superadditivity.analysis.anova import two_way_anova, functional_anova
from superadditivity.analysis.effect_sizes import cohens_d, eta_squared, bootstrap_ci
from superadditivity.analysis.interaction_analyzer import InteractionAnalyzer
from superadditivity.analysis.mixed_effects import fit_mixed_effects
from superadditivity.analysis.multiple_comparisons import correct_pvalues, pairwise_ttests
from superadditivity.analysis.statistical_report import StatisticalReport

__all__ = [
    "ResultAggregator",
    "two_way_anova",
    "functional_anova",
    "cohens_d",
    "eta_squared",
    "bootstrap_ci",
    "InteractionAnalyzer",
    "fit_mixed_effects",
    "correct_pvalues",
    "pairwise_ttests",
    "StatisticalReport",
]
