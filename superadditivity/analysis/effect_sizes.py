"""Effect-size measures and bootstrap confidence intervals."""

from __future__ import annotations

import logging
from typing import Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


def cohens_d(group1: np.ndarray, group2: np.ndarray) -> float:
    """Compute Cohen's d (pooled SD)."""
    n1, n2 = len(group1), len(group2)
    var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    if pooled_std < 1e-15:
        return 0.0
    return float((np.mean(group1) - np.mean(group2)) / pooled_std)


def eta_squared(ss_effect: float, ss_total: float) -> float:
    """Compute eta-squared."""
    if ss_total < 1e-15:
        return 0.0
    return float(ss_effect / ss_total)


def bootstrap_ci(
    data: np.ndarray,
    statistic: str = "mean",
    n_bootstrap: int = 10000,
    confidence: float = 0.95,
    seed: Optional[int] = None,
) -> Tuple[float, float, float]:
    """Non-parametric bootstrap confidence interval.

    Parameters
    ----------
    data:
        1-D array of observations.
    statistic:
        ``"mean"`` or ``"median"``.
    n_bootstrap:
        Number of resamples.
    confidence:
        Confidence level (e.g. 0.95).
    seed:
        RNG seed.

    Returns
    -------
    (point_estimate, ci_lower, ci_upper)
    """
    rng = np.random.RandomState(seed)
    data = np.asarray(data, dtype=np.float64)

    func = np.mean if statistic == "mean" else np.median
    point = float(func(data))

    boot_stats = np.empty(n_bootstrap)
    for i in range(n_bootstrap):
        sample = rng.choice(data, size=len(data), replace=True)
        boot_stats[i] = func(sample)

    alpha = 1.0 - confidence
    ci_lower = float(np.percentile(boot_stats, 100 * alpha / 2))
    ci_upper = float(np.percentile(boot_stats, 100 * (1 - alpha / 2)))

    return point, ci_lower, ci_upper
