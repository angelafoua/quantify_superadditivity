"""Multiple-comparison correction and pairwise t-tests."""

from __future__ import annotations

import logging
from typing import Dict, List, Tuple

import numpy as np
from scipy import stats

logger = logging.getLogger(__name__)


def correct_pvalues(
    pvalues: np.ndarray,
    method: str = "holm-sidak",
) -> np.ndarray:
    """Apply multiple-comparison correction.

    Parameters
    ----------
    pvalues:
        Array of raw p-values.
    method:
        ``"holm-sidak"`` (default), ``"bonferroni"``, or ``"fdr_bh"``.

    Returns
    -------
    Array of corrected p-values.
    """
    pvalues = np.asarray(pvalues, dtype=np.float64)
    n = len(pvalues)

    if method == "bonferroni":
        return np.minimum(pvalues * n, 1.0)

    if method == "holm-sidak":
        order = np.argsort(pvalues)
        corrected = np.empty(n)
        for i, idx in enumerate(order):
            corrected[idx] = 1.0 - (1.0 - pvalues[idx]) ** (n - i)
        running_max = 0.0
        for i in range(n):
            idx = order[i]
            corrected[idx] = max(corrected[idx], running_max)
            running_max = corrected[idx]
        return np.minimum(corrected, 1.0)

    if method == "fdr_bh":
        order = np.argsort(pvalues)
        corrected = np.empty(n)
        for i, idx in enumerate(order):
            corrected[idx] = pvalues[idx] * n / (i + 1)
        running_min = 1.0
        for i in range(n - 1, -1, -1):
            idx = order[i]
            corrected[idx] = min(corrected[idx], running_min)
            running_min = corrected[idx]
        return np.minimum(corrected, 1.0)

    raise ValueError(f"Unknown correction method: {method!r}")


def pairwise_ttests(
    groups: Dict[str, np.ndarray],
    correction: str = "holm-sidak",
) -> List[Dict]:
    """All pairwise Welch t-tests with multiple-comparison correction.

    Parameters
    ----------
    groups:
        Mapping of group name → array of observations.
    correction:
        Correction method for :func:`correct_pvalues`.

    Returns
    -------
    List of dicts with keys: ``group1, group2, t_stat, p_raw, p_corrected, cohens_d``.
    """
    names = sorted(groups.keys())
    pairs = []
    raw_ps = []

    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            g1, g2 = groups[names[i]], groups[names[j]]
            t_stat, p_val = stats.ttest_ind(g1, g2, equal_var=False)

            n1, n2 = len(g1), len(g2)
            pooled = np.sqrt(
                ((n1 - 1) * np.var(g1, ddof=1) + (n2 - 1) * np.var(g2, ddof=1))
                / (n1 + n2 - 2)
            )
            d = float((np.mean(g1) - np.mean(g2)) / pooled) if pooled > 1e-15 else 0.0

            pairs.append({
                "group1": names[i],
                "group2": names[j],
                "t_stat": float(t_stat),
                "p_raw": float(p_val),
                "cohens_d": d,
            })
            raw_ps.append(p_val)

    corrected = correct_pvalues(np.array(raw_ps), method=correction)
    for pair, p_corr in zip(pairs, corrected):
        pair["p_corrected"] = float(p_corr)

    return pairs
