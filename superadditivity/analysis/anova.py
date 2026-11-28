"""Two-way ANOVA and functional ANOVA via permutation."""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def two_way_anova(
    df: pd.DataFrame,
    dv: str = "metric",
    factor_a: str = "data_regime",
    factor_b: str = "network_regime",
) -> Dict[str, Any]:
    """Type-II two-way ANOVA with partial eta-squared.

    Parameters
    ----------
    df:
        DataFrame with the dependent variable and two factors.
    dv:
        Column name of the dependent variable.
    factor_a, factor_b:
        Column names of the two factors.

    Returns
    -------
    dict with keys ``"table"`` (DataFrame), ``"partial_eta_sq"`` (dict).
    """
    try:
        import statsmodels.api as sm
        from statsmodels.formula.api import ols
    except ImportError as e:
        raise ImportError("statsmodels is required for ANOVA") from e

    formula = f"{dv} ~ C({factor_a}) * C({factor_b})"
    model = ols(formula, data=df).fit()
    anova_table = sm.stats.anova_lm(model, typ=2)

    ss_resid = anova_table.loc["Residual", "sum_sq"]
    partial_eta_sq = {}
    for source in anova_table.index:
        if source == "Residual":
            continue
        ss = anova_table.loc[source, "sum_sq"]
        partial_eta_sq[source] = float(ss / (ss + ss_resid))

    logger.info("Two-way ANOVA complete. Interaction p=%.4g",
                anova_table.loc[f"C({factor_a}):C({factor_b})", "PR(>F)"])

    return {
        "table": anova_table,
        "partial_eta_sq": partial_eta_sq,
    }


def functional_anova(
    df: pd.DataFrame,
    dv: str = "metric",
    factor_a: str = "data_regime",
    factor_b: str = "network_regime",
    n_permutations: int = 10000,
    seed: Optional[int] = None,
) -> Dict[str, Any]:
    """Permutation test for the interaction F (Freedman-Lane).

    Under ``H0: no interaction`` the main effects can still be non-zero,
    so the raw responses are *not* exchangeable. Following Freedman & Lane
    (1983), the residuals of the reduced model ``dv ~ A + B`` are permuted
    and added back to its fitted values, and the interaction F of the full
    model ``dv ~ A * B`` is recomputed:

    ``F = ((RSS_reduced - RSS_full) / df_int) / (RSS_full / df_resid)``.

    Parameters
    ----------
    df:
        DataFrame with dependent variable and factors.
    dv:
        Dependent variable column.
    factor_a, factor_b:
        Factor columns.
    n_permutations:
        Number of permutations.
    seed:
        RNG seed.

    Returns
    -------
    dict with ``"observed_F"``, ``"perm_p_value"`` (``(count + 1) /
    (n_permutations + 1)``, so never 0), ``"null_distribution"`` and
    ``"method"``.
    """
    import patsy

    y = df[dv].to_numpy(dtype=np.float64)
    X_full = np.asarray(patsy.dmatrix(f"C({factor_a}) * C({factor_b})", df))
    X_red = np.asarray(patsy.dmatrix(f"C({factor_a}) + C({factor_b})", df))
    n = len(y)
    rank_full = np.linalg.matrix_rank(X_full)
    rank_red = np.linalg.matrix_rank(X_red)
    df_int = rank_full - rank_red
    df_resid = n - rank_full
    if df_int <= 0 or df_resid <= 0:
        raise ValueError(
            f"Cannot test the interaction: df_int={df_int}, df_resid={df_resid}"
        )

    def residual_maker(X: np.ndarray) -> np.ndarray:
        return np.eye(n) - X @ np.linalg.pinv(X)

    M_full = residual_maker(X_full)
    M_red = residual_maker(X_red)

    def f_stats(Y: np.ndarray) -> np.ndarray:
        rss_full = np.einsum("ij,ij->j", M_full @ Y, M_full @ Y)
        rss_red = np.einsum("ij,ij->j", M_red @ Y, M_red @ Y)
        return ((rss_red - rss_full) / df_int) / (rss_full / df_resid)

    observed_F = float(f_stats(y[:, None])[0])

    fitted_red = y - M_red @ y
    resid_red = M_red @ y
    rng = np.random.RandomState(seed)
    null_Fs = np.empty(n_permutations)
    batch = 1000
    for start in range(0, n_permutations, batch):
        k = min(batch, n_permutations - start)
        perms = np.stack([rng.permutation(n) for _ in range(k)], axis=1)  # (n, k)
        Y = fitted_red[:, None] + resid_red[perms]
        null_Fs[start:start + k] = f_stats(Y)

    count = int(np.sum(null_Fs >= observed_F - 1e-12 * abs(observed_F)))
    perm_p = float((count + 1) / (n_permutations + 1))
    logger.info(
        "Freedman-Lane permutation test: interaction F=%.4f, p=%.4g (%d perms)",
        observed_F, perm_p, n_permutations,
    )
    return {
        "observed_F": observed_F,
        "perm_p_value": perm_p,
        "null_distribution": null_Fs,
        "n_permutations": n_permutations,
        "method": "Freedman-Lane (permuted residuals of the main-effects model)",
    }
