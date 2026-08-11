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
    """Permutation-based functional ANOVA for the interaction term.

    Permutes factor assignments to build a null distribution of the
    interaction F-statistic. Reports permutation p-value.

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
    dict with ``"observed_F"``, ``"perm_p_value"``, ``"null_distribution"``.
    """
    from statsmodels.formula.api import ols
    import statsmodels.api as sm

    formula = f"{dv} ~ C({factor_a}) * C({factor_b})"
    model = ols(formula, data=df).fit()
    anova_table = sm.stats.anova_lm(model, typ=2)

    interaction_key = f"C({factor_a}):C({factor_b})"
    observed_F = float(anova_table.loc[interaction_key, "F"])

    rng = np.random.RandomState(seed)
    null_Fs = np.empty(n_permutations)

    for i in range(n_permutations):
        df_perm = df.copy()
        df_perm[dv] = rng.permutation(df_perm[dv].values)
        try:
            model_p = ols(formula, data=df_perm).fit()
            table_p = sm.stats.anova_lm(model_p, typ=2)
            null_Fs[i] = float(table_p.loc[interaction_key, "F"])
        except Exception:
            null_Fs[i] = 0.0

    perm_p = float(np.mean(null_Fs >= observed_F))

    logger.info(
        "Functional ANOVA: observed F=%.4f, permutation p=%.4g (%d perms)",
        observed_F, perm_p, n_permutations,
    )

    return {
        "observed_F": observed_F,
        "perm_p_value": perm_p,
        "null_distribution": null_Fs,
    }
