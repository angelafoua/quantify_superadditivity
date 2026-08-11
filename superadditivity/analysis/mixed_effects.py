"""Mixed-effects models for repeated-measures factorial designs."""

from __future__ import annotations

import logging
from typing import Any, Dict

import pandas as pd

logger = logging.getLogger(__name__)


def fit_mixed_effects(
    df: pd.DataFrame,
    dv: str = "metric",
    factor_a: str = "data_regime",
    factor_b: str = "network_regime",
    random_effect: str = "run_seed",
) -> Dict[str, Any]:
    """Fit a mixed-effects model with crossed fixed effects and random intercept.

    Model: ``metric ~ data_regime * network_regime``, random intercept on ``run_seed``.

    Parameters
    ----------
    df:
        DataFrame with dependent and independent variables.
    dv:
        Dependent variable column.
    factor_a, factor_b:
        Fixed-effect factor columns.
    random_effect:
        Column for the random intercept.

    Returns
    -------
    dict with ``"summary"`` (text), ``"params"`` (dict), ``"converged"`` (bool).
    """
    try:
        import statsmodels.formula.api as smf
    except ImportError as e:
        raise ImportError("statsmodels required for mixed-effects models") from e

    formula = f"{dv} ~ C({factor_a}) * C({factor_b})"

    try:
        model = smf.mixedlm(
            formula, data=df, groups=df[random_effect],
        )
        result = model.fit(reml=True)

        params = {}
        for name, val in result.params.items():
            params[name] = float(val)

        pvalues = {}
        for name, val in result.pvalues.items():
            pvalues[name] = float(val)

        logger.info("Mixed-effects model converged: %s", result.converged)

        return {
            "summary": str(result.summary()),
            "params": params,
            "pvalues": pvalues,
            "converged": bool(result.converged),
            "aic": float(result.aic) if hasattr(result, "aic") else None,
            "bic": float(result.bic) if hasattr(result, "bic") else None,
        }

    except Exception as e:
        logger.error("Mixed-effects model failed: %s", e)
        return {
            "summary": f"Model failed: {e}",
            "params": {},
            "pvalues": {},
            "converged": False,
            "aic": None,
            "bic": None,
        }
