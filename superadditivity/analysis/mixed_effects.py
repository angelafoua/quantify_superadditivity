"""Mixed-effects models for repeated-measures factorial designs."""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

import pandas as pd

logger = logging.getLogger(__name__)


def fit_mixed_effects(
    df: pd.DataFrame,
    dv: str = "metric",
    factor_a: str = "data_regime",
    factor_b: str = "network_regime",
    random_effect: str = "run_seed",
    crossed_effect: Optional[str] = "graph_seed",
) -> Dict[str, Any]:
    """Fit a mixed-effects model for the factorial with seed random effects.

    Model: ``metric ~ data_regime * network_regime`` with a random
    intercept for ``run_seed`` and, when ``crossed_effect`` has more than
    one level, a crossed random intercept for it (``graph_seed``).

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
    crossed_effect:
        Optional second, crossed random intercept (``None`` to disable).

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
        crossed = (
            crossed_effect is not None
            and crossed_effect in df.columns
            and df[crossed_effect].nunique() > 1
        )
        if crossed:
            # Crossed random intercepts for both seeds: one group, two
            # variance components (statsmodels' idiom for crossed effects).
            data = df.assign(_one=1)
            model = smf.mixedlm(
                formula, data=data, groups=data["_one"],
                vc_formula={
                    random_effect: f"0 + C({random_effect})",
                    crossed_effect: f"0 + C({crossed_effect})",
                },
            )
        else:
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
