"""Superadditivity interaction-term computation, bootstrap testing, and surface fitting.

This is the core novel analysis module: it computes the interaction term
I = Drift(D) - Drift(B) - Drift(C) + Drift(A) from a 2x2 factorial,
tests whether I > 0 via bootstrap, and fits the I(alpha, beta) surface
from parametric sweeps.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional, Tuple

import numpy as np
from scipy import optimize

logger = logging.getLogger(__name__)


class InteractionAnalyzer:
    """Compute and test superadditive interaction effects.

    The 2x2 factorial:
        A = IID + Dense (both benign)
        B = Non-IID + Dense (data heterogeneity only)
        C = IID + Community (network heterogeneity only)
        D = Non-IID + Community (both present)

    Superadditive interaction: I = D - B - C + A > 0
    """

    @staticmethod
    def compute_interaction(
        drift_A: np.ndarray,
        drift_B: np.ndarray,
        drift_C: np.ndarray,
        drift_D: np.ndarray,
    ) -> float:
        """Compute the interaction term I = mean(D) - mean(B) - mean(C) + mean(A).

        Parameters
        ----------
        drift_A: Drift values for cell A (IID + Dense).
        drift_B: Drift values for cell B (Non-IID + Dense).
        drift_C: Drift values for cell C (IID + Community).
        drift_D: Drift values for cell D (Non-IID + Community).

        Returns
        -------
        The interaction term I.
        """
        I = (
            np.mean(drift_D) - np.mean(drift_B)
            - np.mean(drift_C) + np.mean(drift_A)
        )
        return float(I)

    @staticmethod
    def bootstrap_interaction_test(
        drift_A: np.ndarray,
        drift_B: np.ndarray,
        drift_C: np.ndarray,
        drift_D: np.ndarray,
        n_bootstrap: int = 10000,
        confidence: float = 0.95,
        seed: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Bootstrap test for I > 0 (one-sided).

        Resamples each cell independently, computes I for each bootstrap
        replicate, and reports the bootstrap CI and p-value for H0: I <= 0.

        Returns
        -------
        dict with:
            ``"I_observed"`` — point estimate.
            ``"I_ci_lower"``, ``"I_ci_upper"`` — confidence interval.
            ``"p_value"`` — proportion of bootstrap I <= 0.
            ``"n_bootstrap"`` — number of replicates.
            ``"bootstrap_distribution"`` — array of bootstrap I values.
        """
        rng = np.random.RandomState(seed)
        A = np.asarray(drift_A, dtype=np.float64)
        B = np.asarray(drift_B, dtype=np.float64)
        C = np.asarray(drift_C, dtype=np.float64)
        D = np.asarray(drift_D, dtype=np.float64)

        I_obs = float(np.mean(D) - np.mean(B) - np.mean(C) + np.mean(A))

        boot_Is = np.empty(n_bootstrap)
        for i in range(n_bootstrap):
            a = rng.choice(A, size=len(A), replace=True)
            b = rng.choice(B, size=len(B), replace=True)
            c = rng.choice(C, size=len(C), replace=True)
            d = rng.choice(D, size=len(D), replace=True)
            boot_Is[i] = np.mean(d) - np.mean(b) - np.mean(c) + np.mean(a)

        alpha = 1.0 - confidence
        ci_lower = float(np.percentile(boot_Is, 100 * alpha / 2))
        ci_upper = float(np.percentile(boot_Is, 100 * (1 - alpha / 2)))

        p_value = float(np.mean(boot_Is <= 0))

        logger.info(
            "Bootstrap interaction test: I=%.4f [%.4f, %.4f], p=%.4g",
            I_obs, ci_lower, ci_upper, p_value,
        )

        return {
            "I_observed": I_obs,
            "I_ci_lower": ci_lower,
            "I_ci_upper": ci_upper,
            "p_value": p_value,
            "n_bootstrap": n_bootstrap,
            "bootstrap_distribution": boot_Is,
        }

    @staticmethod
    def fit_interaction_surface(
        alpha_values: np.ndarray,
        beta_values: np.ndarray,
        I_values: np.ndarray,
    ) -> Dict[str, Any]:
        """Fit a parametric surface I(alpha, beta) to sweep data.

        Model: I(alpha, beta) = a * alpha^(-b) * beta^c + d

        This captures the expected behaviour: I increases as alpha decreases
        (more data heterogeneity) and as beta increases (more network
        heterogeneity / lower p_out in SBM).

        Parameters
        ----------
        alpha_values:
            Dirichlet alpha values (1-D).
        beta_values:
            Network heterogeneity parameter values (1-D).
        I_values:
            Corresponding interaction term values (1-D).

        Returns
        -------
        dict with ``"params"`` (a, b, c, d), ``"r_squared"``,
        ``"predicted"`` array, ``"residuals"`` array.
        """
        alpha_arr = np.asarray(alpha_values, dtype=np.float64)
        beta_arr = np.asarray(beta_values, dtype=np.float64)
        I_arr = np.asarray(I_values, dtype=np.float64)

        def model(X, a, b, c, d):
            alpha, beta = X
            return a * np.power(alpha, -b) * np.power(beta, c) + d

        try:
            popt, pcov = optimize.curve_fit(
                model,
                (alpha_arr, beta_arr),
                I_arr,
                p0=[0.1, 0.5, 1.0, 0.0],
                maxfev=10000,
            )

            predicted = model((alpha_arr, beta_arr), *popt)
            residuals = I_arr - predicted
            ss_res = np.sum(residuals ** 2)
            ss_tot = np.sum((I_arr - np.mean(I_arr)) ** 2)
            r_squared = float(1.0 - ss_res / ss_tot) if ss_tot > 1e-15 else 0.0

            logger.info(
                "Surface fit: a=%.4f, b=%.4f, c=%.4f, d=%.4f, R²=%.4f",
                *popt, r_squared,
            )

            return {
                "params": {
                    "a": float(popt[0]),
                    "b": float(popt[1]),
                    "c": float(popt[2]),
                    "d": float(popt[3]),
                },
                "r_squared": r_squared,
                "predicted": predicted,
                "residuals": residuals,
            }

        except (RuntimeError, optimize.OptimizeWarning) as e:
            logger.warning("Surface fit failed: %s", e)
            return {
                "params": {"a": 0, "b": 0, "c": 0, "d": 0},
                "r_squared": 0.0,
                "predicted": np.zeros_like(I_arr),
                "residuals": I_arr.copy(),
            }

    @staticmethod
    def compute_sweep_interactions(
        results_df,
        alpha_col: str = "dirichlet_alpha",
        beta_col: str = "p_out",
        metric_col: str = "final_cka_cross_community",
        baseline_alpha: Optional[float] = None,
        baseline_beta: Optional[float] = None,
    ) -> Dict[str, np.ndarray]:
        """Compute I for each (alpha, beta) point in a parametric sweep.

        For each (alpha, beta), the baseline is taken from the IID and/or
        dense conditions at that same parameter level. If explicit baseline
        values are given, those cells are used.

        Returns
        -------
        dict with ``"alpha_values"``, ``"beta_values"``, ``"I_values"``
        arrays suitable for :meth:`fit_interaction_surface`.
        """
        import pandas as pd

        df = results_df.copy()
        alphas = sorted(df[alpha_col].dropna().unique())
        betas = sorted(df[beta_col].dropna().unique())

        if baseline_alpha is None:
            baseline_alpha = max(alphas)
        if baseline_beta is None:
            baseline_beta = max(betas)

        alpha_out, beta_out, I_out = [], [], []

        drift_A = df.loc[
            (df[alpha_col] == baseline_alpha) & (df[beta_col] == baseline_beta),
            metric_col,
        ].values

        for a_val in alphas:
            for b_val in betas:
                if a_val == baseline_alpha and b_val == baseline_beta:
                    continue

                drift_D = df.loc[
                    (df[alpha_col] == a_val) & (df[beta_col] == b_val),
                    metric_col,
                ].values
                drift_B = df.loc[
                    (df[alpha_col] == a_val) & (df[beta_col] == baseline_beta),
                    metric_col,
                ].values
                drift_C = df.loc[
                    (df[alpha_col] == baseline_alpha) & (df[beta_col] == b_val),
                    metric_col,
                ].values

                if len(drift_D) == 0 or len(drift_B) == 0 or len(drift_C) == 0:
                    continue

                I = (
                    np.mean(drift_D) - np.mean(drift_B)
                    - np.mean(drift_C) + np.mean(drift_A)
                )
                alpha_out.append(a_val)
                beta_out.append(b_val)
                I_out.append(I)

        return {
            "alpha_values": np.array(alpha_out),
            "beta_values": np.array(beta_out),
            "I_values": np.array(I_out),
        }
