"""Superadditivity interaction-term computation, bootstrap testing, and surface fitting.

This is the core novel analysis module: it computes the interaction term
I = Drift(D) - Drift(B) - Drift(C) + Drift(A) from a 2x2 factorial,
tests whether I > 0 via bootstrap, and fits the I(alpha, beta) surface
from parametric sweeps.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

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
        This treats every run as independent; when runs share seeds across
        cells use :meth:`paired_interaction_test` instead.

        Returns
        -------
        dict with:
            ``"I_observed"`` — point estimate.
            ``"I_ci_lower"``, ``"I_ci_upper"`` — confidence interval.
            ``"p_value"`` — ``(#{bootstrap I <= 0} + 1) / (n_bootstrap + 1)``.
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

        # (count + 1) / (B + 1): never exactly 0 with finite B.
        p_value = float((np.sum(boot_Is <= 0) + 1) / (n_bootstrap + 1))

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

        ``beta`` is ``p_out``, so *lower* beta means *more* network
        heterogeneity. The expected behaviour (I increases as alpha
        decreases and as p_out decreases) corresponds to ``b > 0`` and
        ``c < 0``; neither sign is constrained by the fit.

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
        ``"predicted"`` array, ``"residuals"`` array and ``"fitted"``. If
        there are not more points than parameters, or the fit fails, the
        values are ``nan`` and ``fitted`` is ``False``. R² is descriptive
        only: all points share the reference cells, so they are not
        independent.
        """
        alpha_arr = np.asarray(alpha_values, dtype=np.float64)
        beta_arr = np.asarray(beta_values, dtype=np.float64)
        I_arr = np.asarray(I_values, dtype=np.float64)

        n_params = 4
        if len(I_arr) <= n_params:
            logger.warning(
                "Surface fit skipped: %d point(s) for a %d-parameter model "
                "(need more points than parameters).", len(I_arr), n_params,
            )
            nan = float("nan")
            return {
                "params": {"a": nan, "b": nan, "c": nan, "d": nan},
                "r_squared": nan,
                "predicted": np.full_like(I_arr, np.nan),
                "residuals": np.full_like(I_arr, np.nan),
                "fitted": False,
            }

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
                "fitted": True,
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
            nan = float("nan")
            return {
                "params": {"a": nan, "b": nan, "c": nan, "d": nan},
                "r_squared": nan,
                "predicted": np.full_like(I_arr, np.nan),
                "residuals": np.full_like(I_arr, np.nan),
                "fitted": False,
            }

    @staticmethod
    def paired_interaction_test(
        cells: Dict[str, Any],
        value_col: str = "metric",
        unit_col: str = "run_seed",
        confidence: float = 0.95,
        n_bootstrap: int = 10000,
        seed: Optional[int] = None,
        min_bootstrap_units: int = 10,
    ) -> Dict[str, Any]:
        """Interaction test that pairs runs by experimental unit (run seed).

        Runs that share ``run_seed`` share their initial weights and data
        partition across the four cells, so they are not independent. For
        each unit ``u`` present in all four cells, the cell values are
        averaged over the unit's runs (e.g. over graph seeds), and

        ``I_u = D_u - B_u - C_u + A_u``.

        Inference is on the ``I_u``: a one-sided t-test of ``H0: I <= 0`` and
        a t-based confidence interval. A percentile bootstrap CI over units
        is added only with at least ``min_bootstrap_units`` units: with
        few units it under-covers (about 81% for a nominal 95% at 5 units
        in simulation, against 94% for the t interval).

        Parameters
        ----------
        cells:
            Mapping ``{"A","B","C","D"} -> DataFrame`` with ``value_col``
            and ``unit_col``.
        value_col:
            Column with drift values (higher = more drift).
        unit_col:
            Column identifying the pairing unit.
        confidence:
            Confidence level for both intervals.
        n_bootstrap:
            Bootstrap replicates over units.
        seed:
            RNG seed for the bootstrap.
        min_bootstrap_units:
            Minimum number of units for the bootstrap CI (``nan`` below).

        Returns
        -------
        dict
            ``I_observed`` (mean of ``I_u``), ``I_ci_lower`` /
            ``I_ci_upper`` (t interval), ``p_value`` (one-sided t),
            ``bootstrap_ci_lower`` / ``bootstrap_ci_upper``, ``n_units``,
            ``I_units`` ({unit: I_u}), ``method``. With fewer than 2 units
            the interval and p-value are ``nan``.
        """
        from scipy import stats

        means = {}
        for c in "ABCD":
            df = cells[c]
            means[c] = df.groupby(unit_col)[value_col].mean()
        units = sorted(set.intersection(*(set(m.index) for m in means.values())))
        I_units = np.array([
            means["D"][u] - means["B"][u] - means["C"][u] + means["A"][u]
            for u in units
        ], dtype=np.float64)
        n = len(I_units)
        result: Dict[str, Any] = {
            "method": f"paired by {unit_col} (one-sided t-test on per-unit I)",
            "n_units": n,
            "I_units": {str(u): float(v) for u, v in zip(units, I_units, strict=True)},
            "I_observed": float(I_units.mean()) if n else float("nan"),
            "I_ci_lower": float("nan"),
            "I_ci_upper": float("nan"),
            "p_value": float("nan"),
            "bootstrap_ci_lower": float("nan"),
            "bootstrap_ci_upper": float("nan"),
            "n_bootstrap": n_bootstrap,
        }
        if n < 2:
            logger.warning(
                "Paired interaction test needs >= 2 %s units present in all "
                "cells; got %d. Reporting I without uncertainty.", unit_col, n,
            )
            return result

        mean = I_units.mean()
        sem = I_units.std(ddof=1) / np.sqrt(n)
        alpha = 1.0 - confidence
        if sem > 0:
            t_crit = stats.t.ppf(1 - alpha / 2, df=n - 1)
            result["I_ci_lower"] = float(mean - t_crit * sem)
            result["I_ci_upper"] = float(mean + t_crit * sem)
            result["p_value"] = float(stats.t.sf(mean / sem, df=n - 1))
        else:
            result["I_ci_lower"] = result["I_ci_upper"] = float(mean)
            result["p_value"] = 0.0 if mean > 0 else 1.0

        if n >= min_bootstrap_units:
            rng = np.random.RandomState(seed)
            boot = I_units[rng.randint(0, n, size=(n_bootstrap, n))].mean(axis=1)
            result["bootstrap_ci_lower"] = float(np.percentile(boot, 100 * alpha / 2))
            result["bootstrap_ci_upper"] = float(np.percentile(boot, 100 * (1 - alpha / 2)))

        logger.info(
            "Paired interaction test (%d units): I=%.4g [%.4g, %.4g], p=%.4g",
            n, mean, result["I_ci_lower"], result["I_ci_upper"], result["p_value"],
        )
        return result

    @staticmethod
    def compute_sweep_interactions(
        results_df,
        alpha_col: str = "dirichlet_alpha",
        beta_col: str = "p_out",
        metric_col: str = "final_cka_cross_community",
        baseline_alpha: Optional[float] = None,
        baseline_beta: Optional[float] = None,
        unit_col: str = "run_seed",
        n_bootstrap: int = 10000,
        seed: Optional[int] = 42,
    ) -> Dict[str, np.ndarray]:
        """Compute I for each (alpha, beta) point of a parametric sweep.

        The reference cells are the least heterogeneous levels: IID data
        (``alpha = inf``; IID runs carry ``dirichlet_alpha = inf`` after
        :meth:`ResultAggregator.load_summaries`) and the largest ``p_out``.
        For each non-reference point,

        ``I(a, b) = Drift(a, b) - Drift(a, b0) - Drift(a0, b) + Drift(a0, b0)``.

        Points on the reference row (``a = a0``) or column (``b = b0``) are
        skipped: their I is zero by construction and carries no
        information.

        Parameters
        ----------
        results_df:
            One row per run, with alpha, beta and metric columns (metric in
            drift orientation), and ``unit_col`` for per-point inference.
        alpha_col, beta_col, metric_col:
            Column names.
        baseline_alpha, baseline_beta:
            Reference levels; default ``max(alpha)`` (IID) and
            ``max(beta)`` (densest inter-community connectivity).
        unit_col:
            Pairing unit for the per-point test (see
            :meth:`paired_interaction_test`). Ignored if absent.
        n_bootstrap, seed:
            Passed to :meth:`paired_interaction_test`.

        Returns
        -------
        dict
            Arrays ``alpha_values``, ``beta_values``, ``I_values``,
            ``I_ci_lower``, ``I_ci_upper``, ``p_values``, ``n_units``, plus
            scalars ``baseline_alpha`` and ``baseline_beta``. CI and p are
            ``nan`` for points with fewer than 2 units.
        """
        df = results_df.copy()
        alphas = sorted(df[alpha_col].dropna().unique())
        betas = sorted(df[beta_col].dropna().unique())

        if baseline_alpha is None:
            baseline_alpha = max(alphas)
        if baseline_beta is None:
            baseline_beta = max(betas)
        if not np.isinf(baseline_alpha):
            logger.warning(
                "No IID runs (alpha = inf) in the sweep; using alpha=%s as the "
                "reference data level. I is then relative to that level.",
                baseline_alpha,
            )

        has_units = unit_col in df.columns and df[unit_col].notna().all()
        if not has_units:
            df = df.assign(**{unit_col: 0})

        def cell(a, b):
            return df.loc[(df[alpha_col] == a) & (df[beta_col] == b)]

        out: Dict[str, list] = {k: [] for k in (
            "alpha_values", "beta_values", "I_values", "I_ci_lower",
            "I_ci_upper", "p_values", "n_units",
        )}
        cell_A = cell(baseline_alpha, baseline_beta)
        if cell_A.empty:
            logger.warning(
                "Reference cell (alpha=%s, beta=%s) has no runs.",
                baseline_alpha, baseline_beta,
            )
        for a_val in alphas:
            if a_val == baseline_alpha:
                continue
            for b_val in betas:
                if b_val == baseline_beta:
                    continue
                cells = {
                    "A": cell_A,
                    "B": cell(a_val, baseline_beta),
                    "C": cell(baseline_alpha, b_val),
                    "D": cell(a_val, b_val),
                }
                if any(c.empty for c in cells.values()):
                    continue
                test = InteractionAnalyzer.paired_interaction_test(
                    cells, value_col=metric_col, unit_col=unit_col,
                    n_bootstrap=n_bootstrap, seed=seed,
                )
                if test["n_units"] == 0:
                    continue
                out["alpha_values"].append(float(a_val))
                out["beta_values"].append(float(b_val))
                out["I_values"].append(test["I_observed"])
                out["I_ci_lower"].append(test["I_ci_lower"])
                out["I_ci_upper"].append(test["I_ci_upper"])
                out["p_values"].append(test["p_value"])
                out["n_units"].append(test["n_units"] if has_units else 1)

        result: Dict[str, Any] = {k: np.array(v) for k, v in out.items()}
        result["baseline_alpha"] = float(baseline_alpha)
        result["baseline_beta"] = float(baseline_beta)
        return result
