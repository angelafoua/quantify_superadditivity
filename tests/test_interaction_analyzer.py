"""Tests for the interaction analyzer (superadditivity-specific)."""

from __future__ import annotations

import numpy as np
import pytest

from superadditivity.analysis.interaction_analyzer import InteractionAnalyzer


class TestInteractionAnalyzer:
    def test_additive_case_gives_zero_I(self):
        """When effects are purely additive, I should be ~0."""
        A = np.array([1.0, 1.1, 0.9])
        B = np.array([2.0, 2.1, 1.9])
        C = np.array([3.0, 3.1, 2.9])
        D = np.array([4.0, 4.1, 3.9])
        I = InteractionAnalyzer.compute_interaction(A, B, C, D)
        np.testing.assert_allclose(I, 0.0, atol=0.01)

    def test_superadditive_case_gives_positive_I(self):
        """When D is higher than additive prediction, I > 0."""
        A = np.array([1.0, 1.0, 1.0])
        B = np.array([2.0, 2.0, 2.0])
        C = np.array([3.0, 3.0, 3.0])
        D = np.array([7.0, 7.0, 7.0])  # Additive would be 4
        I = InteractionAnalyzer.compute_interaction(A, B, C, D)
        assert I > 0

    def test_bootstrap_detects_superadditivity(self):
        rng = np.random.RandomState(42)
        A = rng.normal(1.0, 0.1, 20)
        B = rng.normal(2.0, 0.1, 20)
        C = rng.normal(3.0, 0.1, 20)
        D = rng.normal(7.0, 0.1, 20)

        result = InteractionAnalyzer.bootstrap_interaction_test(
            A, B, C, D, n_bootstrap=5000, seed=42,
        )
        assert result["I_observed"] > 0
        assert result["p_value"] < 0.05
        assert result["I_ci_lower"] > 0

    def test_bootstrap_no_false_positive(self):
        rng = np.random.RandomState(42)
        A = rng.normal(1.0, 0.1, 20)
        B = rng.normal(2.0, 0.1, 20)
        C = rng.normal(3.0, 0.1, 20)
        D = rng.normal(4.0, 0.1, 20)  # Purely additive

        result = InteractionAnalyzer.bootstrap_interaction_test(
            A, B, C, D, n_bootstrap=5000, seed=42,
        )
        assert result["p_value"] > 0.01

    def test_surface_fit_returns_params(self):
        alphas = np.array([0.1, 0.5, 1.0, 0.1, 0.5, 1.0])
        betas = np.array([0.01, 0.01, 0.01, 0.1, 0.1, 0.1])
        Is = np.array([5.0, 2.0, 0.5, 1.0, 0.5, 0.1])

        result = InteractionAnalyzer.fit_interaction_surface(alphas, betas, Is)
        assert "params" in result
        assert "r_squared" in result
        assert len(result["predicted"]) == 6


class TestPairedInteractionTest:
    @staticmethod
    def _cells(I_true=0.2, n_units=5, noise=0.01, seed=0):
        import pandas as pd

        rng = np.random.RandomState(seed)
        base = {"A": 0.1, "B": 0.3, "C": 0.3, "D": 0.5 + I_true}
        unit_effect = rng.normal(0, 0.5, n_units)  # shared across cells
        cells = {}
        for c, mu in base.items():
            rows = []
            for u in range(n_units):
                for g in range(3):  # graph seeds
                    rows.append({"run_seed": u, "graph_seed": g,
                                 "metric": mu + unit_effect[u] + noise * rng.randn()})
            cells[c] = pd.DataFrame(rows)
        return cells

    def test_recovers_I_despite_large_shared_unit_effects(self):
        res = InteractionAnalyzer.paired_interaction_test(self._cells(), seed=0)
        assert res["n_units"] == 5
        assert res["I_observed"] == pytest.approx(0.2, abs=0.02)
        assert res["I_ci_lower"] > 0 and res["p_value"] < 0.01

    def test_no_interaction_not_significant(self):
        res = InteractionAnalyzer.paired_interaction_test(self._cells(I_true=0.0), seed=0)
        assert res["I_ci_lower"] < 0 < res["I_ci_upper"]
        assert res["p_value"] > 0.05

    def test_single_unit_gives_nan_uncertainty(self):
        res = InteractionAnalyzer.paired_interaction_test(self._cells(n_units=1))
        assert res["n_units"] == 1
        assert np.isfinite(res["I_observed"])
        assert np.isnan(res["p_value"]) and np.isnan(res["I_ci_lower"])

    def test_per_unit_I_values(self):
        cells = self._cells(noise=0.0)
        res = InteractionAnalyzer.paired_interaction_test(cells)
        for v in res["I_units"].values():
            assert v == pytest.approx(0.2)


class TestSweepInteractions:
    @staticmethod
    def _sweep(n_seeds=3):
        import pandas as pd

        rows = []
        rng = np.random.RandomState(0)
        for seed in range(n_seeds):
            for alpha in (float("inf"), 10.0, 0.5):
                for pout in (0.07, 0.5):
                    data = 0.0 if np.isinf(alpha) else 1.0 / alpha
                    net = 1.0 if pout == 0.07 else 0.0
                    drift = 0.1 + data + net + 2.0 * data * net + 0.01 * rng.randn()
                    rows.append({"dirichlet_alpha": alpha, "p_out": pout,
                                 "run_seed": seed, "metric": drift})
        return pd.DataFrame(rows)

    def test_uses_iid_reference_and_skips_structural_zeros(self):
        out = InteractionAnalyzer.compute_sweep_interactions(
            self._sweep(), metric_col="metric", seed=0
        )
        assert np.isinf(out["baseline_alpha"]) and out["baseline_beta"] == 0.5
        # Only non-reference points: alpha in {10, 0.5} x p_out = 0.07.
        assert sorted(zip(out["alpha_values"], out["beta_values"], strict=False)) == [(0.5, 0.07), (10.0, 0.07)]
        expected = {10.0: 2.0 * 0.1, 0.5: 2.0 * 2.0}
        for a, I in zip(out["alpha_values"], out["I_values"], strict=False):
            assert I == pytest.approx(expected[a], abs=0.05)
        assert (out["n_units"] == 3).all()
        assert np.all(np.isfinite(out["p_values"]))

    def test_iid_summaries_get_infinite_alpha(self, tmp_path):
        import json

        from superadditivity.analysis.aggregator import ResultAggregator

        for reg, alpha in (("iid", None), ("dirichlet_semantic", 0.5)):
            d = tmp_path / f"pout0.07_{reg}" / "seed_42_graph_100"
            d.mkdir(parents=True)
            (d / "summary.json").write_text(json.dumps(
                {"data_regime": reg, "dirichlet_alpha": alpha, "network_regime": "sbm"}
            ))
        df = ResultAggregator(str(tmp_path)).load_summaries()
        alphas = dict(zip(df["data_regime"], df["dirichlet_alpha"], strict=False))
        assert np.isinf(alphas["iid"]) and alphas["dirichlet_semantic"] == 0.5
