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
