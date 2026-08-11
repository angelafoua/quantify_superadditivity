"""Tests for MMD computation."""

from __future__ import annotations

import numpy as np
import pytest

from superadditivity.evaluation.mmd_analyzer import MMDAnalyzer


class TestMMD:
    def test_identical_distributions_low_mmd(self):
        rng = np.random.RandomState(42)
        X = rng.randn(100, 10)
        analyzer = MMDAnalyzer()
        mmd2, p_value = analyzer.compute(X, X)
        assert mmd2 < 0.01

    def test_different_distributions_high_mmd(self):
        rng = np.random.RandomState(42)
        X = rng.randn(100, 10)
        Y = rng.randn(100, 10) + 5.0
        analyzer = MMDAnalyzer()
        mmd2, p_value = analyzer.compute(X, Y)
        assert mmd2 > 0.1

    def test_mmd_non_negative(self):
        rng = np.random.RandomState(42)
        X = rng.randn(50, 10)
        Y = rng.randn(50, 10) + 1.0
        analyzer = MMDAnalyzer()
        mmd2, p_value = analyzer.compute(X, Y)
        assert mmd2 >= -0.01

    def test_permutation_pvalue_significant_for_different_distributions(self):
        rng = np.random.RandomState(42)
        X = rng.randn(50, 10)
        Y = rng.randn(50, 10) + 5.0
        analyzer = MMDAnalyzer(n_permutations=100, seed=42)
        mmd2, p_value = analyzer.compute(X, Y)
        assert p_value < 0.05
