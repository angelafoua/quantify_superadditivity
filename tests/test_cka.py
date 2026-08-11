"""Tests for CKA computation."""

from __future__ import annotations

import numpy as np
import pytest

from superadditivity.evaluation.cka_analyzer import CKAAnalyzer


class TestCKA:
    def test_identical_features_give_cka_one(self, synthetic_features):
        analyzer = CKAAnalyzer()
        cka = analyzer.compute(synthetic_features, synthetic_features)
        np.testing.assert_allclose(cka, 1.0, atol=1e-10)

    def test_cka_symmetry(self, synthetic_features):
        analyzer = CKAAnalyzer()
        X = synthetic_features
        Y = np.random.randn(*X.shape)
        cka_xy = analyzer.compute(X, Y)
        cka_yx = analyzer.compute(Y, X)
        np.testing.assert_allclose(cka_xy, cka_yx, atol=1e-10)

    def test_cka_range(self, synthetic_features):
        analyzer = CKAAnalyzer()
        Y = np.random.randn(*synthetic_features.shape)
        cka = analyzer.compute(synthetic_features, Y)
        assert 0.0 <= cka <= 1.0 + 1e-10

    def test_orthogonal_features_low_cka(self):
        n = 100
        X = np.zeros((n, 10), dtype=np.float64)
        X[:, :5] = np.random.randn(n, 5)
        Y = np.zeros((n, 10), dtype=np.float64)
        Y[:, 5:] = np.random.randn(n, 5)
        analyzer = CKAAnalyzer()
        cka = analyzer.compute(X, Y)
        assert cka < 0.3
