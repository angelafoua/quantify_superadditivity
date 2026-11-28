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


class TestClientCKA:
    def test_matrix_matches_pairwise_linear_cka(self):
        import torch

        from superadditivity.evaluation.cka_analyzer import client_cka_matrix, linear_cka

        rng = np.random.RandomState(0)
        X = rng.randn(5, 40, 6)
        X[1] = X[0] @ rng.randn(6, 6)  # linear map of client 0
        M = client_cka_matrix(torch.as_tensor(X)).numpy()
        for i in range(5):
            assert M[i, i] == 1.0
            for j in range(5):
                if i != j:
                    assert M[i, j] == np.float64(M[j, i])
                    np.testing.assert_allclose(M[i, j], linear_cka(X[i], X[j]), atol=1e-12)

    def test_between_within_and_drift(self):
        from superadditivity.evaluation.cka_analyzer import ClientCKAAnalyzer

        rng = np.random.RandomState(1)
        base = [rng.randn(60, 8) for _ in range(2)]
        emb, comm = {}, {}
        for cid in range(6):
            c = cid // 3
            emb[cid] = base[c] + 0.1 * rng.randn(60, 8)  # alike within community
            comm[cid] = c
        out = ClientCKAAnalyzer().compute(emb, comm)
        assert out["within"] > 0.9 > out["between"]
        assert out["drift"] == pytest.approx(out["within"] - out["between"])
        assert out["matrix"].shape == (6, 6)

    def test_identical_clients(self):
        from superadditivity.evaluation.cka_analyzer import ClientCKAAnalyzer

        x = np.random.RandomState(2).randn(30, 4)
        out = ClientCKAAnalyzer().compute({i: x for i in range(4)}, {0: 0, 1: 0, 2: 1, 3: 1})
        assert out["between"] == pytest.approx(1.0) and out["drift"] == pytest.approx(0.0)
