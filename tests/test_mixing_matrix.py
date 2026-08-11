"""Tests for mixing matrix properties."""

from __future__ import annotations

import networkx as nx
import numpy as np
import pytest

from superadditivity.graphs.mixing_matrix import (
    metropolis_hastings_weights,
    is_doubly_stochastic,
)


def _make_adjacency():
    """Create a simple cycle-graph adjacency matrix."""
    G = nx.cycle_graph(8)
    return nx.to_numpy_array(G, dtype=np.float64)


class TestMixingMatrix:
    def test_doubly_stochastic(self):
        A = _make_adjacency()
        W = metropolis_hastings_weights(A)
        assert is_doubly_stochastic(W), "W must be doubly stochastic"

    def test_symmetric(self):
        A = _make_adjacency()
        W = metropolis_hastings_weights(A)
        np.testing.assert_allclose(W, W.T, atol=1e-12)

    def test_weight_on_edges_only(self):
        A = _make_adjacency()
        W = metropolis_hastings_weights(A)
        n = W.shape[0]
        for i in range(n):
            for j in range(n):
                if i != j and A[i, j] == 0:
                    assert W[i, j] == 0.0, f"W[{i},{j}] should be 0 (no edge)"

    def test_row_sums_to_one(self):
        A = _make_adjacency()
        W = metropolis_hastings_weights(A)
        np.testing.assert_allclose(W.sum(axis=1), 1.0, atol=1e-12)

    def test_col_sums_to_one(self):
        A = _make_adjacency()
        W = metropolis_hastings_weights(A)
        np.testing.assert_allclose(W.sum(axis=0), 1.0, atol=1e-12)
