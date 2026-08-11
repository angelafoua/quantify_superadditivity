"""Tests for SBM graph generation."""

from __future__ import annotations

import numpy as np
import pytest

from superadditivity.graphs.sbm_generator import SBMGenerator
from superadditivity.graphs.mixing_matrix import metropolis_hastings_weights, is_doubly_stochastic


class TestSBMGenerator:
    def test_graph_is_connected(self):
        gen = SBMGenerator(n_clients=32, n_communities=4, p_in=0.5, p_out=0.05, seed=42)
        adjacency, community_map = gen.generate()
        n = adjacency.shape[0]
        import networkx as nx
        G = nx.from_numpy_array(adjacency)
        assert nx.is_connected(G)

    def test_correct_number_of_nodes(self):
        gen = SBMGenerator(n_clients=32, n_communities=4, p_in=0.5, p_out=0.05, seed=42)
        adjacency, _ = gen.generate()
        assert adjacency.shape == (32, 32)

    def test_mixing_matrix_doubly_stochastic(self):
        gen = SBMGenerator(n_clients=32, n_communities=4, p_in=0.5, p_out=0.05, seed=42)
        adjacency, _ = gen.generate()
        W = metropolis_hastings_weights(adjacency)
        assert is_doubly_stochastic(W)

    def test_different_seeds_different_graphs(self):
        gen1 = SBMGenerator(n_clients=32, n_communities=4, p_in=0.5, p_out=0.05, seed=42)
        gen2 = SBMGenerator(n_clients=32, n_communities=4, p_in=0.5, p_out=0.05, seed=123)
        adj1, _ = gen1.generate()
        adj2, _ = gen2.generate()
        assert not np.array_equal(adj1, adj2)
