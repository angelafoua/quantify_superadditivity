"""Integration tests — verify the full pipeline on synthetic data."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from superadditivity.utils.seed import set_all_seeds


class TestIntegration:
    def test_model_creation_and_cloning(self):
        from superadditivity.models.resnet import build_resnet18_cifar
        from superadditivity.models.convnet import SimpleConvNet
        from superadditivity.models.model_utils import clone_model, init_weights

        set_all_seeds(42)
        resnet = build_resnet18_cifar(num_classes=10)
        init_weights(resnet, seed=42)

        resnet_clone = clone_model(resnet)
        for (n1, p1), (n2, p2) in zip(
            resnet.state_dict().items(), resnet_clone.state_dict().items()
        ):
            assert n1 == n2
            torch.testing.assert_close(p1, p2)

        convnet = SimpleConvNet(in_channels=3, num_classes=10)
        init_weights(convnet, seed=42)
        x = torch.randn(2, 3, 32, 32)
        out = convnet(x)
        assert out.shape == (2, 10)

    def test_graph_pipeline(self):
        from superadditivity.graphs.sbm_generator import SBMGenerator
        from superadditivity.graphs.erdos_renyi_generator import ErdosRenyiGenerator
        from superadditivity.graphs.watts_strogatz_generator import WattsStrogatzGenerator
        from superadditivity.graphs.mixing_matrix import (
            metropolis_hastings_weights,
            is_doubly_stochastic,
        )
        from superadditivity.graphs.graph_metrics import compute_spectral_gap
        import networkx as nx

        # SBM
        sbm = SBMGenerator(n_clients=16, n_communities=2, p_in=0.6, p_out=0.1, seed=42)
        adj_sbm, comm_sbm = sbm.generate()
        G = nx.from_numpy_array(adj_sbm)
        assert nx.is_connected(G)
        W = metropolis_hastings_weights(adj_sbm)
        assert is_doubly_stochastic(W)
        sg = compute_spectral_gap(W)
        assert 0.0 < sg <= 1.0

        # ER
        er = ErdosRenyiGenerator(n_clients=16, seed=42, p=0.3, n_communities=2)
        adj_er, comm_er = er.generate()
        W_er = metropolis_hastings_weights(adj_er)
        assert is_doubly_stochastic(W_er)
        sg_er = compute_spectral_gap(W_er)
        assert 0.0 < sg_er <= 1.0

        # WS
        ws = WattsStrogatzGenerator(
            n_clients=16, k_neighbors=4, rewire_prob=0.2, seed=42, n_communities=2
        )
        adj_ws, comm_ws = ws.generate()
        W_ws = metropolis_hastings_weights(adj_ws)
        assert is_doubly_stochastic(W_ws)
        sg_ws = compute_spectral_gap(W_ws)
        assert 0.0 < sg_ws <= 1.0

    def test_lr_schedule(self):
        from superadditivity.training.lr_schedule import CosineDecaySchedule

        sched = CosineDecaySchedule(lr_max=0.1, lr_min=1e-4, total_rounds=100, warmup_rounds=5)
        assert sched.get_lr(0) < sched.get_lr(4)
        assert abs(sched.get_lr(5) - 0.1) < 1e-6
        assert sched.get_lr(100) < sched.get_lr(5)

    def test_seed_reproducibility(self):
        set_all_seeds(42)
        a = torch.randn(10)
        set_all_seeds(42)
        b = torch.randn(10)
        torch.testing.assert_close(a, b)

    def test_analysis_pipeline(self):
        from superadditivity.analysis.interaction_analyzer import InteractionAnalyzer
        from superadditivity.analysis.effect_sizes import bootstrap_ci

        rng = np.random.RandomState(42)
        data = rng.normal(5.0, 1.0, 30)
        point, lo, hi = bootstrap_ci(data, n_bootstrap=1000, seed=42)
        assert lo < point < hi
        assert lo > 3.0
        assert hi < 7.0

        A = rng.normal(1, 0.1, 10)
        B = rng.normal(2, 0.1, 10)
        C = rng.normal(3, 0.1, 10)
        D = rng.normal(6.5, 0.1, 10)
        I = InteractionAnalyzer.compute_interaction(A, B, C, D)
        assert I > 0
