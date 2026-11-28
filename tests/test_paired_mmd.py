"""Tests for client-level paired MMD (PairedMMDAnalyzer).

All tests run on CPU with synthetic embeddings.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from superadditivity.evaluation.cka_analyzer import linear_cka
from superadditivity.evaluation.mmd_analyzer import _rbf_kernel, _unbiased_mmd2
from superadditivity.evaluation.paired_mmd_analyzer import (
    PairedMMDAnalyzer,
    between_within,
    community_permutation_test,
    median_bandwidth,
    multiplier_tag,
    paired_mmd2_matrix,
)


def _naive_paired_mmd2(x: np.ndarray, y: np.ndarray, sigma: float) -> float:
    """Direct double loop over Gretton et al. (2012) eq. 3."""
    n = x.shape[0]

    def k(a, b):
        return np.exp(-np.sum((a - b) ** 2) / (2.0 * sigma ** 2))

    total = 0.0
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            total += k(x[i], x[j]) + k(y[i], y[j]) - k(x[i], y[j]) - k(x[j], y[i])
    return total / (n * (n - 1))


def _as_tensor(arrs) -> torch.Tensor:
    return torch.as_tensor(np.stack(arrs), dtype=torch.float64)


class TestEstimator:
    def test_matches_naive_double_loop(self):
        rng = np.random.RandomState(0)
        clients = [rng.randn(25, 4) + 0.3 * c for c in range(4)]
        sigmas = [0.7, 1.9]
        mats = paired_mmd2_matrix(_as_tensor(clients), sigmas).numpy()
        for s, sigma in enumerate(sigmas):
            for a in range(4):
                for b in range(4):
                    expected = 0.0 if a == b else _naive_paired_mmd2(
                        clients[a], clients[b], sigma
                    )
                    assert mats[s, a, b] == pytest.approx(expected, abs=1e-12)

    def test_symmetric_zero_diagonal(self):
        rng = np.random.RandomState(1)
        X = _as_tensor([rng.randn(30, 6) for _ in range(5)])
        mat = paired_mmd2_matrix(X, [1.0])[0].numpy()
        np.testing.assert_allclose(mat, mat.T, atol=1e-14)
        np.testing.assert_array_equal(np.diag(mat), 0.0)

    def test_identical_clients_give_exact_zero(self):
        """At round 0 all clients share an initialisation: MMD^2 must be 0."""
        rng = np.random.RandomState(2)
        x = rng.randn(40, 8)
        mat = paired_mmd2_matrix(_as_tensor([x, x, x]), [1.3])[0].numpy()
        np.testing.assert_allclose(mat, 0.0, atol=1e-14)

    def test_unbiased_under_pairing_where_legacy_is_biased(self):
        """Paired samples with equal marginals: E[MMD^2] must be 0.

        x_i and y_i share a latent z_i (same probe image), so the
        independent-sample estimator, which keeps k(x_i, y_i) in the cross
        term, is biased downward. The paired U-statistic is not.
        """
        rng = np.random.RandomState(3)
        n, d, reps, sigma = 40, 3, 400, 1.5
        new_vals, legacy_vals = [], []
        for _ in range(reps):
            z = rng.randn(n, d)
            x = z + 0.3 * rng.randn(n, d)
            y = z + 0.3 * rng.randn(n, d)
            new_vals.append(float(paired_mmd2_matrix(_as_tensor([x, y]), [sigma])[0, 0, 1]))
            legacy_vals.append(_unbiased_mmd2(
                _rbf_kernel(x, x, sigma), _rbf_kernel(y, y, sigma), _rbf_kernel(x, y, sigma)
            ))
        new_vals = np.array(new_vals)
        legacy_vals = np.array(legacy_vals)
        se_new = new_vals.std(ddof=1) / np.sqrt(reps)
        se_legacy = legacy_vals.std(ddof=1) / np.sqrt(reps)
        assert abs(new_vals.mean()) < 4 * se_new
        assert legacy_vals.mean() < -8 * se_legacy

    def test_input_validation(self):
        with pytest.raises(ValueError):
            paired_mmd2_matrix(torch.zeros(3, 5), [1.0])
        with pytest.raises(ValueError):
            paired_mmd2_matrix(torch.zeros(2, 1, 3, dtype=torch.float64), [1.0])


class TestBandwidth:
    def test_median_of_pairwise_distances(self):
        rng = np.random.RandomState(4)
        pts = rng.randn(30, 3)
        from scipy.spatial.distance import pdist
        expected = float(np.median(pdist(pts)))
        got = median_bandwidth(torch.as_tensor(pts), subsample=100, seed=0)
        assert got == pytest.approx(expected, rel=1e-12)

    def test_subsample_is_deterministic(self):
        rng = np.random.RandomState(5)
        pts = torch.as_tensor(rng.randn(500, 4))
        a = median_bandwidth(pts, subsample=100, seed=7)
        b = median_bandwidth(pts, subsample=100, seed=7)
        assert a == b

    def test_multiplier_tag(self):
        assert multiplier_tag(0.25) == "bw0p25"
        assert multiplier_tag(2.0) == "bw2"


class TestAggregation:
    def test_between_within_on_crafted_matrix(self):
        labels = np.array([0, 0, 1, 1])
        mat = np.array([
            [0.0, 1.0, 5.0, 7.0],
            [1.0, 0.0, 6.0, 8.0],
            [5.0, 6.0, 0.0, 3.0],
            [7.0, 8.0, 3.0, 0.0],
        ])
        out = between_within(mat, labels)
        assert out["within"] == pytest.approx(2.0)          # (1 + 3) / 2
        assert out["between"] == pytest.approx(6.5)         # (5+7+6+8) / 4
        assert out["drift"] == pytest.approx(4.5)


def _clients(rng, n_comm=2, per_comm=4, n=40, d=3, shift=0.0):
    emb, comm = {}, {}
    cid = 0
    for c in range(n_comm):
        for _ in range(per_comm):
            emb[cid] = rng.randn(n, d) + shift * c
            comm[cid] = c
            cid += 1
    return emb, comm


class TestAnalyzer:
    def test_scale_invariance(self):
        rng = np.random.RandomState(6)
        emb, comm = _clients(rng, shift=0.5)
        analyzer = PairedMMDAnalyzer(n_permutations=0)
        base = analyzer.compute(emb, comm)
        scaled = analyzer.compute({k: 10.0 * v for k, v in emb.items()}, comm)
        assert scaled["drift"] == pytest.approx(base["drift"], rel=1e-9)
        assert scaled["bandwidth"] == pytest.approx(10.0 * base["bandwidth"], rel=1e-9)
        for m in analyzer.bandwidth_multipliers:
            assert scaled["sensitivity"][m]["drift"] == pytest.approx(
                base["sensitivity"][m]["drift"], rel=1e-9
            )

    def test_detects_rotation_that_cka_ignores(self):
        """Community 1 = community 0 rotated: CKA = 1 but MMD drift > 0."""
        rng = np.random.RandomState(7)
        n, d = 80, 4
        scales = np.array([3.0, 1.0, 1.0, 1.0])
        base = rng.randn(n, d) * scales
        theta = np.pi / 2
        R = np.eye(d)
        R[:2, :2] = [[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]]
        emb = {}
        comm = {}
        for cid in range(4):
            noisy = base + 0.05 * rng.randn(n, d)
            emb[cid] = noisy if cid < 2 else noisy @ R.T
            comm[cid] = 0 if cid < 2 else 1
        assert linear_cka(emb[0], emb[2]) > 0.99
        out = PairedMMDAnalyzer(n_permutations=0).compute(emb, comm)
        assert out["drift"] > 10 * abs(out["within"]) + 0.05

    def test_permutation_test_is_calibrated_under_null(self):
        """Exchangeable clients: p-values approximately uniform."""
        rng = np.random.RandomState(8)
        analyzer = PairedMMDAnalyzer(bandwidth_multipliers=[1.0], n_permutations=199)
        pvals = []
        for rep in range(150):
            emb, comm = _clients(rng, n=30)
            analyzer.permutation_seed = rep
            pvals.append(analyzer.compute(emb, comm)["p_value"])
        pvals = np.array(pvals)
        assert 0.40 < pvals.mean() < 0.60
        assert np.mean(pvals <= 0.05) < 0.11

    def test_permutation_test_has_power(self):
        rng = np.random.RandomState(9)
        emb, comm = _clients(rng, shift=1.0)
        out = PairedMMDAnalyzer(n_permutations=999).compute(emb, comm)
        assert out["drift"] > 0
        # 2 communities of 4: only 35 distinct partitions, so the smallest
        # attainable p-value is about 2/70 (the observed one and its mirror).
        assert out["p_value"] < 0.05

    def test_permutation_disabled(self):
        rng = np.random.RandomState(10)
        emb, comm = _clients(rng)
        out = PairedMMDAnalyzer(n_permutations=0).compute(emb, comm)
        assert np.isnan(out["p_value"])

    def test_permutation_function_matches_direct_relabelling(self):
        rng = np.random.RandomState(11)
        mat = np.abs(rng.randn(6, 6))
        mat = (mat + mat.T) / 2
        np.fill_diagonal(mat, 0.0)
        labels = np.array([0, 0, 1, 1, 2, 2])
        p = community_permutation_test(mat, labels, n_permutations=500, seed=3)
        perm_rng = np.random.RandomState(3)
        obs = between_within(mat, labels)["drift"]
        count = sum(
            between_within(mat, perm_rng.permutation(labels))["drift"] >= obs - 1e-12
            for _ in range(500)
        )
        assert p == pytest.approx((count + 1) / 501)

    def test_requires_primary_multiplier(self):
        with pytest.raises(ValueError):
            PairedMMDAnalyzer(bandwidth_multipliers=[0.5, 2.0])

    def test_rejects_mismatched_shapes(self):
        with pytest.raises(ValueError):
            PairedMMDAnalyzer().compute(
                {0: np.zeros((10, 3)), 1: np.zeros((9, 3))}, {0: 0, 1: 1}
            )


class TestEvaluatorIntegration:
    def test_evaluator_records_paired_mmd_metrics(self):
        from torch.utils.data import DataLoader, TensorDataset

        from superadditivity.evaluation import (
            CentroidAnalyzer,
            CKAAnalyzer,
            DriftTracker,
            ExperimentEvaluator,
            FisherAnalyzer,
            RepresentationExtractor,
            RSAAnalyzer,
        )

        torch.manual_seed(0)
        probe = TensorDataset(torch.randn(24, 3, 8, 8), torch.zeros(24, dtype=torch.long))
        loader = DataLoader(probe, batch_size=8, shuffle=False)

        def make_model():
            return torch.nn.Sequential(
                torch.nn.Conv2d(3, 6, 3, padding=1),
                torch.nn.ReLU(),
                torch.nn.AdaptiveAvgPool2d(1),
                torch.nn.Flatten(),
                torch.nn.Linear(6, 4),
            )

        clients = {cid: SimpleNamespace(model=make_model()) for cid in range(4)}
        community_map = {0: 0, 1: 0, 2: 1, 3: 1}
        tracker = DriftTracker()
        evaluator = ExperimentEvaluator(
            extractor=RepresentationExtractor(["2", "4"], device="cpu"),
            cka=CKAAnalyzer(),
            rsa=RSAAnalyzer(),
            mmd=None,
            fisher=FisherAnalyzer(),
            centroid=CentroidAnalyzer(),
            drift_tracker=tracker,
            probe_loader=loader,
            community_map=community_map,
            primary_layer="2",
            paired_mmd=PairedMMDAnalyzer(bandwidth_multipliers=[0.5, 1.0], n_permutations=50),
        )
        scalars = evaluator.evaluate(clients, round_num=0)

        for layer in ("2", "4"):
            for suffix in ("drift", "between", "within", "bandwidth", "p_value",
                           "drift_bw0p5", "between_bw0p5", "within_bw0p5"):
                assert f"mmd_paired_{layer}_{suffix}" in scalars
            assert tracker.get_trajectory(f"mmd_paired_{layer}_matrix").shape == (1, 4, 4)
        assert "mmd_mean" not in scalars  # legacy disabled
        assert "cka_2_mean" in scalars

    def test_rejects_unknown_layer(self):
        from superadditivity.evaluation import (
            CentroidAnalyzer, CKAAnalyzer, DriftTracker, ExperimentEvaluator,
            FisherAnalyzer, RepresentationExtractor, RSAAnalyzer,
        )
        with pytest.raises(ValueError):
            ExperimentEvaluator(
                extractor=RepresentationExtractor(["2"], device="cpu"),
                cka=CKAAnalyzer(), rsa=RSAAnalyzer(), mmd=None,
                fisher=FisherAnalyzer(), centroid=CentroidAnalyzer(),
                drift_tracker=DriftTracker(), probe_loader=None,
                community_map={}, paired_mmd=PairedMMDAnalyzer(),
                paired_mmd_layers=["layer9"],
            )


class TestClientAccuracy:
    def _evaluator(self, accuracy_every, final_round=None):
        from torch.utils.data import DataLoader, TensorDataset

        from superadditivity.evaluation import (
            CentroidAnalyzer, CKAAnalyzer, DriftTracker, ExperimentEvaluator,
            FisherAnalyzer, RepresentationExtractor, RSAAnalyzer,
        )

        torch.manual_seed(0)
        x = torch.randn(40, 3, 8, 8)
        y = torch.randint(0, 4, (40,))
        loader = DataLoader(TensorDataset(x, y), batch_size=16)
        tracker = DriftTracker()
        ev = ExperimentEvaluator(
            extractor=RepresentationExtractor(["2"], device="cpu"),
            cka=CKAAnalyzer(), rsa=RSAAnalyzer(), mmd=None,
            fisher=FisherAnalyzer(), centroid=CentroidAnalyzer(),
            drift_tracker=tracker, probe_loader=loader,
            community_map={0: 0, 1: 0, 2: 1, 3: 1}, test_loader=loader,
            primary_layer="2", accuracy_every=accuracy_every, final_round=final_round,
        )
        return ev, tracker, x, y

    @staticmethod
    def _clients():
        def make():
            return torch.nn.Sequential(
                torch.nn.Conv2d(3, 6, 3, padding=1), torch.nn.ReLU(),
                torch.nn.AdaptiveAvgPool2d(1), torch.nn.Flatten(), torch.nn.Linear(6, 4),
            )
        return {cid: SimpleNamespace(model=make()) for cid in range(4)}

    def test_accuracy_matches_direct_computation(self):
        ev, _, x, y = self._evaluator(accuracy_every=1)
        clients = self._clients()
        out = ev.evaluate_client_accuracy(clients)
        expected = [
            float((clients[c].model.eval()(x).argmax(1) == y).float().mean()) for c in range(4)
        ]
        np.testing.assert_allclose(out["accuracy_clients"], expected)
        assert out["accuracy_client_mean"] == pytest.approx(np.mean(expected))

    def test_schedule(self):
        ev, tracker, _, _ = self._evaluator(accuracy_every=20, final_round=25)
        clients = self._clients()
        for r in (0, 10, 20, 25):
            scalars = ev.evaluate(clients, r)
            assert ("accuracy_client_mean" in scalars) == (r in (0, 20, 25))
        np.testing.assert_array_equal(tracker.get_rounds("accuracy_client_mean"), [0, 20, 25])

    def test_disabled_by_default(self):
        ev, _, _, _ = self._evaluator(accuracy_every=None)
        assert "accuracy_client_mean" not in ev.evaluate(self._clients(), 0)


class TestBandwidthReference:
    def test_within_client_bandwidth_ignores_between_client_offsets(self):
        """Shifting one community far away must not change the kernel width."""
        rng = np.random.RandomState(12)
        emb, comm = _clients(rng, n=40, d=3)
        shifted = {k: v + (100.0 if comm[k] == 1 else 0.0) for k, v in emb.items()}
        within = PairedMMDAnalyzer(n_permutations=0)
        pooled = PairedMMDAnalyzer(n_permutations=0, bandwidth_reference="pooled")
        assert within.compute(shifted, comm)["bandwidth"] == pytest.approx(
            within.compute(emb, comm)["bandwidth"]
        )
        assert pooled.compute(shifted, comm)["bandwidth"] > 10 * pooled.compute(emb, comm)["bandwidth"]

    def test_within_client_matches_direct_median(self):
        from scipy.spatial.distance import pdist

        rng = np.random.RandomState(13)
        X = rng.randn(3, 20, 4)
        from superadditivity.evaluation.paired_mmd_analyzer import within_client_median_bandwidth

        got = within_client_median_bandwidth(torch.as_tensor(X), subsample=10_000, seed=0)
        expected = np.median(np.concatenate([pdist(X[c]) for c in range(3)]))
        assert got == pytest.approx(expected, rel=1e-12)

    def test_invalid_reference(self):
        with pytest.raises(ValueError):
            PairedMMDAnalyzer(bandwidth_reference="global")
