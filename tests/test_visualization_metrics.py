"""Tests for drift-oriented figures and run_visualization --metric."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from superadditivity.analysis.aggregator import ResultAggregator
from superadditivity.analysis.drift_metrics import resolve_drift_metric
from superadditivity.evaluation.drift_tracker import DriftTracker
from superadditivity.visualization.interaction_plots import (
    factorial_cell_stats,
    plot_factorial_grid,
    plot_interaction_bar,
)

# (data_regime, network_regime): (final CKA similarity, final MMD drift)
_CELLS = {
    ("iid", "erdos_renyi"): (0.95, 0.001),
    ("dirichlet_semantic", "erdos_renyi"): (0.85, 0.004),
    ("iid", "sbm"): (0.85, 0.004),
    ("dirichlet_semantic", "sbm"): (0.60, 0.015),
}
_ROUNDS = [0, 10, 20]


def _make_results(root: Path, n_seeds: int = 3) -> None:
    """Fake runs with summary.json and drift_metrics.h5, as run_experiment writes."""
    rng = np.random.RandomState(0)
    for (data, net), (cka, mmd) in _CELLS.items():
        for seed in range(n_seeds):
            run_dir = root / f"{data}_{net}" / f"seed_{seed}_graph_100"
            run_dir.mkdir(parents=True)
            final_cka = cka + 0.01 * rng.randn()
            final_mmd = mmd + 0.0005 * rng.randn()
            tracker = DriftTracker(output_path=str(run_dir / "drift_metrics.h5"))
            for i, r in enumerate(_ROUNDS):
                frac = i / (len(_ROUNDS) - 1)
                tracker.record(r, {
                    "cka_layer4_mean": 1.0 - frac * (1.0 - final_cka),
                    "mmd_paired_layer4_drift": frac * final_mmd,
                })
            tracker.save()
            (run_dir / "summary.json").write_text(json.dumps({
                "data_regime": data,
                "network_regime": net,
                "run_seed": seed,
                "graph_seed": 100,
                "final_cka_cross_community": final_cka,
                "final_mmd_paired_drift": final_mmd,
            }))


class TestPlotFunctions:
    def test_factorial_cell_stats(self):
        table = pd.DataFrame({
            "cell": ["A", "A", "B", "C", "D", "D"],
            "metric": [1.0, 3.0, 2.0, 5.0, 4.0, 4.0],
        })
        means, cis, counts = factorial_cell_stats(table)
        assert means == {"A": 2.0, "B": 2.0, "C": 5.0, "D": 4.0}
        assert counts == {"A": 2, "B": 1, "C": 1, "D": 2}
        sem_a = np.std([1.0, 3.0], ddof=1) / np.sqrt(2)
        assert cis["A"] == pytest.approx((2.0 - 1.96 * sem_a, 2.0 + 1.96 * sem_a))
        assert cis["B"] == (2.0, 2.0)

    def test_factorial_grid_shows_drift_not_similarity(self):
        df = pd.DataFrame({
            "data_regime": ["iid", "iid", "noniid", "noniid"],
            "network_regime": ["er", "sbm", "er", "sbm"],
            "final_cka_cross_community": [0.9, 0.8, 0.8, 0.5],
        })
        fig = plot_factorial_grid(df, drift_metric=resolve_drift_metric("cka"))
        annotations = sorted(float(t.get_text()) for t in fig.axes[0].texts)
        assert annotations == pytest.approx([0.1, 0.2, 0.2, 0.5])
        plt.close(fig)

    def test_factorial_grid_missing_column(self):
        df = pd.DataFrame({"data_regime": ["iid"], "network_regime": ["er"]})
        with pytest.raises(ValueError, match="final_mmd_paired_drift"):
            plot_factorial_grid(df, drift_metric=resolve_drift_metric("mmd"))

    def test_bar_draws_additive_prediction(self):
        means = {"A": 0.05, "B": 0.15, "C": 0.15, "D": 0.40}
        fig = plot_interaction_bar(means, interaction_I=0.15, interaction_p=0.001)
        ax = fig.axes[0]
        segments = ax.collections[-1].get_segments()
        assert segments[0][0][1] == pytest.approx(0.25)   # B + C - A
        assert any("I = 0.15" in t.get_text() for t in ax.texts)
        plt.close(fig)


class TestTrajectoryLoading:
    def test_loads_by_recorded_key_with_regimes(self, tmp_path):
        _make_results(tmp_path, n_seeds=1)
        runs = ResultAggregator(str(tmp_path)).load_trajectories("cka_layer4_mean")
        assert len(runs) == 4
        for run in runs:
            np.testing.assert_array_equal(run["rounds"], _ROUNDS)
            assert run["values"][0] == pytest.approx(1.0)
            assert run["network_regime"] in ("erdos_renyi", "sbm")

    def test_missing_key_returns_empty(self, tmp_path):
        _make_results(tmp_path, n_seeds=1)
        assert ResultAggregator(str(tmp_path)).load_trajectories("nope") == []


@pytest.mark.filterwarnings("ignore")
class TestRunVisualization:
    @pytest.mark.parametrize("metric_name,folder", [("cka", "figures"), ("mmd", "figures_mmd")])
    def test_generates_all_figures(self, tmp_path, monkeypatch, metric_name, folder):
        import scripts.run_analysis as ra
        import scripts.run_visualization as rv

        real = ra.functional_anova
        monkeypatch.setattr(
            ra, "functional_anova",
            lambda df, n_permutations=10000, seed=42: real(df, n_permutations=100, seed=seed),
        )
        _make_results(tmp_path)
        metric = resolve_drift_metric(metric_name)
        ra.run_factorial_analysis(
            str(tmp_path), ra.default_output_dir(str(tmp_path), metric), metric
        )

        monkeypatch.setattr(
            sys, "argv",
            ["run_visualization.py", "--results_dir", str(tmp_path), "--metric", metric_name],
        )
        rv.main()

        fig_dir = tmp_path / folder
        for name in ("factorial_grid.pdf", "interaction_bar.pdf", "drift_trajectories.pdf"):
            assert (fig_dir / name).stat().st_size > 0, name

    def test_bar_skipped_when_not_2x2(self, tmp_path):
        import scripts.run_visualization as rv

        _make_results(tmp_path, n_seeds=1)
        # Drop cell B: the design is no longer 2x2.
        import shutil
        shutil.rmtree(tmp_path / "dirichlet_semantic_erdos_renyi")
        agg = ResultAggregator(str(tmp_path))
        assert not rv.plot_interaction_bar_figure(
            agg, resolve_drift_metric("cka"), tmp_path, tmp_path / "bar.pdf"
        )
        assert not (tmp_path / "bar.pdf").exists()
