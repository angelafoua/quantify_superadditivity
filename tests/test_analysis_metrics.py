"""Tests for drift-metric selection, factorial cells and run_analysis --metric."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from superadditivity.analysis.aggregator import ResultAggregator
from superadditivity.analysis.drift_metrics import (
    DRIFT_METRICS,
    assign_factorial_cells,
    resolve_drift_metric,
)


class TestDriftMetric:
    def test_cka_is_converted_to_divergence(self):
        cka = resolve_drift_metric("cka")
        np.testing.assert_allclose(cka.to_drift([1.0, 0.8, 0.25]), [0.0, 0.2, 0.75])

    def test_mmd_is_used_as_is(self):
        mmd = resolve_drift_metric("mmd")
        np.testing.assert_allclose(mmd.to_drift([0.0, 0.01, -0.001]), [0.0, 0.01, -0.001])

    def test_named_metrics_point_at_summary_aliases(self):
        assert DRIFT_METRICS["cka"].column == "final_cka_cross_community"
        assert DRIFT_METRICS["mmd"].column == "final_mmd_paired_drift"

    def test_raw_column_requires_orientation(self):
        with pytest.raises(ValueError, match="orientation"):
            resolve_drift_metric("final_mmd_paired_layer4_drift_bw0p5")
        m = resolve_drift_metric("final_mmd_paired_layer4_drift_bw0p5", "distance")
        assert m.column == "final_mmd_paired_layer4_drift_bw0p5"
        np.testing.assert_allclose(m.to_drift([0.3]), [0.3])

    def test_orientation_cannot_contradict_named_metric(self):
        with pytest.raises(ValueError):
            resolve_drift_metric("cka", "distance")
        assert resolve_drift_metric("cka", "similarity").name == "cka"

    def test_invalid_orientation(self):
        with pytest.raises(ValueError):
            resolve_drift_metric("final_x", "sideways")

    def test_superadditive_similarity_gives_positive_I(self):
        """Raw CKA: D far below additive. In drift orientation I must be > 0."""
        cka = resolve_drift_metric("cka")
        A, B, C, D = cka.to_drift([0.95, 0.85, 0.85, 0.60])
        assert D - B - C + A == pytest.approx(0.15)


def _table(data, network):
    return pd.DataFrame({"data_regime": data, "network_regime": network})


class TestFactorialCells:
    def test_labels_written_by_run_experiment(self):
        df = _table(
            ["iid", "dirichlet_semantic", "iid", "dirichlet_semantic"],
            ["erdos_renyi", "erdos_renyi", "sbm", "sbm"],
        )
        assert list(assign_factorial_cells(df)) == ["A", "B", "C", "D"]

    def test_legacy_directory_labels(self):
        df = _table(
            ["iid", "moderate_noniid", "iid", "moderate_noniid"],
            ["erdos_renyi", "erdos_renyi", "sbm", "sbm"],
        )
        assert list(assign_factorial_cells(df)) == ["A", "B", "C", "D"]

    def test_missing_cell_raises(self):
        df = _table(["iid", "iid", "dirichlet_semantic"], ["erdos_renyi", "sbm", "sbm"])
        with pytest.raises(ValueError, match="B"):
            assign_factorial_cells(df)

    def test_unknown_topology_raises(self):
        df = _table(["iid", "dirichlet_semantic"], ["watts_strogatz", "sbm"])
        with pytest.raises(ValueError, match="watts_strogatz"):
            assign_factorial_cells(df)

    def test_more_than_two_data_levels_raises(self):
        df = _table(
            ["iid", "mild_noniid", "severe_noniid", "iid"],
            ["erdos_renyi", "erdos_renyi", "sbm", "sbm"],
        )
        with pytest.raises(ValueError, match="non-IID"):
            assign_factorial_cells(df)


# ----------------------------------------------------------------------
# End-to-end on synthetic summaries
# ----------------------------------------------------------------------

_CELLS = {
    # (data_regime, network_regime): (CKA similarity, MMD drift)
    ("iid", "erdos_renyi"): (0.95, 0.001),
    ("dirichlet_semantic", "erdos_renyi"): (0.85, 0.004),
    ("iid", "sbm"): (0.85, 0.004),
    ("dirichlet_semantic", "sbm"): (0.60, 0.015),   # superadditive on both
}


def _write_summaries(root: Path, with_mmd: bool = True) -> None:
    rng = np.random.RandomState(0)
    for (data, net), (cka, mmd) in _CELLS.items():
        for seed in range(5):
            run_dir = root / f"{data}_{net}" / f"seed_{seed}_graph_100"
            run_dir.mkdir(parents=True)
            summary = {
                "data_regime": data,
                "network_regime": net,
                "run_seed": seed,
                "graph_seed": 100,
                "final_cka_cross_community": cka + 0.01 * rng.randn(),
            }
            if with_mmd:
                summary["final_mmd_paired_drift"] = mmd + 0.0005 * rng.randn()
            (run_dir / "summary.json").write_text(json.dumps(summary))


class TestAggregatorTable:
    def test_table_is_in_drift_orientation_with_cells(self, tmp_path):
        _write_summaries(tmp_path)
        df = ResultAggregator(str(tmp_path)).build_factorial_table()
        np.testing.assert_allclose(df["metric"], 1.0 - df["metric_raw"])
        assert sorted(df["cell"].unique()) == ["A", "B", "C", "D"]
        assert (df["cell"].value_counts() == 5).all()

    def test_mmd_table(self, tmp_path):
        _write_summaries(tmp_path)
        df = ResultAggregator(str(tmp_path)).build_factorial_table(
            drift_metric=resolve_drift_metric("mmd")
        )
        np.testing.assert_allclose(df["metric"], df["metric_raw"])

    def test_missing_metric_column_raises(self, tmp_path):
        _write_summaries(tmp_path, with_mmd=False)
        with pytest.raises(ValueError, match="final_mmd_paired_drift"):
            ResultAggregator(str(tmp_path)).build_factorial_table(
                drift_metric=resolve_drift_metric("mmd")
            )


@pytest.mark.filterwarnings("ignore")
class TestRunAnalysis:
    @pytest.mark.parametrize("metric_name,folder", [("cka", "analysis"), ("mmd", "analysis_mmd")])
    def test_interaction_test_runs_and_detects_superadditivity(
        self, tmp_path, monkeypatch, metric_name, folder
    ):
        import scripts.run_analysis as ra
        from scripts.run_analysis import default_output_dir, run_factorial_analysis

        # Keep the test fast: fewer permutations in the permutation ANOVA.
        real_functional_anova = ra.functional_anova
        monkeypatch.setattr(
            ra, "functional_anova",
            lambda df, n_permutations=10000, seed=42: real_functional_anova(
                df, n_permutations=200, seed=seed
            ),
        )
        _write_summaries(tmp_path)
        metric = resolve_drift_metric(metric_name)
        out = default_output_dir(str(tmp_path), metric)
        assert Path(out).name == folder
        run_factorial_analysis(str(tmp_path), out, metric)

        result = json.loads((Path(out) / "interaction_test.json").read_text())
        assert result["metric"] == metric_name
        assert result["I_observed"] > 0
        assert result["I_ci_lower"] > 0
        assert result["p_value"] < 0.05
        assert {c: result["cells"][c]["n_runs"] for c in "ABCD"} == dict.fromkeys("ABCD", 5)
        assert result["cells"]["D"]["network_regime"] == "sbm"
        report = (Path(out) / "statistical_report.md").read_text()
        assert metric.drift_definition in report

    def test_fingerprint_depends_on_metric(self, tmp_path):
        from scripts.run_analysis import _summaries_fingerprint

        _write_summaries(tmp_path)
        assert _summaries_fingerprint(
            str(tmp_path), resolve_drift_metric("cka")
        ) != _summaries_fingerprint(str(tmp_path), resolve_drift_metric("mmd"))
