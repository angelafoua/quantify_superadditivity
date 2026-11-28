"""Tests for the factorial statistics (ANOVA permutation test, mixed model, report)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from superadditivity.analysis.anova import functional_anova, two_way_anova
from superadditivity.analysis.interaction_analyzer import InteractionAnalyzer
from superadditivity.analysis.mixed_effects import fit_mixed_effects
from superadditivity.analysis.statistical_report import StatisticalReport


def _factorial(interaction=0.0, main=5.0, n_seeds=5, n_graph=3, noise=0.1, seed=0,
               seed_sd=0.3):
    rng = np.random.RandomState(seed)
    seed_effect = rng.normal(0, seed_sd, n_seeds)
    rows = []
    for d in ("iid", "noniid"):
        for net in ("er", "sbm"):
            mu = main * (d == "noniid") + main * (net == "sbm")
            mu += interaction * (d == "noniid") * (net == "sbm")
            for s in range(n_seeds):
                for g in range(n_graph):
                    rows.append({"data_regime": d, "network_regime": net,
                                 "run_seed": s, "graph_seed": g,
                                 "metric": mu + seed_effect[s] + noise * rng.randn()})
    return pd.DataFrame(rows)


class TestFreedmanLane:
    def test_observed_F_matches_statsmodels(self):
        df = _factorial(interaction=0.2)
        fl = functional_anova(df, n_permutations=50, seed=0)
        table = two_way_anova(df)["table"]
        assert fl["observed_F"] == pytest.approx(
            table.loc["C(data_regime):C(network_regime)", "F"], rel=1e-8
        )

    def test_detects_interaction(self):
        res = functional_anova(_factorial(interaction=0.5), n_permutations=999, seed=0)
        assert res["perm_p_value"] == pytest.approx(1 / 1000)

    def test_calibrated_with_large_main_effects_and_no_interaction(self):
        """Strong main effects must not make the interaction test reject.

        Independent errors (seed_sd=0): the setting the F-test assumes.
        Shared seed effects are handled by the paired test and mixed model.
        """
        pvals = [
            functional_anova(
                _factorial(interaction=0.0, seed=s, seed_sd=0.0), n_permutations=199, seed=s
            )[
                "perm_p_value"
            ]
            for s in range(60)
        ]
        assert np.mean(np.array(pvals) <= 0.05) < 0.15
        assert 0.35 < np.mean(pvals) < 0.65

    def test_p_value_never_zero(self):
        res = functional_anova(_factorial(interaction=5.0), n_permutations=20, seed=0)
        assert res["perm_p_value"] > 0


class TestBootstrapFloor:
    def test_independent_bootstrap_p_never_zero(self):
        A, B, C, D = (np.full(5, v) for v in (0.0, 1.0, 1.0, 5.0))
        res = InteractionAnalyzer.bootstrap_interaction_test(A, B, C, D, n_bootstrap=100, seed=0)
        assert res["p_value"] == pytest.approx(1 / 101)


@pytest.mark.filterwarnings("ignore")
class TestMixedEffects:
    def test_crossed_seeds_fit_and_recover_interaction(self):
        res = fit_mixed_effects(_factorial(interaction=0.5))
        key = [k for k in res["params"] if ":" in k and "Var" not in k][0]
        assert res["params"][key] == pytest.approx(0.5, abs=0.15)

    def test_single_graph_seed_falls_back_to_run_seed_only(self):
        res = fit_mixed_effects(_factorial(interaction=0.5, n_graph=1))
        assert res["params"]


class TestReportFormatting:
    def test_floor_p_shown_as_bound(self):
        assert StatisticalReport.format_p(1 / 10001, 10000) == "< 0.0001"
        assert StatisticalReport.format_p(0.0312, 10000) == "0.0312"
        assert StatisticalReport.format_p(float("nan")) == "n/a"
