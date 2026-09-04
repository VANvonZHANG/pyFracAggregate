"""T6 literature anchors + T7 regression anchors (statistical).

The v0.3.0 bit gates in test_regression_v030.py are retired after the RNG
switch (plan deviation N9); these anchors assert morphology recovery instead.
Estimator note (Task 0 finding): df recovery uses the Rg-based estimator
log(N)/log(Rg/a) — the same estimator the pre-existing scaling-law tests
use — while baseline_v030_stats.json df_est values come from the
pair-correlation fit; the two estimators are NOT comparable, so T7 anchors
Rg for pca (seed-insensitive by construction) and df recovery against the
target parameter.
"""
import json
from pathlib import Path

import numpy as np
import pytest
import pyFracAggregate as pfa

FIX = Path(__file__).resolve().parents[1] / "fixtures"
STATS = json.loads((FIX / "baseline_v030_stats.json").read_text())


def _df_est(agg):
    rg = pfa.radius_of_gyration(agg)
    return np.log(agg.current_size) / np.log(rg / np.mean(agg.radii))


class TestT6LiteratureAnchors:
    def test_fracval_coordinate_recovers_df(self):
        # Moran 2019 parameter family: N=256, Df=1.8, kf=1.9, lognormal(15, 1.6)
        agg = pfa.generate(256, 1.8, 1.9, method="cca", scaling="mass",
                           placement="constructed", seed=11,
                           particle_dist=pfa.LognormalDistribution(15.0, 1.6))
        assert abs(_df_est(agg) - 1.8) < 0.3

    def test_filippov_coordinate_recovers_df(self):
        # Filippov 2000 family: (cca, count, sampled)
        agg = pfa.generate(200, 1.8, 1.3, method="cca", scaling="count",
                           placement="sampled", seed=11)
        assert abs(_df_est(agg) - 1.8) < 0.5


class TestT7RegressionAnchors:
    @pytest.mark.parametrize("case,kwargs", [
        ("pca_default", dict(method="pca", scaling="count", placement="solved")),
        ("cca_default", dict(method="cca", scaling="count", placement="solved")),
        ("cca_poly", dict(method="cca", scaling="count", placement="solved",
                          particle_dist=pfa.LognormalDistribution(1.0, 1.6))),
    ])
    def test_df_recovery_not_degraded(self, case, kwargs):
        agg = pfa.generate(100, 1.8, 1.3, seed=0, **kwargs)
        assert abs(_df_est(agg) - 1.8) < 0.5

    def test_pca_rg_is_seed_insensitive(self):
        # v0.3.0 behavior: pca Rg is essentially fixed by (N, df, kf);
        # the new default (mass on monodisperse) must preserve that and the
        # baseline value within 2%.
        agg = pfa.generate(100, 1.8, 1.3, method="pca", scaling="mass",
                           placement="solved", seed=0)
        assert abs(pfa.radius_of_gyration(agg) - STATS["pca_default"]["rg"]) / \
            STATS["pca_default"]["rg"] < 0.02
