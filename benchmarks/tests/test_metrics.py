"""Unit tests for benchmark metrics (no grid runs needed, seconds each)."""
import numpy as np
import pytest
from benchmarks.metrics import (
    pair_count_cr, fit_df, ks_lognormal, overlap_worst, evaluate_run,
)
import pyFracAggregate as pfa


def test_cr_normalization_random_cloud():
    rng = np.random.default_rng(0)
    pos = rng.uniform(0, 10, size=(200, 3))
    rc, cr, dr = pair_count_cr(pos, n_bins=50)
    integral = np.sum(cr * 4 * np.pi * rc**2 * dr)
    assert abs(integral / ((len(pos) - 1) / 2) - 1) < 0.02


def test_df_est_against_analytic_structures():
    """Estimator anchored to analytic truth: RW chain Df=2, uniform ball Df=3."""
    rng = np.random.default_rng(2)
    # random-walk chain of unit steps: pair scaling exponent is exactly 2
    steps = rng.normal(size=(800, 3))
    steps /= np.linalg.norm(steps, axis=1, keepdims=True)
    rw = np.cumsum(steps, axis=0)
    rc, cr, dr = pair_count_cr(rw)
    rg = float(np.sqrt(np.mean(np.sum((rw - rw.mean(axis=0)) ** 2, axis=1))))
    df_rw, _, n_rw = fit_df(rc, cr, 2.0, 0.5 * rg)
    # uniform points in a ball: Df = 3
    u = rng.normal(size=(1000, 3))
    u /= np.linalg.norm(u, axis=1, keepdims=True)
    rad = 20.0 * rng.uniform(0.0, 1.0, size=1000) ** (1.0 / 3.0)
    ball = u * rad[:, None]
    rc, cr, dr = pair_count_cr(ball)
    df_ball, _, n_ball = fit_df(rc, cr, 2.0, 5.0)
    assert n_rw >= 3 and n_ball >= 3
    assert abs(df_rw - 2.0) < 0.8
    assert abs(df_ball - 3.0) < 0.8
    assert df_ball > df_rw


def test_ks_lognormal_matches_own_distribution():
    rng = np.random.default_rng(1)
    r = rng.lognormal(np.log(1.0), np.log(2.0), size=4000)
    assert ks_lognormal(r, 1.0, 2.0) < 0.03


def test_ks_lognormal_rejects_wrong_sg():
    rng = np.random.default_rng(1)
    r = rng.lognormal(np.log(1.0), np.log(2.0), size=4000)
    assert ks_lognormal(r, 1.0, 1.2) > ks_lognormal(r, 1.0, 2.0)


def test_overlap_stats():
    pos = np.array([[0, 0, 0], [1.5, 0, 0]])
    rad = np.array([1.0, 1.0])
    assert 0.4 < overlap_worst(pos, rad) < 0.6
    pos_t = np.array([[0, 0, 0], [2.0, 0, 0]])
    assert overlap_worst(pos_t, rad) <= 1e-9
    pos_f = np.array([[0, 0, 0], [10, 0, 0]])
    assert overlap_worst(pos_f, rad) == 0.0


def test_evaluate_run_end_to_end():
    spec = dict(exp=1, method="pca", placement="solved", beta=None,
                N=50, df=1.8, kf=1.3, sg=1.0, seed=0)
    np.random.seed(0)
    agg = pfa.generate(n_particles=50, df=1.8, kf=1.3, method="pca")
    row = evaluate_run(spec, agg, elapsed=1.234)
    for key in ("df_est", "df_est_r2", "fit_npts", "norm_err", "rg",
                "rg_target", "ks_d", "overlap_worst"):
        assert key in row
    assert row["rg_target"] == pytest.approx(1.0 * (50 / 1.3) ** (1 / 1.8))
    assert 0 < row["fit_npts"] <= 50
