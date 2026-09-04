"""Regression tests for the surface_beta prefilter parameter."""
import numpy as np
import pytest
import pyFracAggregate as pfa
from pyFracAggregate.generators.placement.solved import SolvedPlacement


SNAP = np.load("tests/fixtures/surface_beta_snapshot.npy")


def test_surface_beta_default_attr():
    assert SolvedPlacement().surface_beta == 0.3


def test_snapshot_default_beta_unchanged():
    np.random.seed(42)
    agg = pfa.generate(n_particles=100, df=1.8, kf=1.3, method="cca")
    assert agg.positions.shape == SNAP.shape
    assert np.allclose(agg.positions, SNAP)


@pytest.mark.parametrize("beta", [0.0, 0.7, 1.0])
def test_surface_beta_accepts_values(beta):
    np.random.seed(0)
    agg = pfa.generate(n_particles=50, df=1.8, kf=1.3, method="cca",
                       surface_beta=beta)
    assert agg.positions.shape == (50, 3)


def test_surface_beta_explicit_default_equals_implicit():
    np.random.seed(7)
    a = pfa.generate(n_particles=50, df=1.8, kf=1.3, method="pca", surface_beta=0.3)
    np.random.seed(7)
    b = pfa.generate(n_particles=50, df=1.8, kf=1.3, method="pca")
    assert np.allclose(a.positions, b.positions)


def test_surface_beta_rejected_for_random_placement():
    with pytest.raises(TypeError):
        pfa.generate(n_particles=20, df=1.8, kf=1.3, method="pca",
                     placement="random", surface_beta=0.5)


def test_surface_beta_rejected_for_fracval():
    with pytest.raises(TypeError):
        pfa.generate(n_particles=20, df=1.8, kf=1.3, method="fracval",
                     surface_beta=0.5)
