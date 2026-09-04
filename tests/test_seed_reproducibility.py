"""T5: seed gives bit-identical reruns; different seeds differ."""
import numpy as np
import pytest
import pyFracAggregate as pfa

CASES = [
    dict(method="pca", scaling="mass", placement="solved"),
    dict(method="pca", scaling="count", placement="sampled"),
    dict(method="cca", scaling="mass", placement="constructed"),
    dict(method="cca", scaling="count", placement="sampled"),
]


@pytest.mark.parametrize("case", CASES)
def test_same_seed_reproduces_exactly(case):
    a = pfa.generate(50, 1.8, 1.3, seed=123, particle_dist=pfa.LognormalDistribution(1.0, 1.6), **case)
    b = pfa.generate(50, 1.8, 1.3, seed=123, particle_dist=pfa.LognormalDistribution(1.0, 1.6), **case)
    assert np.array_equal(a.positions, b.positions)
    assert np.array_equal(a.radii, b.radii)


@pytest.mark.parametrize("case", CASES)
def test_different_seed_differs(case):
    a = pfa.generate(50, 1.8, 1.3, seed=1, **case)
    b = pfa.generate(50, 1.8, 1.3, seed=2, **case)
    assert not np.allclose(a.positions, b.positions)


def test_no_seed_uses_fresh_entropy():
    a = pfa.generate(50, 1.8, 1.3, method="pca")
    b = pfa.generate(50, 1.8, 1.3, method="pca")
    assert not np.array_equal(a.positions, b.positions)


def test_seed_does_not_touch_global_state():
    np.random.seed(999)
    a = pfa.generate(50, 1.8, 1.3, method="pca", seed=0)
    draws = np.random.random(3)
    np.random.seed(999)
    b = pfa.generate(50, 1.8, 1.3, method="pca", seed=0)
    assert np.array_equal(a.positions, b.positions)
    assert np.array_equal(draws, np.random.random(3))
