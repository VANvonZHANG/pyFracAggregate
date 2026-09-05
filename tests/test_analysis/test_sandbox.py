import numpy as np
import pytest

import pyFracAggregate as pfa
from pyFracAggregate.core.aggregate import Aggregate
from pyFracAggregate.analysis.sandbox import (
    number_radius_function,
    mass_radius_function,
)


@pytest.fixture(scope="module")
def agg_mono():
    return pfa.generate(100, 1.8, 1.9, method="pca", seed=3)


def test_curve_shape_grid_and_monotonicity(agg_mono):
    r, curve = number_radius_function(agg_mono)
    assert r.shape == curve.shape == (15,)
    assert np.all(np.diff(r) > 0)          # log-spaced grid, strictly increasing
    assert np.all(np.diff(curve) >= 0)     # cumulative curve is non-decreasing
    assert np.all(curve >= 1.0)            # seed particle counts itself


def test_default_window_matches_convention(agg_mono):
    r, _ = number_radius_function(agg_mono)
    assert np.isclose(r[0], float(np.mean(agg_mono.radii)))
    assert np.isclose(r[-1], pfa.radius_of_gyration(agg_mono))


def test_explicit_window_and_bins(agg_mono):
    r, _ = number_radius_function(agg_mono, bins=7, r_min=2.0, r_max=10.0)
    assert len(r) == 7
    assert np.isclose(r[0], 2.0)
    assert np.isclose(r[-1], 10.0)


def test_mass_curve_is_scaled_number_curve_for_monodisperse(agg_mono):
    r_n, c_n = number_radius_function(agg_mono)
    r_m, c_m = mass_radius_function(agg_mono)
    assert np.array_equal(r_n, r_m)
    ratio = c_m / c_n
    assert np.allclose(ratio, ratio[0])    # constant factor r^3 per particle


def test_too_few_particles_returns_empty():
    for agg in (Aggregate(10), Aggregate(1)):
        if agg.current_size == 1:
            agg.add_particle(0.0, 0.0, 0.0, 1.0, 1.0)
        r, curve = number_radius_function(agg)
        assert len(r) == 0 and len(curve) == 0
        r, curve = mass_radius_function(agg)
        assert len(r) == 0 and len(curve) == 0
