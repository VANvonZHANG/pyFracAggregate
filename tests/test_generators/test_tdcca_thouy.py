import pytest
import numpy as np

import pyFracAggregate as pfa


def test_tdcca_basic_generation():
    agg = pfa.generate(n_particles=16, df=1.8, kf=1.3, method='tdcca')
    assert agg.current_size == 16


def test_tdcca_power_of_two_required():
    """tdCCA requires N to be a power of 2."""
    with pytest.raises(ValueError, match="power of 2"):
        pfa.generate(n_particles=15, df=1.8, kf=1.3, method='tdcca')


def test_tdcca_no_overlaps():
    agg = pfa.generate(n_particles=8, df=1.5, kf=1.3, method='tdcca')
    positions = agg.positions
    radii = agg.radii
    for i in range(agg.current_size):
        for j in range(i + 1, agg.current_size):
            dist = np.linalg.norm(positions[i] - positions[j])
            min_dist = radii[i] + radii[j] - 1e-5
            assert dist >= min_dist - 1e-5


def test_tdcca_scaling_law():
    agg = pfa.generate(n_particles=32, df=1.8, kf=1.3, method='tdcca')
    rg = pfa.radius_of_gyration(agg)
    a = np.mean(agg.radii)
    df_est = np.log(agg.current_size) / np.log(rg / a)
    assert abs(df_est - 1.8) < 0.5
