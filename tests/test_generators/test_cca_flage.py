import pytest
import numpy as np
import pyFracAggregate as pfa


def test_flage_cca_basic_generation():
    agg = pfa.generate(n_particles=50, df=1.8, kf=1.3, method='flage_cca')
    assert agg.current_size == 50


def test_flage_cca_no_overlaps():
    agg = pfa.generate(n_particles=30, df=1.8, kf=1.3, method='flage_cca')
    positions = agg.positions
    radii = agg.radii
    for i in range(agg.current_size):
        for j in range(i + 1, agg.current_size):
            dist = np.linalg.norm(positions[i] - positions[j])
            min_dist = radii[i] + radii[j] - 1e-5
            assert dist >= min_dist - 1e-5


def test_flage_cca_scaling_law():
    """CCA should approximately satisfy N = kf * (Rg/a)^Df."""
    agg = pfa.generate(n_particles=100, df=1.8, kf=1.3, method='flage_cca')
    rg = pfa.radius_of_gyration(agg)
    a = np.mean(agg.radii)
    df_est = np.log(agg.current_size) / np.log(rg / a)
    assert abs(df_est - 1.8) < 0.4
