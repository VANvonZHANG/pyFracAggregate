import pytest
import numpy as np
import pyFracAggregate as pfa

pytestmark = pytest.mark.filterwarnings("ignore::DeprecationWarning")


def test_cca_generation():
    agg = pfa.generate(n_particles=15, df=1.8, kf=1.3, method='cca')
    assert agg.current_size == 15

    positions = agg.positions
    radii = agg.radii
    overlap_tolerance = 1e-5

    for i in range(15):
        for j in range(i + 1, 15):
            dist = np.linalg.norm(positions[i] - positions[j])
            min_dist = radii[i] + radii[j] - overlap_tolerance
            assert dist >= min_dist - 1e-4


def test_cca_small_particles():
    agg = pfa.generate(n_particles=5, df=1.8, kf=1.3, method='cca')
    assert agg.current_size == 5


def test_cca_random_placement():
    agg = pfa.generate(n_particles=30, df=1.8, kf=1.3, method='cca', placement='random')
    assert agg.current_size == 30


def test_cca_random_placement_no_overlaps():
    agg = pfa.generate(n_particles=20, df=1.8, kf=1.3, method='cca', placement='random')
    positions = agg.positions
    radii = agg.radii
    for i in range(agg.current_size):
        for j in range(i + 1, agg.current_size):
            dist = np.linalg.norm(positions[i] - positions[j])
            min_dist = radii[i] + radii[j] - 1e-5
            assert dist >= min_dist - 1e-5


def test_cca_scaling_law():
    agg = pfa.generate(n_particles=100, df=1.8, kf=1.3, method='cca')
    rg = pfa.radius_of_gyration(agg)
    a = np.mean(agg.radii)
    df_est = np.log(agg.current_size) / np.log(rg / a)
    assert abs(df_est - 1.8) < 0.4
