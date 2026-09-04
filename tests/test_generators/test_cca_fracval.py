import pytest
import numpy as np

import pyFracAggregate as pfa

pytestmark = pytest.mark.filterwarnings("ignore:method='fracval':DeprecationWarning")

def test_cca_fracval_generation():
    # Test polydisperse generation using the FracVAL method
    dist = pfa.LognormalDistribution(mean=1.0, std=1.2)
    agg = pfa.generate(
        n_particles=15,
        df=1.8,
        kf=1.3,
        method='fracval',
        particle_dist=dist
    )

    assert agg.current_size == 15

    positions = agg.positions
    radii = agg.radii
    overlap_tolerance = 1e-4

    # Check for significant overlaps
    for i in range(15):
        for j in range(i + 1, 15):
            dist_val = np.linalg.norm(positions[i] - positions[j])
            min_dist = radii[i] + radii[j] - overlap_tolerance
            assert dist_val >= min_dist - 1e-3

def test_cca_fracval_small_particles():
    # For small particle counts (<= 8), the algorithm automatically falls back to PCA
    agg = pfa.generate(
        n_particles=5,
        df=1.8,
        kf=1.3,
        method='fracval'
    )
    assert agg.current_size == 5


def test_fracval_deterministic_merge():
    agg = pfa.generate(n_particles=50, df=1.8, kf=1.3, method='fracval')
    assert agg.current_size == 50


def test_fracval_no_overlaps():
    agg = pfa.generate(n_particles=30, df=1.8, kf=1.3, method='fracval')
    positions = agg.positions
    radii = agg.radii
    for i in range(agg.current_size):
        for j in range(i + 1, agg.current_size):
            dist = np.linalg.norm(positions[i] - positions[j])
            min_dist = radii[i] + radii[j] - 1e-5
            assert dist >= min_dist - 1e-5


def test_fracval_scaling_law():
    agg = pfa.generate(n_particles=100, df=1.8, kf=1.3, method='fracval')
    rg = pfa.radius_of_gyration(agg)
    a = np.mean(agg.radii)
    df_est = np.log(agg.current_size) / np.log(rg / a)
    assert abs(df_est - 1.8) < 0.4
