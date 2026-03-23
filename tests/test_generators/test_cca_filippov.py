import pytest
import numpy as np

import pyFracAggregate as pfa

def test_cca_filippov_generation():
    # Generate 15 particles, forcing the use of the CCA method
    agg = pfa.generate(
        n_particles=15,
        df=1.8,
        kf=1.3,
        method='cca'
    )
    
    assert agg.current_size == 15
    
    positions = agg.positions
    radii = agg.radii
    overlap_tolerance = 1e-5
    
    # Check for overlaps
    for i in range(15):
        for j in range(i + 1, 15):
            dist = np.linalg.norm(positions[i] - positions[j])
            min_dist = radii[i] + radii[j] - overlap_tolerance
            assert dist >= min_dist - 1e-4

def test_cca_filippov_small_particles():
    # For small particle counts (<= 8), the algorithm automatically falls back to PCA
    agg = pfa.generate(
        n_particles=5,
        df=1.8,
        kf=1.3,
        method='cca'
    )
    assert agg.current_size == 5
