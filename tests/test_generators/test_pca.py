import pytest
import numpy as np

import pyFracAggregate as pfa

def test_pca_filippov_generation():
    # Generate 10 particles
    agg = pfa.generate(
        n_particles=10,
        df=1.8,
        kf=1.3,
        method='pca'
    )
    
    assert agg.current_size == 10
    
    # Verify no overlaps (allowing some tolerance)
    # The minimum distance between two particles should be roughly r_i + r_j
    positions = agg.positions
    radii = agg.radii
    overlap_tolerance = 1e-5
    
    for i in range(10):
        for j in range(i + 1, 10):
            dist = np.linalg.norm(positions[i] - positions[j])
            min_dist = radii[i] + radii[j] - overlap_tolerance
            assert dist >= min_dist - 1e-5 # Float precision

def test_pca_filippov_single_particle():
    agg = pfa.generate(n_particles=1, df=1.8, kf=1.3, method='pca')
    assert agg.current_size == 1
    assert np.allclose(agg.positions[0], [0.0, 0.0, 0.0])

def test_invalid_api_parameters():
    with pytest.raises(ValueError, match="df must be in"):
        pfa.generate(n_particles=10, df=3.5, kf=1.3)
        
    with pytest.raises(ValueError, match="Unknown generation method"):
        pfa.generate(n_particles=10, df=1.8, kf=1.3, method='unknown')
