import pytest
import numpy as np
from pyFracAggregate.core.aggregate import Aggregate
from pyFracAggregate.analysis.correlation import pair_correlation_function

def test_pair_correlation_function_two_particles():
    agg = Aggregate(2)
    agg.add_particle(0.0, 0.0, 0.0, 1.0, 1.0)
    agg.add_particle(10.0, 0.0, 0.0, 1.0, 1.0)
    
    # 2 particles, distance is exactly 10
    r_centers, c_r = pair_correlation_function(agg, bins=20, r_max=12.0)
    
    assert len(r_centers) == 20
    assert len(c_r) == 20
    
    # C(r) = n(r) / (4 * pi * r^2 * h * N)
    # n(r): Only 1 pair of particles with distance 10. Due to count_neighbors 
    # being bidirectional, the count should be 2.
    # Find the bin where r_centers is closest to 10
    bin_idx = np.argmin(np.abs(r_centers - 10.0))
    
    h = 12.0 / 20.0
    N = 2
    r_val = r_centers[bin_idx]
    
    expected_c_r = 2 / (4.0 * np.pi * r_val**2 * h * N)
    assert np.isclose(c_r[bin_idx], expected_c_r)
    
    # Other bins should be 0
    for i in range(20):
        if i != bin_idx:
            assert c_r[i] == 0.0

def test_pair_correlation_function_empty_or_single():
    agg0 = Aggregate(10)
    r_centers, c_r = pair_correlation_function(agg0)
    assert len(r_centers) == 0
    assert len(c_r) == 0
    
    agg1 = Aggregate(1)
    agg1.add_particle(0.0, 0.0, 0.0, 1.0, 1.0)
    r_centers, c_r = pair_correlation_function(agg1)
    assert len(r_centers) == 0
    assert len(c_r) == 0
