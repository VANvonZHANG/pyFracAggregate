import pytest
import numpy as np
from pyFracAggregate.core.aggregate import Aggregate

def test_aggregate_init():
    agg = Aggregate(10)
    assert agg.current_size == 0
    assert agg.max_size == 10
    assert agg._data.shape == (10, 5)

    with pytest.raises(ValueError, match="positive"):
        Aggregate(0)
    
    with pytest.raises(ValueError, match="positive"):
        Aggregate(-5)

def test_aggregate_add_particle():
    agg = Aggregate(2)
    agg.add_particle(1.0, 2.0, 3.0, 0.5, 10.0)
    assert agg.current_size == 1
    np.testing.assert_allclose(agg.positions[0], [1.0, 2.0, 3.0])
    assert agg.radii[0] == 0.5
    assert agg.masses[0] == 10.0

    agg.add_particle(4.0, 5.0, 6.0, 1.5, 20.0)
    assert agg.current_size == 2

    # Test out of bounds
    with pytest.raises(RuntimeError, match="capacity exceeded"):
        agg.add_particle(7.0, 8.0, 9.0, 2.0, 30.0)

def test_aggregate_zero_copy_views():
    agg = Aggregate(5)
    agg.add_particle(1.0, 2.0, 3.0, 0.5, 1.0)
    
    pos = agg.positions
    radii = agg.radii
    masses = agg.masses
    
    # Modify views and check if original data is changed (zero-copy)
    pos[0, 0] = 99.0
    assert agg._data[0, 0] == 99.0
    
    radii[0] = 5.0
    assert agg._data[0, 3] == 5.0
    
    masses[0] = 10.0
    assert agg._data[0, 4] == 10.0

def test_aggregate_to_numpy():
    agg = Aggregate(2)
    agg.add_particle(1.0, 2.0, 3.0, 0.5, 10.0)
    data = agg.to_numpy()
    
    assert data.shape == (1, 5)
    np.testing.assert_allclose(data[0], [1.0, 2.0, 3.0, 0.5, 10.0])
    
    # modify data shouldn't affect agg since it's a copy
    data[0, 0] = 99.0
    assert agg.positions[0, 0] == 1.0
