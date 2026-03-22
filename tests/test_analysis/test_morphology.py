import pytest
import numpy as np
from pyFracAggregate.core.aggregate import Aggregate
from pyFracAggregate.analysis.morphology import center_of_mass, radius_of_gyration

def test_center_of_mass_two_spheres():
    agg = Aggregate(2)
    # Two identical spheres at x=1 and x=-1
    agg.add_particle(1.0, 0.0, 0.0, 1.0, 10.0)
    agg.add_particle(-1.0, 0.0, 0.0, 1.0, 10.0)
    
    com = center_of_mass(agg)
    np.testing.assert_allclose(com, [0.0, 0.0, 0.0])

def test_center_of_mass_different_masses():
    agg = Aggregate(2)
    # Sphere 1 at x=2, mass=10
    # Sphere 2 at x=8, mass=20
    # COM should be at (2*10 + 8*20) / 30 = 180 / 30 = 6
    agg.add_particle(2.0, 0.0, 0.0, 1.0, 10.0)
    agg.add_particle(8.0, 0.0, 0.0, 1.0, 20.0)
    
    com = center_of_mass(agg)
    np.testing.assert_allclose(com, [6.0, 0.0, 0.0])

def test_radius_of_gyration_single_sphere():
    agg = Aggregate(1)
    agg.add_particle(0.0, 0.0, 0.0, 2.0, 5.0)
    
    rg = radius_of_gyration(agg)
    # Rg for single sphere = sqrt(3/5) * r
    expected_rg = np.sqrt(3.0 / 5.0) * 2.0
    assert np.isclose(rg, expected_rg)

def test_radius_of_gyration_two_spheres():
    agg = Aggregate(2)
    # Two identical spheres of radius r at distance d from center
    # r = 1.0, mass = 10.0
    # placed at x=2 and x=-2
    agg.add_particle(2.0, 0.0, 0.0, 1.0, 10.0)
    agg.add_particle(-2.0, 0.0, 0.0, 1.0, 10.0)
    
    # com is at 0
    # Rg^2 = 1/M * sum(m_i * (dist_i^2 + 3/5 r_i^2))
    # dist_i = 2 for both
    # Rg^2 = 1/20 * (10 * (4 + 3/5) + 10 * (4 + 3/5)) = 4.6
    # Rg = sqrt(4.6)
    
    rg = radius_of_gyration(agg)
    expected_rg = np.sqrt(4.6)
    assert np.isclose(rg, expected_rg)

def test_empty_aggregate():
    agg = Aggregate(10)
    
    com = center_of_mass(agg)
    np.testing.assert_allclose(com, [0.0, 0.0, 0.0])
    
    rg = radius_of_gyration(agg)
    assert rg == 0.0
