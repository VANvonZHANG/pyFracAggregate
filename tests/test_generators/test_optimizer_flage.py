import pytest
import numpy as np
from pyFracAggregate.generators.optimizer_flage import find_exact_touching_points_pca, filter_overlapping_candidates

def test_find_exact_touching_points():
    center = np.array([0.0, 0.0, 0.0])
    ref_pos = np.array([2.0, 0.0, 0.0])
    L = 3.0
    r_new = 1.0
    r_ref = 1.0
    
    # |CA| = 3
    # |CB| = 2
    # |AB| = 2
    # cos(alpha) = (9 + 4 - 4) / (2 * 3 * 2) = 9/12 = 0.75
    # angle alpha exists.
    
    pts = find_exact_touching_points_pca(center, L, ref_pos, r_new, r_ref, num_points=8)
    
    assert pts.shape == (8, 3)
    
    # Check constraints
    for pt in pts:
        # Distance to center is L
        assert np.isclose(np.linalg.norm(pt - center), L)
        # Distance to ref is r_new + r_ref
        assert np.isclose(np.linalg.norm(pt - ref_pos), r_new + r_ref)

def test_no_intersection():
    center = np.array([0.0, 0.0, 0.0])
    ref_pos = np.array([10.0, 0.0, 0.0])
    L = 2.0
    r_new = 1.0
    r_ref = 1.0
    
    # L + (r_new+r_ref) = 2 + 2 = 4 < 10, so spheres don't intersect.
    pts = find_exact_touching_points_pca(center, L, ref_pos, r_new, r_ref)
    assert pts.shape == (0, 3)
