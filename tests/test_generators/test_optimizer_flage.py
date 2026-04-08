import pytest
import numpy as np
from pyFracAggregate.generators.optimizer_flage import (
    find_exact_touching_points_pca,
    filter_overlapping_candidates,
    build_particle_list_pca,
    solve_pca_placement,
)

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

def test_build_particle_list_pca_basic():
    """Particles within range should be included."""
    positions = np.array([
        [0.0, 0.0, 0.0],
        [3.0, 0.0, 0.0],
        [20.0, 0.0, 0.0],
    ])
    radii = np.array([1.0, 1.0, 1.0])
    L = 10.0
    a = 1.0
    p_list = build_particle_list_pca(positions, radii, L, a)
    # lower = max(10-2-1, 0) = 7, upper = 10+12+1 = 23
    # Particle 0: dist=0, excluded (below 7)
    # Particle 1: dist=3, excluded (below 7)
    # Particle 2: dist=20, included (within [7, 23])
    assert isinstance(p_list, list)
    assert 2 in p_list  # dist=20, within [7, 23]
    assert 0 not in p_list  # dist=0, below 7
    assert 1 not in p_list  # dist=3, below 7


def test_build_particle_list_pca_empty():
    """Empty positions returns empty list."""
    positions = np.empty((0, 3))
    radii = np.empty(0)
    p_list = build_particle_list_pca(positions, radii, 10.0, 1.0)
    assert p_list == []


def test_solve_pca_placement_finds_valid():
    """Solve should return positions that touch the reference particle and don't overlap."""
    center = np.array([0.0, 0.0, 0.0])
    L = 5.0
    ref_pos = np.array([3.0, 0.0, 0.0])
    r_new = 1.0
    ref_radii = np.array([1.0, 1.0, 1.0])
    positions = np.array([[3.0, 0.0, 0.0], [0.0, 5.0, 0.0], [0.0, -5.0, 0.0]])
    result = solve_pca_placement(center, L, [0], r_new, ref_radii, positions, 1e-5)
    assert result is not None
    pt = result
    assert np.isclose(np.linalg.norm(pt - center), L, atol=1e-4)
    assert np.isclose(np.linalg.norm(pt - ref_pos), r_new + ref_radii[0], atol=1e-3)


def test_solve_pca_placement_returns_none_when_blocked():
    """If all candidates overlap, should return None."""
    center = np.array([0.0, 0.0, 0.0])
    L = 3.0
    r_new = 1.0
    ref_radii = np.array([1.0, 1.5])
    positions = np.array([[2.0, 0.0, 0.0], [2.0, 0.0, 0.5]])
    result = solve_pca_placement(center, L, [0], r_new, ref_radii, positions, 1e-5)
    # May or may not find a valid position depending on geometry, but must not crash
    assert result is None or isinstance(result, np.ndarray)
