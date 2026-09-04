import pytest
import numpy as np
from pyFracAggregate.generators.placement.solvers import (
    build_particle_list_pca,
    filter_overlapping_candidates,
    solve_tangency,
    mc_touch_merge,
    mc_touch_place,
)


def test_solve_tangency():
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

    pts = solve_tangency(center, L, ref_pos, r_new, r_ref, num_points=8)

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
    pts = solve_tangency(center, L, ref_pos, r_new, r_ref)
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


from pyFracAggregate.core.aggregate import Aggregate


def _two_cluster_set():
    mass = (4.0 / 3.0) * np.pi
    agg1 = Aggregate(3); agg2 = Aggregate(3)
    for agg, xs in ((agg1, [0.0, 2.0, 0.0]), (agg2, [0.0, 2.0, 0.0])):
        for i, x in enumerate(xs):
            pos = np.zeros(3); pos[0] = x
            agg.add_particle(pos[0], 1.0, 0.0, 1.0, mass)
    return agg1, agg2


def test_mc_touch_place_finds_contact():
    agg1, _ = _two_cluster_set()
    np.random.seed(1)
    result = mc_touch_place(agg1, 1.0, np.zeros(3), 4.0, 1.0, 1e-5)
    assert result is not None and len(result) == 3
    pos = np.asarray(result)
    assert np.linalg.norm(pos) == pytest.approx(4.0, abs=1e-9)


def test_mc_touch_merge_returns_valid_shape():
    agg1, agg2 = _two_cluster_set()
    np.random.seed(1)
    out = mc_touch_merge(agg1.positions, agg1.radii, agg2.positions, agg2.radii,
                         5.0, 1.0, 1e-5)
    assert out is not None and out.shape == (3, 3)


def test_mc_touch_merge_track_best_never_returns_none():
    agg1, agg2 = _two_cluster_set()
    np.random.seed(1)
    out = mc_touch_merge(agg1.positions, agg1.radii, agg2.positions, agg2.radii,
                         5.0, 1.0, 1e-5, track_best=True, max_attempts=100)
    assert out is not None and out.shape == (3, 3)
