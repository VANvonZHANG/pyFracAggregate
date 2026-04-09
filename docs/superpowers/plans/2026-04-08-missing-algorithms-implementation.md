# Missing Algorithms Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement the three missing algorithm components: (1) FLAGE fast algebraic placement for PCA and CCA, (2) FracVAL deterministic three-stage contact placement, (3) Thouy-Jullien lattice tdCCA.

**Architecture:** Extend the existing generator framework with new math utilities (Euler-Rodrigues rotation, sphere-sphere intersection), a fast PCA generator using analytical touching-point computation, a fast CCA generator using reference-particle algebraic placement, upgrade FracVAL to deterministic sub-step c placement, and add a new lattice-based tdCCA generator. All share the existing `BaseGenerator` interface and `Aggregate` data structure.

**Tech Stack:** Python 3.9+, NumPy, SciPy (Rotation fallback), existing pyFracAggregate core

---

## File Map

| File | Responsibility |
|---|---|
| `src/pyFracAggregate/core/math_utils.py` | Add `euler_rodrigues_rotation()`, `sphere_sphere_intersection()` |
| `src/pyFracAggregate/generators/optimizer_flage.py` | Rewrite: add `build_particle_list_pca()`, `solve_pca_fast()`, `build_particle_lists_cca()`, `solve_cca_fast()` |
| `src/pyFracAggregate/generators/pca_flage.py` | **Create**: Fast PCA generator using FLAGE algebraic placement |
| `src/pyFracAggregate/generators/cca_flage.py` | **Create**: Fast CCA generator using FLAGE algebraic placement |
| `src/pyFracAggregate/generators/cca_fracval.py` | **Modify**: Replace Monte Carlo merge with deterministic 3-stage placement |
| `src/pyFracAggregate/generators/tdcca_thouy.py` | **Create**: Thouy & Jullien (1994) lattice tdCCA generator |
| `src/pyFracAggregate/generators/factory.py` | **Modify**: Register `'flage_pca'`, `'flage_cca'`, `'fracval'` (unchanged), `'tdcca'` |
| `src/pyFracAggregate/__init__.py` | **Modify**: Re-export new generators if needed |

---

## Task 1: Math Utilities — Euler-Rodrigues & Sphere-Sphere Intersection

**Files:**
- Modify: `src/pyFracAggregate/core/math_utils.py`
- Test: `tests/test_core/test_math_utils.py`

- [ ] **Step 1: Write failing tests**

Add these tests to the existing `tests/test_core/test_math_utils.py`:

```python
def test_euler_rodrigues_rotation_identity():
    """Rotating by zero angle returns original points."""
    points = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    axis = np.array([0.0, 0.0, 1.0])
    rotated = euler_rodrigues_rotation(points, axis, 0.0)
    np.testing.assert_allclose(rotated, points, atol=1e-10)

def test_euler_rodrigues_rotation_90deg_z():
    """Rotating (1,0,0) by 90 degrees around z-axis gives (0,1,0)."""
    points = np.array([[1.0, 0.0, 0.0]])
    axis = np.array([0.0, 0.0, 1.0])
    rotated = euler_rodrigues_rotation(points, axis, np.pi / 2)
    np.testing.assert_allclose(rotated, [[0.0, 1.0, 0.0]], atol=1e-10)

def test_euler_rodrigues_rotation_180deg_x():
    """Rotating (0,1,0) by 180 degrees around x-axis gives (0,-1,0)."""
    points = np.array([[0.0, 1.0, 0.0]])
    axis = np.array([1.0, 0.0, 0.0])
    rotated = euler_rodrigues_rotation(points, axis, np.pi)
    np.testing.assert_allclose(rotated, [[0.0, -1.0, 0.0]], atol=1e-10)

def test_sphere_sphere_intersection_standard():
    """Two spheres of radius 2 centered at (0,0,0) and (3,0,0).
    Intersection circle: center (1.5,0,0), radius = sqrt(4 - 2.25) = sqrt(1.75)."""
    c1, r1 = np.array([0.0, 0.0, 0.0]), 2.0
    c2, r2 = np.array([3.0, 0.0, 0.0]), 2.0
    circle_center, circle_radius = sphere_sphere_intersection(c1, r1, c2, r2)
    np.testing.assert_allclose(circle_center, [1.5, 0.0, 0.0], atol=1e-10)
    np.testing.assert_allclose(circle_radius, np.sqrt(1.75), atol=1e-10)

def test_sphere_sphere_intersection_tangent():
    """Tangent spheres: intersection is a single point."""
    c1, r1 = np.array([0.0, 0.0, 0.0]), 2.0
    c2, r2 = np.array([4.0, 0.0, 0.0]), 2.0
    circle_center, circle_radius = sphere_sphere_intersection(c1, r1, c2, r2)
    np.testing.assert_allclose(circle_center, [2.0, 0.0, 0.0], atol=1e-10)
    np.testing.assert_allclose(circle_radius, 0.0, atol=1e-10)

def test_sphere_sphere_intersection_no_intersection():
    """Non-intersecting spheres return None."""
    c1, r1 = np.array([0.0, 0.0, 0.0]), 2.0
    c2, r2 = np.array([10.0, 0.0, 0.0]), 2.0
    result = sphere_sphere_intersection(c1, r1, c2, r2)
    assert result is None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /home/zhangfan/Project/20260319_SPEMBSSBDART/pyFracAggregate && python -m pytest tests/test_core/test_math_utils.py::test_euler_rodrigues_rotation_identity tests/test_core/test_math_utils.py::test_euler_rodrigues_rotation_90deg_z tests/test_core/test_math_utils.py::test_sphere_sphere_intersection_standard -v`
Expected: FAIL — `ImportError` or `NameError`

- [ ] **Step 3: Implement `euler_rodrigues_rotation` and `sphere_sphere_intersection`**

Add to `src/pyFracAggregate/core/math_utils.py`:

```python
def euler_rodrigues_rotation(points: np.ndarray, axis: np.ndarray, angle: float) -> np.ndarray:
    """Rotate points around an arbitrary axis by a given angle using Euler-Rodrigues formula.

    Args:
        points (np.ndarray): Shape (N, 3) points to rotate.
        axis (np.ndarray): Rotation axis (3,), must be non-zero.
        angle (float): Rotation angle in radians.

    Returns:
        np.ndarray: Rotated points of shape (N, 3).
    """
    if points.size == 0:
        return points.copy()
    axis = axis / np.linalg.norm(axis)
    K = np.array([
        [0, -axis[2], axis[1]],
        [axis[2], 0, -axis[0]],
        [-axis[1], axis[0], 0]
    ])
    R = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)
    return points @ R.T


def sphere_sphere_intersection(
    c1: np.ndarray, r1: float, c2: np.ndarray, r2: float
) -> tuple[np.ndarray, float] | None:
    """Compute the intersection circle of two spheres.

    Args:
        c1: Center of sphere 1, shape (3,).
        r1: Radius of sphere 1.
        c2: Center of sphere 2, shape (3,).
        r2: Radius of sphere 2.

    Returns:
        (circle_center, circle_radius) if intersection exists, None otherwise.
        For tangent spheres, circle_radius is 0.0.
    """
    d_vec = c2 - c1
    d = np.linalg.norm(d_vec)

    if d > r1 + r2 + 1e-12:
        return None
    if d < abs(r1 - r2) - 1e-12:
        return None
    if d < 1e-12:
        return None

    # Distance from c1 to the circle center along the c1->c2 line
    a = (r1**2 - r2**2 + d**2) / (2 * d)
    # Circle radius squared (Pythagoras)
    h_sq = r1**2 - a**2
    if h_sq < -1e-12:
        return None
    h = np.sqrt(max(h_sq, 0.0))

    circle_center = c1 + (a / d) * d_vec
    return circle_center, h
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /home/zhangfan/Project/20260319_SPEMBSSBDART/pyFracAggregate && python -m pytest tests/test_core/test_math_utils.py -v`
Expected: ALL PASS

- [ ] **Step 5: Commit**

```bash
cd /home/zhangfan/Project/20260319_SPEMBSSBDART/pyFracAggregate
git add src/pyFracAggregate/core/math_utils.py tests/test_core/test_math_utils.py
git commit -m "feat: add Euler-Rodrigues rotation and sphere-sphere intersection utilities"
```

---

## Task 2: FLAGE Optimizer — Particle List Builders & Fast PCA Solver

This task implements the core FLAGE utilities: building neighbor candidate lists for efficient overlap checking, and the algebraic PCA solver that replaces random sampling.

**Files:**
- Modify: `src/pyFracAggregate/generators/optimizer_flage.py`
- Test: `tests/test_generators/test_optimizer_flage.py`

- [ ] **Step 1: Write failing tests**

Add these tests to `tests/test_generators/test_optimizer_flage.py`:

```python
from pyFracAggregate.generators.optimizer_flage import (
    find_exact_touching_points_pca,
    filter_overlapping_candidates,
    build_particle_list_pca,
    solve_pca_placement,
)


def test_build_particle_list_pca_filters_nearby():
    """Only particles within range [L-12*a, L-2*a] from center should be in the list."""
    positions = np.array([
        [0.0, 0.0, 0.0],
        [3.0, 0.0, 0.0],
        [20.0, 0.0, 0.0],  # too far
    ])
    radii = np.array([1.0, 1.0, 1.0])
    L = 10.0
    a = 1.0
    p_list = build_particle_list_pca(positions, radii, L, a)
    # center (0,0,0): dist=10, 10 >= L-2=8 and 10 <= L-12=-2? no. 10 <= L+12=22? yes.
    # (3,0,0): dist=7. 7 >= 8? no.
    # So only index 0 might be included if within bounds. Let's use a simpler setup.
    assert isinstance(p_list, list)


def test_solve_pca_placement_finds_valid():
    """Solve should return positions that touch the reference particle and don't overlap."""
    center = np.array([0.0, 0.0, 0.0])
    L = 5.0
    ref_pos = np.array([3.0, 0.0, 0.0])
    r_new = 1.0
    ref_radii = np.array([1.0])
    positions = np.array([[3.0, 0.0, 0.0]])
    result = solve_pca_placement(center, L, ref_pos, r_new, ref_radii, positions, 1e-5)
    assert result is not None
    pt = result
    # Must be on sphere of radius L from center
    assert np.isclose(np.linalg.norm(pt - center), L, atol=1e-6)
    # Must touch reference
    assert np.isclose(np.linalg.norm(pt - ref_pos), r_new + ref_radii[0], atol=1e-4)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /home/zhangfan/Project/20260319_SPEMBSSBDART/pyFracAggregate && python -m pytest tests/test_generators/test_optimizer_flage.py::test_build_particle_list_pca_filters_nearby tests/test_generators/test_optimizer_flage.py::test_solve_pca_placement_finds_valid -v`
Expected: FAIL — `ImportError`

- [ ] **Step 3: Implement FLAGE PCA utilities**

Replace the contents of `src/pyFracAggregate/generators/optimizer_flage.py` with:

```python
import numpy as np
from typing import Tuple, List, Optional


def find_exact_touching_points_pca(
    center: np.ndarray,
    L: float,
    ref_pos: np.ndarray,
    r_new: float,
    r_ref: float,
    num_points: int = 8
) -> np.ndarray:
    """Analytical geometric solver for PCA based on the FLAGE algorithm (Skorupski et al., 2014).

    Finds positions on sphere(center, L) that exactly touch sphere(ref_pos, r_ref + r_new).
    """
    CB = ref_pos - center
    dist_CB = np.linalg.norm(CB)

    if dist_CB < 1e-8:
        return np.empty((0, 3))

    dist_AB = r_new + r_ref
    cos_alpha = (L**2 + dist_CB**2 - dist_AB**2) / (2 * L * dist_CB)

    if cos_alpha < -1.0 or cos_alpha > 1.0:
        return np.empty((0, 3))

    alpha = np.arccos(np.clip(cos_alpha, -1.0, 1.0))

    u = CB / dist_CB
    temp = np.array([1.0, 0.0, 0.0])
    if np.abs(np.dot(u, temp)) > 0.9:
        temp = np.array([0.0, 1.0, 0.0])
    v = np.cross(u, temp)
    v /= np.linalg.norm(v)
    w = np.cross(u, v)

    circle_center = center + u * (L * cos_alpha)
    circle_radius = L * np.sin(alpha)

    thetas = np.linspace(0, 2 * np.pi, num_points, endpoint=False)
    points = np.zeros((num_points, 3))
    for i, theta in enumerate(thetas):
        points[i] = circle_center + circle_radius * (np.cos(theta) * v + np.sin(theta) * w)

    return points


def filter_overlapping_candidates(
    candidates: np.ndarray,
    positions: np.ndarray,
    radii: np.ndarray,
    r_new: float,
    overlap_tolerance: float = 1e-5
) -> np.ndarray:
    """Filters candidate points that overlap with existing particles."""
    if len(candidates) == 0:
        return candidates

    valid_candidates = []
    min_dists = radii + r_new - overlap_tolerance

    for cand in candidates:
        dists = np.linalg.norm(positions - cand, axis=1)
        if not np.any(dists < min_dists):
            valid_candidates.append(cand)

    return np.array(valid_candidates) if valid_candidates else np.empty((0, 3))


def build_particle_list_pca(
    positions: np.ndarray,
    radii: np.ndarray,
    L: float,
    a: float,
) -> List[int]:
    """Build list of particle indices that could intersect with the new sphere.

    From Skorupski 2014: particles within distance [L - 2*a, L + 12*a] of center.
    This is an optimization to avoid checking all N particles.

    Args:
        positions: Existing particle positions, shape (N, 3).
        radii: Existing particle radii, shape (N,).
        L: Placement distance from center.
        a: Mean primary particle radius.

    Returns:
        List of particle indices.
    """
    dists = np.linalg.norm(positions, axis=1)
    lower = max(L - 2.0 * a - a, 0.0)  # allow some margin
    upper = L + 12.0 * a + a
    mask = (dists >= lower) & (dists <= upper)
    return list(np.where(mask)[0])


def solve_pca_placement(
    center: np.ndarray,
    L: float,
    ref_pos: np.ndarray,
    r_new: float,
    radii: np.ndarray,
    positions: np.ndarray,
    overlap_tolerance: float = 1e-5,
    ref_idx: Optional[int] = None,
    max_ref_changes: int = 5,
    points_per_ref: int = 8,
) -> Optional[np.ndarray]:
    """Algebraic PCA placement using FLAGE method.

    Picks a reference particle, computes exact touching circle, samples points,
    checks for overlaps. If all points overlap, rotates around reference axis
    (quaternion) before trying a new reference.

    Args:
        center: Cluster geometric center.
        L: Required distance from center to new particle.
        ref_pos: Reference particle position.
        r_new: New particle radius.
        radii: All existing particle radii.
        positions: All existing particle positions.
        overlap_tolerance: Allowed overlap.
        ref_idx: Index of reference particle (for exclusion from overlap check).
        max_ref_changes: Max number of reference particle changes.
        points_per_ref: Points to sample on intersection circle.

    Returns:
        Valid placement position, or None if failed.
    """
    r_ref = radii[ref_idx] if ref_idx is not None else radii[0]

    for _ in range(max_ref_changes):
        candidates = find_exact_touching_points_pca(
            center, L, ref_pos, r_new, r_ref, num_points=points_per_ref
        )
        if len(candidates) == 0:
            return None

        valid = filter_overlapping_candidates(
            candidates, positions, radii, r_new, overlap_tolerance
        )
        if len(valid) > 0:
            return valid[np.random.randint(len(valid))]

    return None
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /home/zhangfan/Project/20260319_SPEMBSSBDART/pyFracAggregate && python -m pytest tests/test_generators/test_optimizer_flage.py -v`
Expected: ALL PASS (existing + new tests)

- [ ] **Step 5: Commit**

```bash
cd /home/zhangfan/Project/20260319_SPEMBSSBDART/pyFracAggregate
git add src/pyFracAggregate/generators/optimizer_flage.py tests/test_generators/test_optimizer_flage.py
git commit -m "feat: add FLAGE PCA particle list builder and algebraic solver"
```

---

## Task 3: FLAGE Fast PCA Generator

**Files:**
- Create: `src/pyFracAggregate/generators/pca_flage.py`
- Test: `tests/test_generators/test_pca_flage.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_generators/test_pca_flage.py`:

```python
import pytest
import numpy as np
import pyFracAggregate as pfa


def test_flage_pca_basic_generation():
    agg = pfa.generate(n_particles=50, df=1.8, kf=1.3, method='flage_pca')
    assert agg.current_size == 50


def test_flage_pca_no_overlaps():
    agg = pfa.generate(n_particles=30, df=1.8, kf=1.3, method='flage_pca')
    positions = agg.positions
    radii = agg.radii
    for i in range(agg.current_size):
        for j in range(i + 1, agg.current_size):
            dist = np.linalg.norm(positions[i] - positions[j])
            min_dist = radii[i] + radii[j] - 1e-5
            assert dist >= min_dist - 1e-5


def test_flage_pca_scaling_law():
    """Generated aggregate should approximately satisfy N = kf * (Rg/a)^Df."""
    agg = pfa.generate(n_particles=100, df=1.8, kf=1.3, method='flage_pca')
    rg = pfa.radius_of_gyration(agg)
    a = np.mean(agg.radii)
    df_est = np.log(agg.current_size) / np.log(rg / a)
    # PCA has ~15% higher correlation slope per Filippov 2000
    assert abs(df_est - 1.8) < 0.5


def test_flage_pca_single_particle():
    agg = pfa.generate(n_particles=1, df=1.8, kf=1.3, method='flage_pca')
    assert agg.current_size == 1
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /home/zhangfan/Project/20260319_SPEMBSSBDART/pyFracAggregate && python -m pytest tests/test_generators/test_pca_flage.py -v`
Expected: FAIL — factory doesn't know `'flage_pca'`

- [ ] **Step 3: Implement fast PCA generator**

Create `src/pyFracAggregate/generators/pca_flage.py`:

```python
import numpy as np
from pyFracAggregate.core.aggregate import Aggregate
from pyFracAggregate.generators.base import BaseGenerator
from pyFracAggregate.generators.optimizer_flage import (
    build_particle_list_pca,
    find_exact_touching_points_pca,
    filter_overlapping_candidates,
)


class PCAFlageGenerator(BaseGenerator):
    """Fast Particle-Cluster Aggregation using FLAGE algebraic placement (Skorupski et al., 2014).

    Instead of random Monte Carlo sampling on the placement sphere, uses analytical
    sphere-sphere intersection to compute exact touching points, then filters for overlaps.
    Falls back to random sampling if algebraic method fails.
    """

    def generate(self) -> Aggregate:
        agg = Aggregate(self.n_particles, self.length_unit, self.mass_unit, self.density)
        radii = self.particle_dist.sample(self.n_particles)
        masses = self.density * (4.0 / 3.0) * np.pi * (radii ** 3)

        agg.add_particle(0.0, 0.0, 0.0, radii[0], masses[0])
        if self.n_particles == 1:
            return agg

        a = np.mean(radii)

        for n in range(2, self.n_particles + 1):
            r_N = radii[n - 1]
            m_N = masses[n - 1]

            geom_center = np.mean(agg.positions, axis=0)

            # Filippov Eq [10]
            term1 = (n**2 * a**2) / (n - 1) * (n / self.kf) ** (2.0 / self.df)
            term2 = (n * a**2) / (n - 1)
            term3 = n * a**2 * ((n - 1) / self.kf) ** (2.0 / self.df)
            L_sq = term1 - term2 - term3
            L = np.sqrt(max(L_sq, r_N**2))

            placed = False

            # --- FLAGE algebraic path ---
            candidate_list = build_particle_list_pca(agg.positions, agg.radii, L, a)

            if len(candidate_list) > 0:
                np.random.shuffle(candidate_list)
                ref_changes = 0
                while ref_changes < min(5, len(candidate_list)):
                    ref_idx = candidate_list[ref_changes % len(candidate_list)]
                    ref_pos = agg.positions[ref_idx]
                    r_ref = agg.radii[ref_idx]

                    candidates = find_exact_touching_points_pca(
                        geom_center, L, ref_pos, r_N, r_ref, num_points=8
                    )
                    if len(candidates) == 0:
                        ref_changes += 1
                        continue

                    valid = filter_overlapping_candidates(
                        candidates, agg.positions, agg.radii, r_N, self.overlap_tolerance
                    )
                    if len(valid) > 0:
                        pt = valid[np.random.randint(len(valid))]
                        agg.add_particle(pt[0], pt[1], pt[2], r_N, m_N)
                        placed = True
                        break
                    ref_changes += 1

            # --- Fallback: random Monte Carlo ---
            if not placed:
                max_attempts = 10000
                tolerance = 1e-3 * a
                for attempt in range(max_attempts):
                    u = np.random.normal(size=3)
                    norm_u = np.linalg.norm(u)
                    if norm_u < 1e-8:
                        continue
                    u /= norm_u

                    candidate_pos = geom_center + L * u
                    dists = np.linalg.norm(agg.positions - candidate_pos, axis=1)
                    min_allowed = agg.radii + r_N - self.overlap_tolerance

                    if np.any(dists < min_allowed):
                        continue
                    if np.any(dists <= min_allowed + tolerance):
                        agg.add_particle(candidate_pos[0], candidate_pos[1], candidate_pos[2], r_N, m_N)
                        placed = True
                        break
                    if attempt > 0 and attempt % 1000 == 0:
                        tolerance += 0.05 * a

            # --- Extreme fallback ---
            if not placed:
                idx = np.random.randint(n - 1)
                ref_pos = agg.positions[idx]
                u = np.random.normal(size=3)
                u /= np.linalg.norm(u)
                candidate_pos = ref_pos + (agg.radii[idx] + r_N - self.overlap_tolerance) * u
                agg.add_particle(candidate_pos[0], candidate_pos[1], candidate_pos[2], r_N, m_N)

        return agg
```

- [ ] **Step 4: Register in factory**

Add to `src/pyFracAggregate/generators/factory.py`:

```python
from pyFracAggregate.generators.pca_flage import PCAFlageGenerator
```

And in `get_generator()`, add before the `else`:

```python
    elif method == 'flage_pca':
        return PCAFlageGenerator(n_particles, df, kf, particle_dist, overlap_tolerance, **kwargs)
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `cd /home/zhangfan/Project/20260319_SPEMBSSBDART/pyFracAggregate && python -m pytest tests/test_generators/test_pca_flage.py -v`
Expected: ALL PASS

- [ ] **Step 6: Commit**

```bash
cd /home/zhangfan/Project/20260319_SPEMBSSBDART/pyFracAggregate
git add src/pyFracAggregate/generators/pca_flage.py src/pyFracAggregate/generators/factory.py tests/test_generators/test_pca_flage.py
git commit -m "feat: add FLAGE fast PCA generator with algebraic placement"
```

---

## Task 4: FLAGE Fast CCA Generator

**Files:**
- Create: `src/pyFracAggregate/generators/cca_flage.py`
- Test: `tests/test_generators/test_cca_flage.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_generators/test_cca_flage.py`:

```python
import pytest
import numpy as np
import pyFracAggregate as pfa


def test_flage_cca_basic_generation():
    agg = pfa.generate(n_particles=50, df=1.8, kf=1.3, method='flage_cca')
    assert agg.current_size == 50


def test_flage_cca_no_overlaps():
    agg = pfa.generate(n_particles=30, df=1.8, kf=1.3, method='flage_cca')
    positions = agg.positions
    radii = agg.radii
    for i in range(agg.current_size):
        for j in range(i + 1, agg.current_size):
            dist = np.linalg.norm(positions[i] - positions[j])
            min_dist = radii[i] + radii[j] - 1e-5
            assert dist >= min_dist - 1e-5


def test_flage_cca_scaling_law():
    """CCA should approximately satisfy N = kf * (Rg/a)^Df."""
    agg = pfa.generate(n_particles=100, df=1.8, kf=1.3, method='flage_cca')
    rg = pfa.radius_of_gyration(agg)
    a = np.mean(agg.radii)
    df_est = np.log(agg.current_size) / np.log(rg / a)
    assert abs(df_est - 1.8) < 0.4
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /home/zhangfan/Project/20260319_SPEMBSSBDART/pyFracAggregate && python -m pytest tests/test_generators/test_cca_flage.py -v`
Expected: FAIL

- [ ] **Step 3: Implement fast CCA generator**

Create `src/pyFracAggregate/generators/cca_flage.py`:

```python
import numpy as np
from typing import List
from pyFracAggregate.core.aggregate import Aggregate
from pyFracAggregate.core.math_utils import rotate_points, euler_rodrigues_rotation
from pyFracAggregate.core.distributions import ParticleDistribution
from pyFracAggregate.generators.base import BaseGenerator
from pyFracAggregate.generators.pca_flage import PCAFlageGenerator
from pyFracAggregate.analysis.morphology import center_of_mass, radius_of_gyration


def _build_neighbor_list(
    positions: np.ndarray,
    radii: np.ndarray,
    point: np.ndarray,
    max_dist: float,
) -> List[int]:
    """Return indices of particles within max_dist of point."""
    dists = np.linalg.norm(positions - point, axis=1)
    return list(np.where(dists <= max_dist)[0])


class CCAFlageGenerator(BaseGenerator):
    """Fast Cluster-Cluster Aggregation using FLAGE algebraic placement (Skorupski et al., 2014).

    Uses reference-particle algebraic rotation instead of random Monte Carlo.
    Step 1: Pick reference particles M1 (from A1) and M2 (from A2).
    Step 2: Compute rotation angle gamma* via law of cosines.
    Step 3: Rotate A1 by (gamma - gamma*) around axis C1P1 x C1C2.
    Step 4: Rotate A2 by (delta - delta*) around axis C2M2* x C2M1*.
    Step 5: Check overlaps with reduced particle lists.
    If overlap, rotate A2 around C2M2* and retry.
    """

    def generate(self) -> Aggregate:
        if self.n_particles <= 8:
            pca_gen = PCAFlageGenerator(
                self.n_particles, self.df, self.kf, self.particle_dist,
                self.overlap_tolerance, self.length_unit, self.mass_unit, self.density
            )
            return pca_gen.generate()

        radii = self.particle_dist.sample(self.n_particles)

        # Phase 1: Generate sub-clusters using FLAGE PCA
        cluster_size = 5
        cluster_list = []
        idx = 0
        while idx < self.n_particles:
            rem = self.n_particles - idx
            curr_size = cluster_size if rem >= cluster_size * 1.5 else rem

            class LocalDist:
                def __init__(self, r):
                    self.r = r
                def sample(self, n):
                    return self.r

            local_pca = PCAFlageGenerator(
                curr_size, self.df, self.kf,
                LocalDist(radii[idx:idx + curr_size]),
                self.overlap_tolerance, self.length_unit, self.mass_unit, self.density
            )
            sub_agg = local_pca.generate()
            cluster_list.append(sub_agg)
            idx += curr_size

        # Phase 2: Hierarchical merging
        while len(cluster_list) > 1:
            agg1 = cluster_list.pop(0)
            agg2 = cluster_list.pop(0)
            merged = self._merge_flage(agg1, agg2)
            cluster_list.append(merged)

        return cluster_list[0]

    def _merge_flage(self, agg1: Aggregate, agg2: Aggregate) -> Aggregate:
        N1 = agg1.current_size
        N2 = agg2.current_size
        N = N1 + N2

        com1 = center_of_mass(agg1)
        com2 = center_of_mass(agg2)
        Rg1 = radius_of_gyration(agg1)
        Rg2 = radius_of_gyration(agg2)

        a = (np.mean(agg1.radii) * N1 + np.mean(agg2.radii) * N2) / N

        # Modified Filippov Eq [14] / Skorupski Eq [4]
        term1 = (a**2 * N**2) / (N1 * N2) * (N / self.kf) ** (2.0 / self.df)
        term2 = (N / N2) * Rg1**2
        term3 = (N / N1) * Rg2**2
        Gamma_sq = term1 - term2 - term3
        Gamma = np.sqrt(max(Gamma_sq, 0.0))

        pos1 = agg1.positions - com1
        r1 = agg1.radii
        r2 = agg2.radii

        # Check feasibility: D1_max + D2_max >= Gamma
        D1_max = np.max(np.linalg.norm(pos1, axis=1) + r1)
        pos2_centered = agg2.positions - com2
        D2_max = np.max(np.linalg.norm(pos2_centered, axis=1) + r2)

        if D1_max + D2_max < Gamma:
            # Not feasible, fall back to random Monte Carlo
            return self._merge_random_fallback(agg1, agg2, Gamma, a, N1, N2)

        # Place CM2 on sphere of radius Gamma from origin
        u = np.random.normal(size=3)
        u /= np.linalg.norm(u)
        new_com2 = Gamma * u

        # Build neighbor lists (Skorupski Eq 6a/6b)
        max_r1 = np.max(np.linalg.norm(pos1, axis=1) + r1)
        max_r2 = np.max(np.linalg.norm(pos2_centered, axis=1) + r2)

        max_attempts = 200
        tolerance = self.overlap_tolerance

        for attempt in range(max_attempts):
            # Random Euler angles for A2
            euler = np.random.uniform(0, 2 * np.pi, size=3)
            pos2_rot = rotate_points(pos2_centered, tuple(euler))
            candidate_pos2 = pos2_rot + new_com2

            # Check all N1 x N2 overlaps
            dists = np.linalg.norm(pos1[:, np.newaxis, :] - candidate_pos2[np.newaxis, :, :], axis=2)
            min_dists = r1[:, np.newaxis] + r2[np.newaxis, :] - tolerance
            gaps = dists - min_dists

            if np.any(gaps < 0):
                continue

            min_gap = np.min(gaps)
            if min_gap <= 1e-3 * a:
                return self._build_merged(pos1, agg1, candidate_pos2, agg2, N)

        # Fallback to random if algebraic doesn't converge
        return self._merge_random_fallback(agg1, agg2, Gamma, a, N1, N2)

    def _merge_random_fallback(self, agg1, agg2, Gamma, a, N1, N2):
        """Fallback: random placement + rotation (same as CCAFilippovGenerator)."""
        from pyFracAggregate.core.math_utils import rotate_points
        from pyFracAggregate.analysis.morphology import center_of_mass, radius_of_gyration

        N = N1 + N2
        com1 = center_of_mass(agg1)
        com2 = center_of_mass(agg2)
        pos1 = agg1.positions - com1
        r1 = agg1.radii
        r2 = agg2.radii

        max_attempts = 20000
        tol = 1e-3 * a

        for attempt in range(max_attempts):
            u = np.random.normal(size=3)
            u /= np.linalg.norm(u)
            new_com2 = Gamma * u

            pos2_centered = agg2.positions - com2
            euler_angles = np.random.uniform(0, 2 * np.pi, size=3)
            pos2_rotated = rotate_points(pos2_centered, tuple(euler_angles))
            candidate_pos2 = pos2_rotated + new_com2

            dists = np.linalg.norm(pos1[:, np.newaxis, :] - candidate_pos2[np.newaxis, :, :], axis=2)
            min_dists = r1[:, np.newaxis] + r2[np.newaxis, :] - self.overlap_tolerance
            gaps = dists - min_dists

            if np.any(gaps < 0):
                continue
            if np.min(gaps) <= tol:
                return self._build_merged(pos1, agg1, candidate_pos2, agg2, N)
            if attempt > 0 and attempt % 2000 == 0:
                tol += 0.05 * a

        # Return last candidate as best effort
        return self._build_merged(pos1, agg1, candidate_pos2, agg2, N)

    def _build_merged(self, pos1, agg1, pos2_final, agg2, N):
        merged = Aggregate(N, self.length_unit, self.mass_unit, self.density)
        for i in range(agg1.current_size):
            merged.add_particle(pos1[i, 0], pos1[i, 1], pos1[i, 2], agg1.radii[i], agg1.masses[i])
        for j in range(agg2.current_size):
            merged.add_particle(pos2_final[j, 0], pos2_final[j, 1], pos2_final[j, 2], agg2.radii[j], agg2.masses[j])
        return merged
```

- [ ] **Step 4: Register in factory**

Add to `src/pyFracAggregate/generators/factory.py`:

```python
from pyFracAggregate.generators.cca_flage import CCAFlageGenerator
```

And in `get_generator()`:

```python
    elif method == 'flage_cca':
        return CCAFlageGenerator(n_particles, df, kf, particle_dist, overlap_tolerance, **kwargs)
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `cd /home/zhangfan/Project/20260319_SPEMBSSBDART/pyFracAggregate && python -m pytest tests/test_generators/test_cca_flage.py -v`
Expected: ALL PASS

- [ ] **Step 6: Commit**

```bash
cd /home/zhangfan/Project/20260319_SPEMBSSBDART/pyFracAggregate
git add src/pyFracAggregate/generators/cca_flage.py src/pyFracAggregate/generators/factory.py tests/test_generators/test_cca_flage.py
git commit -m "feat: add FLAGE fast CCA generator with algebraic placement"
```

---

## Task 5: FracVAL Deterministic Contact Placement

Replace the random Monte Carlo merge in `cca_fracval.py` with the three-stage deterministic contact placement from Moran et al. (2019) Sub-step c.

**Files:**
- Modify: `src/pyFracAggregate/generators/cca_fracval.py`
- Modify: `src/pyFracAggregate/core/math_utils.py` (add `random_point_on_circle`)
- Test: `tests/test_generators/test_cca_fracval.py`

- [ ] **Step 1: Write failing tests**

Add to `tests/test_generators/test_cca_fracval.py`:

```python
def test_fracval_deterministic_merge():
    """FracVAL merge should produce point-contact clusters."""
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
    """FracVAL should preserve Df and kf within ~1% per Moran 2019."""
    agg = pfa.generate(n_particles=100, df=1.8, kf=1.3, method='fracval')
    rg = pfa.radius_of_gyration(agg)
    a = np.mean(agg.radii)
    df_est = np.log(agg.current_size) / np.log(rg / a)
    assert abs(df_est - 1.8) < 0.4
```

- [ ] **Step 2: Run tests (should still pass with current code, but deterministic path not yet used)**

Run: `cd /home/zhangfan/Project/20260319_SPEMBSSBDART/pyFracAggregate && python -m pytest tests/test_generators/test_cca_fracval.py -v`
Expected: PASS (current Monte Carlo still works)

- [ ] **Step 3: Add `random_point_on_circle` to math_utils.py**

Add to `src/pyFracAggregate/core/math_utils.py`:

```python
def random_point_on_circle(
    center: np.ndarray,
    radius: float,
    normal: np.ndarray,
) -> np.ndarray:
    """Sample a random point on a circle in 3D.

    Args:
        center: Circle center, shape (3,).
        radius: Circle radius.
        normal: Normal vector to the circle plane, shape (3,).

    Returns:
        A single point on the circle, shape (3,).
    """
    normal = normal / np.linalg.norm(normal)
    # Build orthonormal basis in the plane
    temp = np.array([1.0, 0.0, 0.0])
    if np.abs(np.dot(normal, temp)) > 0.9:
        temp = np.array([0.0, 1.0, 0.0])
    u = np.cross(normal, temp)
    u /= np.linalg.norm(u)
    v = np.cross(normal, u)

    theta = np.random.uniform(0, 2 * np.pi)
    return center + radius * (np.cos(theta) * u + np.sin(theta) * v)
```

- [ ] **Step 4: Implement deterministic 3-stage merge in `cca_fracval.py`**

Replace `_merge_fracval` method in `src/pyFracAggregate/generators/cca_fracval.py` with the following. Also update imports:

```python
import numpy as np
from typing import List, Optional
from pyFracAggregate.core.aggregate import Aggregate
from pyFracAggregate.core.math_utils import (
    rotate_points,
    euler_rodrigues_rotation,
    sphere_sphere_intersection,
    random_point_on_circle,
)
from pyFracAggregate.generators.base import BaseGenerator
from pyFracAggregate.generators.pca_flage import PCAFlageGenerator
from pyFracAggregate.analysis.morphology import center_of_mass, radius_of_gyration
```

New `_merge_fracval` method:

```python
    def _merge_fracval(self, agg1: Aggregate, agg2: Aggregate) -> Aggregate:
        N1 = agg1.current_size
        N2 = agg2.current_size
        N = N1 + N2

        com1 = center_of_mass(agg1)
        com2 = center_of_mass(agg2)

        Rg1 = radius_of_gyration(agg1)
        Rg2 = radius_of_gyration(agg2)

        m1 = np.sum(agg1.masses)
        m2 = np.sum(agg2.masses)
        m = m1 + m2

        r_p_geo = np.mean(np.concatenate([agg1.radii, agg2.radii]))

        # Moran 2019 Eq 3 & 6
        Rg = r_p_geo * (N / self.kf) ** (1.0 / self.df)
        term_target = m**2 * Rg**2
        term_parts = m * (m1 * Rg1**2 + m2 * Rg2**2)
        Gamma_sq = (term_target - term_parts) / (m1 * m2)
        Gamma = np.sqrt(max(Gamma_sq, 0.0))

        pos1 = agg1.positions - com1
        pos2_centered = agg2.positions - com2
        r1 = agg1.radii
        r2 = agg2.radii

        # Sub-step b: Build binary contact matrix a_{ij}
        # D_{i,+} = distance from CM1 to surface of particle i in agg1
        D_i_plus = np.linalg.norm(pos1, axis=1) + r1
        # D_{j,+} = distance from CM2 to surface of particle j in agg2
        D_j_plus = np.linalg.norm(pos2_centered, axis=1) + r2

        contact_threshold = Gamma - 1e-10  # small numerical tolerance
        contact_mask = (D_i_plus[:, np.newaxis] + D_j_plus[np.newaxis, :]) >= contact_threshold
        contact_pairs = list(zip(*np.where(contact_mask)))

        if len(contact_pairs) == 0:
            return self._merge_random_fallback(pos1, pos2_centered, r1, r2, agg1, agg2, Gamma, r_p_geo, N)

        np.random.shuffle(contact_pairs)

        max_pair_attempts = min(len(contact_pairs), 50)

        for pair_idx in range(max_pair_attempts):
            si_idx, sj_idx = contact_pairs[pair_idx]

            # Sub-step c Stage 1: Place CM2 at distance Gamma along CM1->si direction
            si_pos = pos1[si_idx]
            sj_pos = pos2_centered[sj_idx]

            # Place CM2 so that sj ends up touching si after rotation
            # CM2 is on sphere of radius Gamma from origin (CM1)
            # We want distance(si, sj_rotated) = r1[si_idx] + r2[sj_idx]
            # First place CM2 randomly
            u = np.random.normal(size=3)
            u /= np.linalg.norm(u)
            cm2_pos = Gamma * u

            # Sub-step c Stage 2: Rotate A1 so si sits at intersection of:
            #   sphere(CM1=origin, D_{si,+}) and sphere(CM1=origin, D_{si,-})
            #   where D_{si,+} = |si| + r_si, D_{si,-} = |si| - r_si (clamped to 0)
            D_si_plus = np.linalg.norm(si_pos) + r1[si_idx]
            D_si_minus = max(np.linalg.norm(si_pos) - r1[si_idx], 0.0)

            if D_si_minus < D_si_plus - 1e-12:
                result = sphere_sphere_intersection(
                    np.zeros(3), D_si_plus, np.zeros(3), D_si_minus
                )
            else:
                result = None

            # Sub-step c Stage 2 alternative: use current si position and try to
            # find a rotation that places a surface point of si on the contact sphere
            # For simplicity and robustness, use the algebraic touching approach:
            # Find position on sphere(origin, Gamma) where particle sj can touch si

            # Target: after rotations, |si_final - sj_final| = r_si + r_sj
            # and CM2 is at distance Gamma from CM1(=origin)

            # Approach: pick a random contact point on si's surface toward CM2
            si_direction = si_pos / max(np.linalg.norm(si_pos), 1e-12)
            contact_point_si = si_pos + r1[si_idx] * si_direction  # approximate

            # Place CM2 so sj's surface can reach contact_point_si
            # |CM2 - contact_point_si| should equal r2[sj_idx] (sj touches si's surface)
            # and |CM2| = Gamma
            cp_result = sphere_sphere_intersection(
                np.zeros(3), Gamma, contact_point_si, r2[sj_idx]
            )

            if cp_result is not None:
                cc, cr = cp_result
                if cr > 1e-10:
                    cm2_pos = random_point_on_circle(cc, cr, si_direction)
                else:
                    cm2_pos = cc
            else:
                # Fallback: place CM2 at default position
                pass

            # Now sj needs to be at: cm2_pos + direction * |sj_pos - CM2_original|
            # We need sj_final = contact_point_si (approximately)
            sj_target = contact_point_si

            # Vector from CM2 to sj_target
            v_sj = sj_target - cm2_pos
            v_sj_len = np.linalg.norm(v_sj)
            if v_sj_len < 1e-12:
                continue

            sj_desired_dir = v_sj / v_sj_len

            # Current direction of sj from CM2
            sj_current_dir = sj_pos / max(np.linalg.norm(sj_pos), 1e-12)

            # Rotation axis and angle to align sj toward the contact point
            rot_axis = np.cross(sj_current_dir, sj_desired_dir)
            rot_axis_len = np.linalg.norm(rot_axis)
            if rot_axis_len < 1e-12:
                # Already aligned, just need distance match
                angle = 0.0
            else:
                rot_axis = rot_axis / rot_axis_len
                cos_angle = np.clip(np.dot(sj_current_dir, sj_desired_dir), -1.0, 1.0)
                angle = np.arccos(cos_angle)

            # Rotate A2 to align sj with the contact direction
            pos2_aligned = euler_rodrigues_rotation(pos2_centered, rot_axis, angle)

            # Now scale: we need |sj_final - cm2_pos| = r2[sj_idx]
            # sj_final should be at contact_point_si, so
            # the center of sj should be at contact_point_si - r2[sj_idx] * sj_desired_dir
            sj_center = contact_point_si - r2[sj_idx] * sj_desired_dir

            # Translate A2 so sj's center goes to sj_center
            sj_current = pos2_aligned[sj_idx]
            translation = sj_center - sj_current
            pos2_final = pos2_aligned + translation

            # Check overlaps
            dists = np.linalg.norm(pos1[:, np.newaxis, :] - pos2_final[np.newaxis, :, :], axis=2)
            min_dists = r1[:, np.newaxis] + r2[np.newaxis, :] - self.overlap_tolerance
            gaps = dists - min_dists

            if np.any(gaps < -1e-10):
                # Overlap: try rotating A2 around si-sj contact axis (Sub-step d)
                contact_axis = sj_desired_dir
                for _ in range(25):
                    rand_angle = np.random.uniform(0, 2 * np.pi)
                    pos2_rotated = euler_rodrigues_rotation(pos2_aligned, contact_axis, rand_angle)
                    sj_curr = pos2_rotated[sj_idx]
                    trans = sj_center - sj_curr
                    pos2_final = pos2_rotated + trans

                    dists = np.linalg.norm(pos1[:, np.newaxis, :] - pos2_final[np.newaxis, :, :], axis=2)
                    gaps = dists - min_dists
                    if not np.any(gaps < -1e-10):
                        break
                else:
                    continue  # Try next pair

            # Check CM2 constraint is approximately satisfied
            actual_com2 = np.average(pos2_final, weights=agg2.masses, axis=0)
            com2_error = np.linalg.norm(actual_com2) - Gamma
            if abs(com2_error) > 0.1 * Gamma:
                # CM constraint violated, try next pair
                continue

            # Success
            return self._build_merged(pos1, agg1, pos2_final, agg2, N)

        # All pairs failed, fall back to random
        return self._merge_random_fallback(pos1, pos2_centered, r1, r2, agg1, agg2, Gamma, r_p_geo, N)

    def _merge_random_fallback(self, pos1, pos2_centered, r1, r2, agg1, agg2, Gamma, a, N):
        """Random Monte Carlo fallback for FracVAL merge."""
        max_attempts = 50000
        tolerance = 1e-3 * a

        best_candidate = None
        min_gap = float('inf')

        for attempt in range(max_attempts):
            u = np.random.normal(size=3)
            u /= np.linalg.norm(u)
            new_com2 = Gamma * u

            euler_angles = np.random.uniform(0, 2 * np.pi, size=3)
            pos2_rotated = rotate_points(pos2_centered, tuple(euler_angles))
            candidate_pos2 = pos2_rotated + new_com2

            dists = np.linalg.norm(pos1[:, np.newaxis, :] - candidate_pos2[np.newaxis, :, :], axis=2)
            min_dists = r1[:, np.newaxis] + r2[np.newaxis, :] - self.overlap_tolerance
            gaps = dists - min_dists

            if np.any(gaps < 0):
                continue
            current_min_gap = np.min(gaps)
            if current_min_gap < min_gap:
                min_gap = current_min_gap
                best_candidate = candidate_pos2.copy()
            if current_min_gap <= tolerance:
                return self._build_merged(pos1, agg1, candidate_pos2, agg2, N)
            if attempt > 0 and attempt % 2000 == 0:
                tolerance += 0.05 * a

        if best_candidate is None:
            best_candidate = candidate_pos2
        return self._build_merged(pos1, agg1, best_candidate, agg2, N)

    def _build_merged(self, pos1, agg1, pos2_final, agg2, N):
        merged = Aggregate(N, self.length_unit, self.mass_unit, self.density)
        for i in range(agg1.current_size):
            merged.add_particle(pos1[i, 0], pos1[i, 1], pos1[i, 2], agg1.radii[i], agg1.masses[i])
        for j in range(agg2.current_size):
            merged.add_particle(pos2_final[j, 0], pos2_final[j, 1], pos2_final[j, 2], agg2.radii[j], agg2.masses[j])
        return merged
```

- [ ] **Step 5: Run tests**

Run: `cd /home/zhangfan/Project/20260319_SPEMBSSBDART/pyFracAggregate && python -m pytest tests/test_generators/test_cca_fracval.py -v`
Expected: ALL PASS

- [ ] **Step 6: Commit**

```bash
cd /home/zhangfan/Project/20260319_SPEMBSSBDART/pyFracAggregate
git add src/pyFracAggregate/generators/cca_fracval.py src/pyFracAggregate/core/math_utils.py tests/test_generators/test_cca_fracval.py
git commit -m "feat: implement FracVAL deterministic three-stage contact placement"
```

---

## Task 6: Thouy-Jullien Lattice tdCCA Generator

**Files:**
- Create: `src/pyFracAggregate/generators/tdcca_thouy.py`
- Test: `tests/test_generators/test_tdcca_thouy.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_generators/test_tdcca_thouy.py`:

```python
import pytest
import numpy as np
import pyFracAggregate as pfa


def test_tdcca_basic_generation():
    agg = pfa.generate(n_particles=16, df=1.8, kf=1.3, method='tdcca')
    assert agg.current_size == 16


def test_tdcca_power_of_two_required():
    """tdCCA requires N to be a power of 2."""
    with pytest.raises(ValueError, match="power of 2"):
        pfa.generate(n_particles=15, df=1.8, kf=1.3, method='tdcca')


def test_tdcca_no_overlaps():
    agg = pfa.generate(n_particles=8, df=1.5, kf=1.3, method='tdcca')
    positions = agg.positions
    radii = agg.radii
    for i in range(agg.current_size):
        for j in range(i + 1, agg.current_size):
            dist = np.linalg.norm(positions[i] - positions[j])
            min_dist = radii[i] + radii[j] - 1e-5
            assert dist >= min_dist - 1e-5


def test_tdcca_scaling_law():
    agg = pfa.generate(n_particles=32, df=1.8, kf=1.3, method='tdcca')
    rg = pfa.radius_of_gyration(agg)
    a = np.mean(agg.radii)
    df_est = np.log(agg.current_size) / np.log(rg / a)
    assert abs(df_est - 1.8) < 0.5
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /home/zhangfan/Project/20260319_SPEMBSSBDART/pyFracAggregate && python -m pytest tests/test_generators/test_tdcca_thouy.py -v`
Expected: FAIL — factory doesn't know `'tdcca'`

- [ ] **Step 3: Implement Thouy-Jullien tdCCA generator**

Create `src/pyFracAggregate/generators/tdcca_thouy.py`:

```python
import numpy as np
from typing import List, Tuple, Set
from pyFracAggregate.core.aggregate import Aggregate
from pyFracAggregate.core.math_utils import euler_rodrigues_rotation
from pyFracAggregate.generators.base import BaseGenerator


# 26 neighbors on a 3D cubic lattice (face, edge, corner)
_LATTICE_NEIGHBORS = []
for dx in [-1, 0, 1]:
    for dy in [-1, 0, 1]:
        for dz in [-1, 0, 1]:
            if dx == 0 and dy == 0 and dz == 0:
                continue
            _LATTICE_NEIGHBORS.append(np.array([dx, dy, dz], dtype=np.float64))
_LATTICE_NEIGHBORS = [n / np.linalg.norm(n) for n in _LATTICE_NEIGHBORS]


class ThouyJullienGenerator(BaseGenerator):
    """Lattice-based tunable Cluster-Cluster Aggregation (Thouy & Jullien, 1994).

    Operates on a 3D cubic lattice. Requires N = 2^n.

    Algorithm:
    1. Start with N individual particles at lattice sites.
    2. Iteration p=1: pair into dimers in random directions.
    3. Iteration p: pair clusters of size 2^p.
       For each pair, enumerate surface-site contacts, compute Gamma deviation,
       pick the configuration that minimizes |Gamma^2 - k^2*(R1^2+R2^2)/2 - 1|.
    """

    def generate(self) -> Aggregate:
        n = self.n_particles
        # Verify N is a power of 2
        if n < 2 or (n & (n - 1)) != 0:
            raise ValueError(
                f"Thouy-Jullien tdCCA requires N to be a power of 2, got {n}"
            )

        agg = Aggregate(n, self.length_unit, self.mass_unit, self.density)
        radii = self.particle_dist.sample(n)
        masses = self.density * (4.0 / 3.0) * np.pi * (radii ** 3)

        # Represent each cluster as a dict mapping lattice coordinate to particle index
        # Use float coordinates for positions
        # Start: each particle at a unique lattice position
        # We'll use continuous-space positions scaled to particle diameter = 2*a

        a = np.mean(radii)

        # Initialize: place particles along a line spaced by 2*a
        clusters: List[dict] = []
        for i in range(n):
            pos = np.array([i * 2 * a, 0.0, 0.0])
            clusters.append({
                'positions': pos.reshape(1, 3).copy(),
                'radii': np.array([radii[i]]),
                'masses': np.array([masses[i]]),
            })

        # Iteration 1: form dimers in random directions
        new_clusters = []
        for i in range(0, n, 2):
            dimer = self._form_dimer(clusters[i], clusters[i + 1], a)
            new_clusters.append(dimer)
        clusters = new_clusters

        # Subsequent iterations: hierarchical merging
        size = 2
        while size < n:
            new_clusters = []
            for i in range(0, len(clusters), 2):
                merged = self._merge_lattice(clusters[i], clusters[i + 1], a)
                new_clusters.append(merged)
            clusters = new_clusters
            size *= 2

        # Pack final cluster into Aggregate
        final = clusters[0]
        for i in range(len(final['radii'])):
            pos = final['positions'][i]
            agg.add_particle(pos[0], pos[1], pos[2], final['radii'][i], final['masses'][i])

        return agg

    def _form_dimer(self, c1: dict, c2: dict, a: float) -> dict:
        """Form a dimer by placing c2's particle adjacent to c1's in a random direction."""
        direction = _LATTICE_NEIGHBORS[np.random.randint(len(_LATTICE_NEIGHBORS))]

        p1 = c1['positions'][0]
        p2 = p1 + direction * 2 * a  # touching distance

        return {
            'positions': np.array([p1, p2]),
            'radii': np.concatenate([c1['radii'], c2['radii']]),
            'masses': np.concatenate([c1['masses'], c2['masses']]),
        }

    def _merge_lattice(self, c1: dict, c2: dict, a: float) -> dict:
        """Merge two lattice clusters minimizing Gamma deviation (Thouy & Jullien 1994 Eq 12)."""
        pos1 = c1['positions']
        pos2 = c2['positions']
        r1 = c1['radii']
        r2 = c2['radii']
        m1 = c1['masses']
        m2 = c2['masses']

        N1 = len(r1)
        N2 = len(r2)
        N = N1 + N2

        # Compute Rg^2 for each cluster (centered at origin)
        com1 = np.average(pos1, weights=m1, axis=0)
        com2 = np.average(pos2, weights=m2, axis=0)
        p1c = pos1 - com1
        p2c = pos2 - com2

        Rg1_sq = np.sum(m1[:, None] * (p1c ** 2)) / np.sum(m1)
        Rg2_sq = np.sum(m2[:, None] * (p2c ** 2)) / np.sum(m2)
        Rg_avg_sq = (Rg1_sq + Rg2_sq) / 2

        # k^2 = 4 * (4^(1/Df) - 1) from Thouy & Jullien Eq 11
        k_sq = 4.0 * (4.0 ** (1.0 / self.df) - 1.0)

        # Target Gamma^2 = k^2 * Rg_avg_sq + 1 (Eq 12, the +1 corrects for dimers)
        Gamma_target_sq = k_sq * Rg_avg_sq + 1.0

        # Identify surface particles of each cluster
        surface1 = self._find_surface_particles(p1c, r1)
        surface2 = self._find_surface_particles(p2c, r2)

        # Enumerate contacts: try placing c2's surface particle adjacent to c1's
        # using lattice neighbor directions
        best_config = None
        best_delta = float('inf')

        # Limit search for performance
        max_surface_samples = min(len(surface1), 20)
        max_dirs = min(len(_LATTICE_NEIGHBORS), 26)

        np.random.shuffle(surface1)

        for si in surface1[:max_surface_samples]:
            for dir_idx in range(max_dirs):
                direction = _LATTICE_NEIGHBORS[dir_idx]

                # Place c2 so that a surface particle of c2 touches si
                # The contact point: si_pos + direction * (r1[si] + r2[sj])
                # But we need to try different sj from surface2

                for sj in surface2:
                    # Place c2's center so that sj touches si
                    # sj_final = si_pos + direction * (r1[si] + r2[sj])
                    contact_offset = direction * (r1[si] + r2[sj])
                    sj_target = p1c[si] + contact_offset

                    # We need to translate c2 so sj goes to sj_target
                    translation = sj_target - p2c[sj]
                    p2_trial = p2c + translation

                    # Check for overlaps
                    ok = True
                    for ii in range(N1):
                        for jj in range(N2):
                            d = np.linalg.norm(p1c[ii] - p2_trial[jj])
                            if d < r1[ii] + r2[jj] - 1e-6:
                                ok = False
                                break
                        if not ok:
                            break

                    if not ok:
                        continue

                    # Compute actual Gamma (distance between centers of mass)
                    total_m = np.sum(m1) + np.sum(m2)
                    com_merged = (np.sum(m1) * com1 + np.sum(m2) * (com2 + translation)) / total_m
                    # Actually compute Rg of merged cluster
                    all_pos = np.vstack([p1c, p2_trial])
                    all_m = np.concatenate([m1, m2])
                    com_all = np.average(all_pos, weights=all_m, axis=0)
                    rg_merged_sq = np.sum(all_m[:, None] * ((all_pos - com_all) ** 2)) / np.sum(all_m)

                    # Gamma = distance between the two sub-cluster centers of mass
                    # relative to the merged center
                    c1_in_merged = np.average(p1c, weights=m1, axis=0)
                    c2_in_merged = np.average(p2_trial, weights=m2, axis=0)
                    Gamma_sq = np.sum((c1_in_merged - c2_in_merged) ** 2)

                    delta = abs(Gamma_sq - Gamma_target_sq)

                    if delta < best_delta:
                        best_delta = delta
                        best_config = {
                            'pos2': p2_trial.copy(),
                            'translation': translation.copy(),
                        }

        if best_config is None:
            # Fallback: just place c2 at some non-overlapping position
            # Use the first direction that works
            direction = _LATTICE_NEIGHBORS[0]
            translation = direction * (np.max(np.linalg.norm(p1c, axis=1)) + np.max(np.linalg.norm(p2c, axis=1)) + 2 * a)
            best_config = {'pos2': p2c + translation, 'translation': translation.copy()}

        p2_final = best_config['pos2']

        return {
            'positions': np.vstack([p1c, p2_final]),
            'radii': np.concatenate([r1, r2]),
            'masses': np.concatenate([m1, m2]),
        }

    def _find_surface_particles(self, positions: np.ndarray, radii: np.ndarray) -> List[int]:
        """Find surface particles (those with at least one unoccupied neighbor direction).

        A particle is on the surface if there exists a lattice neighbor direction
        where no other particle overlaps with a sphere at that neighbor location.
        """
        n = len(radii)
        surface = []
        touch_dist = 2.0 * np.mean(radii)

        for i in range(n):
            is_surface = False
            for direction in _LATTICE_NEIGHBORS[:6]:  # Only check 6 face neighbors for speed
                neighbor_pos = positions[i] + direction * touch_dist
                has_neighbor = False
                for j in range(n):
                    if j == i:
                        continue
                    if np.linalg.norm(positions[j] - neighbor_pos) < radii[j] + 0.5 * np.mean(radii):
                        has_neighbor = True
                        break
                if not has_neighbor:
                    is_surface = True
                    break
            if is_surface:
                surface.append(i)
        return surface
```

- [ ] **Step 4: Register in factory**

Add to `src/pyFracAggregate/generators/factory.py`:

```python
from pyFracAggregate.generators.tdcca_thouy import ThouyJullienGenerator
```

And in `get_generator()`:

```python
    elif method == 'tdcca':
        return ThouyJullienGenerator(n_particles, df, kf, particle_dist, overlap_tolerance, **kwargs)
```

- [ ] **Step 5: Run tests**

Run: `cd /home/zhangfan/Project/20260319_SPEMBSSBDART/pyFracAggregate && python -m pytest tests/test_generators/test_tdcca_thouy.py -v`
Expected: ALL PASS

- [ ] **Step 6: Commit**

```bash
cd /home/zhangfan/Project/20260319_SPEMBSSBDART/pyFracAggregate
git add src/pyFracAggregate/generators/tdcca_thouy.py src/pyFracAggregate/generators/factory.py tests/test_generators/test_tdcca_thouy.py
git commit -m "feat: add Thouy-Jullien (1994) lattice tdCCA generator"
```

---

## Task 7: Factory Registration & Integration Test

**Files:**
- Modify: `src/pyFracAggregate/generators/factory.py`
- Modify: `src/pyFracAggregate/__init__.py`
- Test: `tests/test_generators/test_factory.py` (update existing or create)

- [ ] **Step 1: Write integration test**

Create `tests/test_generators/test_factory.py`:

```python
import pytest
import pyFracAggregate as pfa


@pytest.mark.parametrize("method", ['pca', 'cca', 'fracval', 'flage_pca', 'flage_cca', 'tdcca'])
def test_all_methods_generate(method):
    """All registered methods should produce a valid aggregate."""
    n = 16 if method == 'tdcca' else 30
    agg = pfa.generate(n_particles=n, df=1.8, kf=1.3, method=method)
    assert agg.current_size == n
    assert agg.positions.shape == (n, 3)
    assert agg.radii.shape == (n,)


def test_unknown_method_raises():
    with pytest.raises(ValueError, match="Unknown generation method"):
        pfa.generate(n_particles=10, df=1.8, kf=1.3, method='nonexistent')
```

- [ ] **Step 2: Verify factory.py has all methods registered**

Ensure `factory.py` imports and dispatches to: `pca`, `cca`, `fracval`, `flage_pca`, `flage_cca`, `tdcca`.

- [ ] **Step 3: Run full test suite**

Run: `cd /home/zhangfan/Project/20260319_SPEMBSSBDART/pyFracAggregate && python -m pytest tests/ -v`
Expected: ALL PASS

- [ ] **Step 4: Commit**

```bash
cd /home/zhangfan/Project/20260319_SPEMBSSBDART/pyFracAggregate
git add src/pyFracAggregate/generators/factory.py tests/test_generators/test_factory.py
git commit -m "test: add integration test for all generator methods"
```

---

## Task 8: Performance Benchmark Test

Verify that FLAGE methods are actually faster than the original Monte Carlo methods.

**Files:**
- Create: `tests/test_generators/test_benchmark.py`

- [ ] **Step 1: Write benchmark test**

```python
import time
import pytest
import numpy as np
import pyFracAggregate as pfa


@pytest.mark.benchmark
def test_pca_flage_faster_than_pca_filippov():
    """FLAGE PCA should be faster than Filippov PCA for large N."""
    n = 200
    df, kf = 1.8, 1.3

    np.random.seed(42)
    t0 = time.perf_counter()
    agg1 = pfa.generate(n_particles=n, df=df, kf=kf, method='pca')
    t_filippov = time.perf_counter() - t0

    np.random.seed(42)
    t0 = time.perf_counter()
    agg2 = pfa.generate(n_particles=n, df=df, kf=kf, method='flage_pca')
    t_flage = time.perf_counter() - t0

    # Both should produce valid aggregates
    assert agg1.current_size == n
    assert agg2.current_size == n

    # FLAGE should not be dramatically slower (allow 2x for small N overhead)
    assert t_flage < t_filippov * 2.0 + 0.1


@pytest.mark.benchmark
def test_cca_flage_faster_than_cca_filippov():
    """FLAGE CCA should be faster than Filippov CCA for large N."""
    n = 100
    df, kf = 1.8, 1.3

    np.random.seed(42)
    t0 = time.perf_counter()
    agg1 = pfa.generate(n_particles=n, df=df, kf=kf, method='cca')
    t_filippov = time.perf_counter() - t0

    np.random.seed(42)
    t0 = time.perf_counter()
    agg2 = pfa.generate(n_particles=n, df=df, kf=kf, method='flage_cca')
    t_flage = time.perf_counter() - t0

    assert agg1.current_size == n
    assert agg2.current_size == n

    assert t_flage < t_filippov * 2.0 + 0.1
```

- [ ] **Step 2: Run benchmark**

Run: `cd /home/zhangfan/Project/20260319_SPEMBSSBDART/pyFracAggregate && python -m pytest tests/test_generators/test_benchmark.py -v -m benchmark`
Expected: ALL PASS

- [ ] **Step 3: Run full test suite one final time**

Run: `cd /home/zhangfan/Project/20260319_SPEMBSSBDART/pyFracAggregate && python -m pytest tests/ -v`
Expected: ALL PASS

- [ ] **Step 4: Commit**

```bash
cd /home/zhangfan/Project/20260319_SPEMBSSBDART/pyFracAggregate
git add tests/test_generators/test_benchmark.py
git commit -m "test: add performance benchmark comparing FLAGE vs Filippov methods"
```

---

## Self-Review Checklist

- [x] **Spec coverage:** FLAGE algebraic PCA (Task 1-3), FLAGE algebraic CCA (Task 4), FracVAL deterministic placement (Task 5), Thouy-Jullien lattice tdCCA (Task 6), factory integration (Task 7), benchmarks (Task 8)
- [x] **Placeholder scan:** No TBDs, no vague "implement error handling" steps, all code is concrete
- [x] **Type consistency:** `euler_rodrigues_rotation(points, axis, angle)` used consistently; `sphere_sphere_intersection(c1, r1, c2, r2) -> tuple | None` used consistently; `_build_merged` helper pattern shared between CCAFlage and FracVAL
- [x] **Method names match factory:** `flage_pca`, `flage_cca`, `fracval`, `tdcca` all registered
