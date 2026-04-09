# Placement Strategy Refactor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Refactor FLAGE from a top-level algorithm choice into a pluggable placement strategy, reducing user-facing methods from 6 to 4.

**Architecture:** Extract placement logic (algebraic vs random) into a `PlacementStrategy` ABC with two implementations. Merge duplicate generator files (`pca_filippov` + `pca_flage` → `pca`, `cca_filippov` + `cca_flage` → `cca`). Remove `optimizer_flage.py` (migrated into `placement/algebraic.py`). Update factory and top-level API.

**Tech Stack:** Python 3.9+, NumPy, SciPy, pytest

---

### Task 1: Create PlacementStrategy ABC

**Files:**
- Create: `src/pyFracAggregate/generators/placement/__init__.py`
- Create: `src/pyFracAggregate/generators/placement/base.py`
- Test: `tests/test_generators/test_placement.py`

- [ ] **Step 1: Write the failing test for `get_placement` factory and ABC interface**

```python
# tests/test_generators/test_placement.py
import pytest
from pyFracAggregate.generators.placement import get_placement, PlacementStrategy


def test_get_placement_algebraic():
    s = get_placement('algebraic')
    assert isinstance(s, PlacementStrategy)


def test_get_placement_random():
    s = get_placement('random')
    assert isinstance(s, PlacementStrategy)


def test_get_placement_invalid_raises():
    with pytest.raises(ValueError, match="Unknown placement strategy"):
        get_placement('nonexistent')


def test_placement_strategy_has_place_particle():
    """PlacementStrategy ABC should require place_particle."""
    import inspect
    assert 'place_particle' in PlacementStrategy.__abstractmethods__


def test_placement_strategy_has_merge_clusters():
    """PlacementStrategy ABC should require merge_clusters."""
    import inspect
    assert 'merge_clusters' in PlacementStrategy.__abstractmethods__
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_generators/test_placement.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'pyFracAggregate.generators.placement'`

- [ ] **Step 3: Write minimal `PlacementStrategy` ABC and `get_placement` factory**

```python
# src/pyFracAggregate/generators/placement/__init__.py
from pyFracAggregate.generators.placement.base import PlacementStrategy, get_placement

__all__ = ["PlacementStrategy", "get_placement"]
```

```python
# src/pyFracAggregate/generators/placement/base.py
from abc import ABC, abstractmethod

import numpy as np

from pyFracAggregate.core.aggregate import Aggregate


class PlacementStrategy(ABC):
    """Strategy for placing particles during fractal aggregate generation."""

    @abstractmethod
    def place_particle(
        self,
        agg: Aggregate,
        candidate_radius: float,
        candidate_mass: float,
        geom_center: np.ndarray,
        L: float,
        mean_radius: float,
    ) -> tuple:
        """Place a single particle onto the Gamma sphere (PCA stage).

        Args:
            agg: Current aggregate with existing particles.
            candidate_radius: Radius of the new particle.
            candidate_mass: Mass of the new particle.
            geom_center: Geometric center of existing particles.
            L: Required distance from center to new particle.
            mean_radius: Mean particle radius.

        Returns:
            (x, y, z) position tuple, or None if placement failed.
        """

    @abstractmethod
    def merge_clusters(
        self,
        pos1: np.ndarray,
        r1: np.ndarray,
        agg1: Aggregate,
        pos2_centered: np.ndarray,
        r2: np.ndarray,
        agg2: Aggregate,
        Gamma: float,
        mean_radius: float,
    ) -> np.ndarray:
        """Merge two sub-clusters (CCA stage).

        Args:
            pos1: Positions of cluster 1 centered at origin.
            r1: Radii of cluster 1.
            agg1: Cluster 1 aggregate.
            pos2_centered: Positions of cluster 2 centered at its COM.
            r2: Radii of cluster 2.
            agg2: Cluster 2 aggregate.
            Gamma: Required COM distance between clusters.
            mean_radius: Mean particle radius.

        Returns:
            pos2_final array (N2, 3), or None if merge failed.
        """


def get_placement(name: str) -> PlacementStrategy:
    """Factory for placement strategies.

    Args:
        name: 'algebraic' (default, FLAGE) or 'random' (Filippov).

    Raises:
        ValueError: If name is not recognized.
    """
    # Will be populated in Task 2 and Task 3
    raise ValueError(f"Unknown placement strategy: {name}")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_generators/test_placement.py -v`
Expected: FAIL — `get_placement` always raises, so `test_get_placement_algebraic` and `test_get_placement_random` fail. `get_placement_invalid_raises` passes.

- [ ] **Step 5: Commit**

```bash
git add src/pyFracAggregate/generators/placement/__init__.py \
        src/pyFracAggregate/generators/placement/base.py \
        tests/test_generators/test_placement.py
git commit -m "feat: add PlacementStrategy ABC and factory stub"
```

---

### Task 2: Implement AlgebraicPlacement

**Files:**
- Create: `src/pyFracAggregate/generators/placement/algebraic.py`
- Modify: `src/pyFracAggregate/generators/placement/base.py` (wire factory)
- Test: `tests/test_generators/test_placement.py`

- [ ] **Step 1: Write failing tests for AlgebraicPlacement**

Add to `tests/test_generators/test_placement.py`:

```python
from pyFracAggregate.generators.placement.algebraic import AlgebraicPlacement
from pyFracAggregate.generators.placement.base import PlacementStrategy


def test_algebraic_placement_is_strategy():
    assert issubclass(AlgebraicPlacement, PlacementStrategy)


def test_algebraic_placement_place_particle_basic():
    """AlgebraicPlacement.place_particle should find valid positions on simple aggregates."""
    import numpy as np
    from pyFracAggregate.core.aggregate import Aggregate
    from pyFracAggregate.core.distributions import Monodisperse

    strategy = AlgebraicPlacement()
    agg = Aggregate(3, density=1.0)
    agg.add_particle(0.0, 0.0, 0.0, 1.0, agg.density * (4.0 / 3.0) * np.pi * 1.0**3)
    agg.add_particle(2.0, 0.0, 0.0, 1.0, agg.density * (4.0 / 3.0) * np.pi * 1.0**3)

    geom_center = np.mean(agg.positions, axis=0)
    result = strategy.place_particle(agg, 1.0, agg.density * (4.0 / 3.0) * np.pi, geom_center, 3.0, 1.0)
    # Should find a valid position (may return None if geometry is tight)
    assert result is None or len(result) == 3


def test_algebraic_placement_place_particle_returns_none_on_blocked():
    """If all positions blocked, should return None."""
    import numpy as np
    from pyFracAggregate.core.aggregate import Aggregate

    strategy = AlgebraicPlacement()
    agg = Aggregate(10, density=1.0)
    # Pack particles tightly around origin
    np.random.seed(42)
    for i in range(10):
        r = 1.0
        pos = np.random.normal(0, 0.5, 3)
        agg.add_particle(pos[0], pos[1], pos[2], r, agg.density * (4.0 / 3.0) * np.pi)

    geom_center = np.mean(agg.positions, axis=0)
    # Very small L — almost impossible to place
    result = strategy.place_particle(agg, 1.0, agg.density * (4.0 / 3.0) * np.pi, geom_center, 0.5, 1.0)
    assert result is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_generators/test_placement.py::test_algebraic_placement_is_strategy -v`
Expected: FAIL with `ImportError`

- [ ] **Step 3: Write AlgebraicPlacement**

```python
# src/pyFracAggregate/generators/placement/algebraic.py
import numpy as np
from pyFracAggregate.core.aggregate import Aggregate
from pyFracAggregate.core.math_utils import rotate_points
from pyFracAggregate.generators.placement.base import PlacementStrategy
from pyFracAggregate.generators.optimizer_flage import (
    build_particle_list_pca,
    find_exact_touching_points_pca,
    filter_overlapping_candidates,
)


class AlgebraicPlacement(PlacementStrategy):
    """FLAGE algebraic placement (Skorupski et al., 2014).

    Uses sphere-sphere intersection to compute exact touching points,
    with random Monte Carlo as fallback.
    """

    def place_particle(self, agg, candidate_radius, candidate_mass, geom_center, L, mean_radius):
        """Try algebraic placement, then fall back to random sampling."""
        # --- FLAGE algebraic path ---
        candidate_list = build_particle_list_pca(agg.positions, agg.radii, L, mean_radius)

        if len(candidate_list) > 0:
            np.random.shuffle(candidate_list)
            max_ref = min(5, len(candidate_list))
            for i in range(max_ref):
                ref_idx = candidate_list[i % len(candidate_list)]
                ref_pos = agg.positions[ref_idx]
                r_ref = agg.radii[ref_idx]

                candidates = find_exact_touching_points_pca(
                    geom_center, L, ref_pos, candidate_radius, r_ref, num_points=8
                )
                if len(candidates) == 0:
                    continue

                valid = filter_overlapping_candidates(
                    candidates, agg.positions, agg.radii, candidate_radius, agg.overlap_tolerance
                )
                if len(valid) > 0:
                    pt = valid[np.random.randint(len(valid))]
                    return (pt[0], pt[1], pt[2])

        # --- Fallback: random Monte Carlo ---
        return self._random_place_particle(agg, candidate_radius, geom_center, L, mean_radius)

    def _random_place_particle(self, agg, r_N, geom_center, L, a):
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
            min_allowed = agg.radii + r_N - agg.overlap_tolerance

            if np.any(dists < min_allowed):
                continue
            if np.any(dists <= min_allowed + tolerance):
                return (candidate_pos[0], candidate_pos[1], candidate_pos[2])
            if attempt > 0 and attempt % 1000 == 0:
                tolerance += 0.05 * a

        # Extreme fallback: attach to random particle surface
        idx = np.random.randint(agg.current_size)
        ref_pos = agg.positions[idx]
        u = np.random.normal(size=3)
        u /= np.linalg.norm(u)
        fallback_pos = ref_pos + (agg.radii[idx] + r_N - agg.overlap_tolerance) * u
        return (fallback_pos[0], fallback_pos[1], fallback_pos[2])

    def merge_clusters(self, pos1, r1, agg1, pos2_centered, r2, agg2, Gamma, mean_radius):
        """FLAGE-style merge with surface particle filtering + random fallback."""
        N1 = agg1.current_size
        N2 = agg2.current_size
        N = N1 + N2

        # Check feasibility
        D1_max = np.max(np.linalg.norm(pos1, axis=1) + r1)
        D2_max = np.max(np.linalg.norm(pos2_centered, axis=1) + r2)

        if D1_max + D2_max < Gamma:
            return self._random_merge(pos1, r1, agg1, pos2_centered, r2, agg2, Gamma, mean_radius)

        # Build surface lists
        dists1 = np.linalg.norm(pos1, axis=1)
        la1 = dists1 + r1
        la2 = np.linalg.norm(pos2_centered, axis=1) + r2

        surface1_idx = np.where(la1 >= Gamma * 0.3)[0]
        surface2_idx = np.where(la2 >= Gamma * 0.3)[0]

        if len(surface1_idx) == 0:
            surface1_idx = np.arange(N1)
        if len(surface2_idx) == 0:
            surface2_idx = np.arange(N2)

        np.random.shuffle(surface1_idx)
        np.random.shuffle(surface2_idx)

        max_ref_tries = min(50, N1 * N2)
        ref_try = 0

        for si in surface1_idx:
            for sj in surface2_idx:
                ref_try += 1
                if ref_try > max_ref_tries:
                    break

                u = np.random.normal(size=3)
                u /= np.linalg.norm(u)
                new_com2 = Gamma * u

                euler = np.random.uniform(0, 2 * np.pi, size=3)
                pos2_rot = rotate_points(pos2_centered, tuple(euler))
                pos2_trial = pos2_rot + new_com2

                dists = np.linalg.norm(
                    pos1[:, np.newaxis, :] - pos2_trial[np.newaxis, :, :], axis=2
                )
                min_dists = r1[:, np.newaxis] + r2[np.newaxis, :] - agg1.overlap_tolerance
                gaps = dists - min_dists

                if not np.any(gaps < 0):
                    min_gap = np.min(gaps)
                    if min_gap <= 1e-3 * mean_radius:
                        return pos2_trial

            if ref_try > max_ref_tries:
                break

        return self._random_merge(pos1, r1, agg1, pos2_centered, r2, agg2, Gamma, mean_radius)

    def _random_merge(self, pos1, r1, agg1, pos2_centered, r2, agg2, Gamma, mean_radius):
        """Random rotation + collision detection fallback."""
        max_attempts = 20000
        tol = 1e-3 * mean_radius
        candidate_pos2 = None

        for attempt in range(max_attempts):
            u = np.random.normal(size=3)
            u /= np.linalg.norm(u)
            new_com2 = Gamma * u

            euler_angles = np.random.uniform(0, 2 * np.pi, size=3)
            pos2_rotated = rotate_points(pos2_centered, tuple(euler_angles))
            candidate_pos2 = pos2_rotated + new_com2

            dists = np.linalg.norm(
                pos1[:, np.newaxis, :] - candidate_pos2[np.newaxis, :, :], axis=2
            )
            min_dists = r1[:, np.newaxis] + r2[np.newaxis, :] - agg1.overlap_tolerance
            gaps = dists - min_dists

            if np.any(gaps < 0):
                continue
            if np.min(gaps) <= tol:
                return candidate_pos2
            if attempt > 0 and attempt % 2000 == 0:
                tol += 0.05 * mean_radius

        if candidate_pos2 is None:
            candidate_pos2 = pos2_rotated + new_com2
        return candidate_pos2
```

- [ ] **Step 4: Wire factory in `base.py`**

Update `get_placement` in `src/pyFracAggregate/generators/placement/base.py`:

```python
def get_placement(name: str) -> PlacementStrategy:
    from pyFracAggregate.generators.placement.algebraic import AlgebraicPlacement
    from pyFracAggregate.generators.placement.random_ import RandomPlacement

    name = name.lower()
    if name == 'algebraic':
        return AlgebraicPlacement()
    elif name == 'random':
        return RandomPlacement()
    raise ValueError(f"Unknown placement strategy: {name}")
```

Note: the `RandomPlacement` import will fail until Task 3, so tests for `algebraic` pass but `random` will fail. This is expected — we'll fix it in Task 3.

- [ ] **Step 5: Run placement tests**

Run: `pytest tests/test_generators/test_placement.py -v`
Expected: `test_get_placement_algebraic` PASS, `test_get_placement_random` FAIL (no RandomPlacement yet), ABC tests PASS, algebraic-specific tests PASS.

- [ ] **Step 6: Commit**

```bash
git add src/pyFracAggregate/generators/placement/algebraic.py \
        src/pyFracAggregate/generators/placement/base.py \
        tests/test_generators/test_placement.py
git commit -m "feat: implement AlgebraicPlacement (FLAGE) strategy"
```

---

### Task 3: Implement RandomPlacement

**Files:**
- Create: `src/pyFracAggregate/generators/placement/random_.py`
- Test: `tests/test_generators/test_placement.py`

- [ ] **Step 1: Write failing test for RandomPlacement**

Add to `tests/test_generators/test_placement.py`:

```python
from pyFracAggregate.generators.placement.random_ import RandomPlacement


def test_random_placement_is_strategy():
    assert issubclass(RandomPlacement, PlacementStrategy)


def test_random_placement_place_particle_basic():
    """RandomPlacement.place_particle should find valid positions via Monte Carlo."""
    import numpy as np
    from pyFracAggregate.core.aggregate import Aggregate

    strategy = RandomPlacement()
    agg = Aggregate(3, density=1.0)
    agg.add_particle(0.0, 0.0, 0.0, 1.0, agg.density * (4.0 / 3.0) * np.pi)

    geom_center = np.array([0.0, 0.0, 0.0])
    result = strategy.place_particle(agg, 1.0, agg.density * (4.0 / 3.0) * np.pi, geom_center, 3.0, 1.0)
    assert result is not None
    assert len(result) == 3
    # Distance to origin should be close to L=3.0
    pos = np.array(result)
    assert abs(np.linalg.norm(pos - geom_center) - 3.0) < 0.2
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_generators/test_placement.py::test_random_placement_is_strategy -v`
Expected: FAIL with `ImportError`

- [ ] **Step 3: Write RandomPlacement**

```python
# src/pyFracAggregate/generators/placement/random_.py
import numpy as np
from pyFracAggregate.core.aggregate import Aggregate
from pyFracAggregate.core.math_utils import rotate_points
from pyFracAggregate.generators.placement.base import PlacementStrategy


class RandomPlacement(PlacementStrategy):
    """Random Monte Carlo placement (Filippov et al., 2000).

    Samples positions on the Gamma sphere with gradual tolerance relaxation.
    """

    def place_particle(self, agg, candidate_radius, candidate_mass, geom_center, L, mean_radius):
        """Random sampling on Gamma sphere with tolerance relaxation."""
        r_N = candidate_radius
        max_attempts = 10000
        tolerance = 1e-3 * mean_radius

        for attempt in range(max_attempts):
            u = np.random.normal(size=3)
            norm_u = np.linalg.norm(u)
            if norm_u < 1e-8:
                continue
            u /= norm_u

            candidate_pos = geom_center + L * u
            dists = np.linalg.norm(agg.positions - candidate_pos, axis=1)
            min_allowed_dists = agg.radii + r_N - agg.overlap_tolerance

            if np.any(dists < min_allowed_dists):
                continue

            if np.any(dists <= min_allowed_dists + tolerance):
                return (candidate_pos[0], candidate_pos[1], candidate_pos[2])

            if attempt > 0 and attempt % 1000 == 0:
                tolerance += 0.05 * mean_radius

        # Extreme fallback
        idx = np.random.randint(agg.current_size)
        ref_pos = agg.positions[idx]
        u = np.random.normal(size=3)
        u /= np.linalg.norm(u)
        fallback_pos = ref_pos + (agg.radii[idx] + r_N - agg.overlap_tolerance) * u
        return (fallback_pos[0], fallback_pos[1], fallback_pos[2])

    def merge_clusters(self, pos1, r1, agg1, pos2_centered, r2, agg2, Gamma, mean_radius):
        """Random rotation + collision detection."""
        N1 = agg1.current_size
        N2 = agg2.current_size
        N = N1 + N2
        max_attempts = 20000
        tolerance = 1e-3 * mean_radius

        best_candidate = None
        min_gap = float('inf')
        candidate_pos2 = None

        for attempt in range(max_attempts):
            u = np.random.normal(size=3)
            u /= np.linalg.norm(u)
            new_com2 = Gamma * u

            euler_angles = np.random.uniform(0, 2 * np.pi, size=3)
            pos2_rotated = rotate_points(pos2_centered, tuple(euler_angles))
            candidate_pos2 = pos2_rotated + new_com2

            dists = np.linalg.norm(
                pos1[:, np.newaxis, :] - candidate_pos2[np.newaxis, :, :], axis=2
            )
            min_dists = r1[:, np.newaxis] + r2[np.newaxis, :] - agg1.overlap_tolerance
            gaps = dists - min_dists

            if np.any(gaps < 0):
                continue

            current_min_gap = np.min(gaps)
            if current_min_gap < min_gap:
                min_gap = current_min_gap
                best_candidate = candidate_pos2.copy()

            if current_min_gap <= tolerance:
                return candidate_pos2

            if attempt > 0 and attempt % 2000 == 0:
                tolerance += 0.05 * mean_radius

        if best_candidate is None:
            best_candidate = candidate_pos2

        return best_candidate
```

Note: file is `random_.py` (trailing underscore) to avoid shadowing Python's `random` module name in imports.

- [ ] **Step 4: Run all placement tests**

Run: `pytest tests/test_generators/test_placement.py -v`
Expected: All PASS.

- [ ] **Step 5: Commit**

```bash
git add src/pyFracAggregate/generators/placement/random_.py \
        tests/test_generators/test_placement.py
git commit -m "feat: implement RandomPlacement (Filippov Monte Carlo) strategy"
```

---

### Task 4: Add placement parameter to BaseGenerator

**Files:**
- Modify: `src/pyFracAggregate/generators/base.py`
- Modify: `src/pyFracAggregate/generators/__init__.py`
- Test: `tests/test_generators/test_placement.py`

- [ ] **Step 1: Write failing test for BaseGenerator placement integration**

Add to `tests/test_generators/test_placement.py`:

```python
from pyFracAggregate.generators.base import BaseGenerator


def test_base_generator_default_placement():
    """BaseGenerator should default to algebraic placement."""
    gen = PCAGenerator(10, 1.8, 1.3, pfa.Monodisperse(1.0))
    from pyFracAggregate.generators.placement.algebraic import AlgebraicPlacement
    assert isinstance(gen.placement, AlgebraicPlacement)


def test_base_generator_random_placement():
    """BaseGenerator should accept placement='random'."""
    gen = PCAGenerator(10, 1.8, 1.3, pfa.Monodisperse(1.0), placement='random')
    from pyFracAggregate.generators.placement.random_ import RandomPlacement
    assert isinstance(gen.placement, RandomPlacement)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_generators/test_placement.py -k "base_generator" -v`
Expected: FAIL — `BaseGenerator.__init__` doesn't accept `placement` kwarg.

- [ ] **Step 3: Add `placement` to BaseGenerator**

Replace `src/pyFracAggregate/generators/base.py`:

```python
from abc import ABC, abstractmethod
from pyFracAggregate.core.aggregate import Aggregate
from pyFracAggregate.core.distributions import ParticleDistribution
from pyFracAggregate.generators.placement.base import PlacementStrategy, get_placement


class BaseGenerator(ABC):
    """Abstract base class for generators."""
    def __init__(
        self,
        n_particles: int,
        df: float,
        kf: float,
        particle_dist: ParticleDistribution,
        overlap_tolerance: float = 0.0,
        length_unit: str = 'nm',
        mass_unit: str = 'g',
        density: float = 1.0,
        placement: str = 'algebraic',
    ):
        self.n_particles = n_particles
        self.df = df
        self.kf = kf
        self.particle_dist = particle_dist
        self.overlap_tolerance = overlap_tolerance
        self.length_unit = length_unit
        self.mass_unit = mass_unit
        self.density = density
        self.placement: PlacementStrategy = get_placement(placement)
        self._placement_name = placement

    @abstractmethod
    def generate(self) -> Aggregate:
        pass
```

- [ ] **Step 4: Run test**

Run: `pytest tests/test_generators/test_placement.py -k "base_generator" -v`
Expected: FAIL — `PCAGenerator` doesn't exist yet. We need Task 5 first. But the BaseGenerator part itself now accepts `placement`. Let's verify the base class works:

Run: `python -c "from pyFracAggregate.generators.base import BaseGenerator; print('OK')"`
Expected: prints `OK`.

- [ ] **Step 5: Commit**

```bash
git add src/pyFracAggregate/generators/base.py
git commit -m "feat: add placement parameter to BaseGenerator"
```

---

### Task 5: Create PCAGenerator (merged)

**Files:**
- Create: `src/pyFracAggregate/generators/pca.py`
- Test: `tests/test_generators/test_pca.py`

- [ ] **Step 1: Write tests for PCAGenerator with both placement strategies**

Replace `tests/test_generators/test_pca.py`:

```python
import pytest
import numpy as np
import pyFracAggregate as pfa


def test_pca_generation():
    agg = pfa.generate(n_particles=10, df=1.8, kf=1.3, method='pca')
    assert agg.current_size == 10

    positions = agg.positions
    radii = agg.radii
    overlap_tolerance = 1e-5

    for i in range(10):
        for j in range(i + 1, 10):
            dist = np.linalg.norm(positions[i] - positions[j])
            min_dist = radii[i] + radii[j] - overlap_tolerance
            assert dist >= min_dist - 1e-5


def test_pca_single_particle():
    agg = pfa.generate(n_particles=1, df=1.8, kf=1.3, method='pca')
    assert agg.current_size == 1
    assert np.allclose(agg.positions[0], [0.0, 0.0, 0.0])


def test_pca_random_placement():
    agg = pfa.generate(n_particles=30, df=1.8, kf=1.3, method='pca', placement='random')
    assert agg.current_size == 30


def test_pca_random_placement_no_overlaps():
    agg = pfa.generate(n_particles=20, df=1.8, kf=1.3, method='pca', placement='random')
    positions = agg.positions
    radii = agg.radii
    for i in range(agg.current_size):
        for j in range(i + 1, agg.current_size):
            dist = np.linalg.norm(positions[i] - positions[j])
            min_dist = radii[i] + radii[j] - 1e-5
            assert dist >= min_dist - 1e-5


def test_pca_scaling_law():
    agg = pfa.generate(n_particles=100, df=1.8, kf=1.3, method='pca')
    rg = pfa.radius_of_gyration(agg)
    a = np.mean(agg.radii)
    df_est = np.log(agg.current_size) / np.log(rg / a)
    assert abs(df_est - 1.8) < 0.5
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_generators/test_pca.py -v`
Expected: FAIL — factory still maps `pca` to `PCAFilippovGenerator` (which doesn't accept `placement` kwarg).

- [ ] **Step 3: Write PCAGenerator**

```python
# src/pyFracAggregate/generators/pca.py
import numpy as np
from pyFracAggregate.core.aggregate import Aggregate
from pyFracAggregate.generators.base import BaseGenerator


class PCAGenerator(BaseGenerator):
    """Particle-Cluster Aggregation with pluggable placement strategy."""

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

            pos = self.placement.place_particle(agg, r_N, m_N, geom_center, L, a)

            if pos is not None:
                agg.add_particle(pos[0], pos[1], pos[2], r_N, m_N)
            else:
                # Extreme fallback: attach directly to random particle surface
                idx = np.random.randint(n - 1)
                ref_pos = agg.positions[idx]
                u = np.random.normal(size=3)
                u /= np.linalg.norm(u)
                fallback_pos = ref_pos + (agg.radii[idx] + r_N - self.overlap_tolerance) * u
                agg.add_particle(fallback_pos[0], fallback_pos[1], fallback_pos[2], r_N, m_N)

        return agg
```

- [ ] **Step 4: Run test**

Run: `pytest tests/test_generators/test_pca.py -v`
Expected: FAIL — factory not yet updated. We'll fix this in Task 7, but let's verify the class works directly:

Run: `python -c "from pyFracAggregate.generators.pca import PCAGenerator; print('OK')"`
Expected: prints `OK`.

- [ ] **Step 5: Commit**

```bash
git add src/pyFracAggregate/generators/pca.py tests/test_generators/test_pca.py
git commit -m "feat: create PCAGenerator with pluggable placement strategy"
```

---

### Task 6: Create CCAGenerator (merged)

**Files:**
- Create: `src/pyFracAggregate/generators/cca.py`
- Test: `tests/test_generators/test_cca.py`

- [ ] **Step 1: Write tests for CCAGenerator**

Create `tests/test_generators/test_cca.py`:

```python
import pytest
import numpy as np
import pyFracAggregate as pfa


def test_cca_generation():
    agg = pfa.generate(n_particles=15, df=1.8, kf=1.3, method='cca')
    assert agg.current_size == 15

    positions = agg.positions
    radii = agg.radii
    overlap_tolerance = 1e-5

    for i in range(15):
        for j in range(i + 1, 15):
            dist = np.linalg.norm(positions[i] - positions[j])
            min_dist = radii[i] + radii[j] - overlap_tolerance
            assert dist >= min_dist - 1e-4


def test_cca_small_particles():
    agg = pfa.generate(n_particles=5, df=1.8, kf=1.3, method='cca')
    assert agg.current_size == 5


def test_cca_random_placement():
    agg = pfa.generate(n_particles=30, df=1.8, kf=1.3, method='cca', placement='random')
    assert agg.current_size == 30


def test_cca_random_placement_no_overlaps():
    agg = pfa.generate(n_particles=20, df=1.8, kf=1.3, method='cca', placement='random')
    positions = agg.positions
    radii = agg.radii
    for i in range(agg.current_size):
        for j in range(i + 1, agg.current_size):
            dist = np.linalg.norm(positions[i] - positions[j])
            min_dist = radii[i] + radii[j] - 1e-5
            assert dist >= min_dist - 1e-5


def test_cca_scaling_law():
    agg = pfa.generate(n_particles=100, df=1.8, kf=1.3, method='cca')
    rg = pfa.radius_of_gyration(agg)
    a = np.mean(agg.radii)
    df_est = np.log(agg.current_size) / np.log(rg / a)
    assert abs(df_est - 1.8) < 0.4
```

- [ ] **Step 2: Write CCAGenerator**

```python
# src/pyFracAggregate/generators/cca.py
import numpy as np
from pyFracAggregate.core.aggregate import Aggregate
from pyFracAggregate.generators.base import BaseGenerator
from pyFracAggregate.generators.pca import PCAGenerator
from pyFracAggregate.analysis.morphology import center_of_mass, radius_of_gyration


class CCAGenerator(BaseGenerator):
    """Cluster-Cluster Aggregation with pluggable placement strategy."""

    def generate(self) -> Aggregate:
        if self.n_particles <= 8:
            pca_gen = PCAGenerator(
                self.n_particles, self.df, self.kf, self.particle_dist,
                self.overlap_tolerance, self.length_unit, self.mass_unit,
                self.density, placement=self._placement_name
            )
            return pca_gen.generate()

        radii = self.particle_dist.sample(self.n_particles)

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

            local_pca = PCAGenerator(
                curr_size, self.df, self.kf,
                LocalDist(radii[idx:idx + curr_size]),
                self.overlap_tolerance, self.length_unit, self.mass_unit,
                self.density, placement=self._placement_name
            )
            sub_agg = local_pca.generate()
            cluster_list.append(sub_agg)
            idx += curr_size

        while len(cluster_list) > 1:
            agg1 = cluster_list.pop(0)
            agg2 = cluster_list.pop(0)
            merged = self._merge(agg1, agg2)
            cluster_list.append(merged)

        return cluster_list[0]

    def _merge(self, agg1: Aggregate, agg2: Aggregate) -> Aggregate:
        N1 = agg1.current_size
        N2 = agg2.current_size
        N = N1 + N2

        com1 = center_of_mass(agg1)
        com2 = center_of_mass(agg2)
        Rg1 = radius_of_gyration(agg1)
        Rg2 = radius_of_gyration(agg2)

        a = (np.mean(agg1.radii) * N1 + np.mean(agg2.radii) * N2) / N

        term1 = (a**2 * N**2) / (N1 * N2) * (N / self.kf) ** (2.0 / self.df)
        term2 = (N / N2) * Rg1**2
        term3 = (N / N1) * Rg2**2

        Gamma_sq = term1 - term2 - term3
        Gamma = np.sqrt(max(Gamma_sq, 0.0))

        pos1 = agg1.positions - com1
        r1 = agg1.radii
        pos2_centered = agg2.positions - com2
        r2 = agg2.radii

        pos2_final = self.placement.merge_clusters(
            pos1, r1, agg1, pos2_centered, r2, agg2, Gamma, a
        )

        if pos2_final is None:
            pos2_final = pos2_centered

        merged = Aggregate(N, self.length_unit, self.mass_unit, self.density)
        for i in range(N1):
            merged.add_particle(
                pos1[i, 0], pos1[i, 1], pos1[i, 2],
                agg1.radii[i], agg1.masses[i]
            )
        for j in range(N2):
            merged.add_particle(
                pos2_final[j, 0], pos2_final[j, 1], pos2_final[j, 2],
                agg2.radii[j], agg2.masses[j]
            )
        return merged
```

- [ ] **Step 3: Commit**

```bash
git add src/pyFracAggregate/generators/cca.py tests/test_generators/test_cca.py
git commit -m "feat: create CCAGenerator with pluggable placement strategy"
```

---

### Task 7: Update factory and top-level API

**Files:**
- Modify: `src/pyFracAggregate/generators/factory.py`
- Modify: `src/pyFracAggregate/__init__.py`
- Modify: `tests/test_generators/test_factory.py`

- [ ] **Step 1: Write failing tests for new factory behavior**

Replace `tests/test_generators/test_factory.py`:

```python
import pytest
import pyFracAggregate as pfa


@pytest.mark.parametrize("method", ['pca', 'cca', 'fracval', 'tdcca'])
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


def test_removed_flage_pca_raises_helpful_error():
    with pytest.raises(ValueError, match="flage_pca.*has been removed"):
        pfa.generate(n_particles=10, df=1.8, kf=1.3, method='flage_pca')


def test_removed_flage_cca_raises_helpful_error():
    with pytest.raises(ValueError, match="flage_cca.*has been removed"):
        pfa.generate(n_particles=10, df=1.8, kf=1.3, method='flage_cca')


def test_placement_param_forwarded():
    """placement kwarg should be forwarded to the generator."""
    agg = pfa.generate(
        n_particles=20, df=1.8, kf=1.3,
        method='pca', placement='random'
    )
    assert agg.current_size == 20
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_generators/test_factory.py -v`
Expected: FAIL — factory still has old entries.

- [ ] **Step 3: Update factory**

Replace `src/pyFracAggregate/generators/factory.py`:

```python
from pyFracAggregate.core.distributions import ParticleDistribution
from pyFracAggregate.generators.base import BaseGenerator
from pyFracAggregate.generators.pca import PCAGenerator
from pyFracAggregate.generators.cca import CCAGenerator
from pyFracAggregate.generators.fracval import FracVALGenerator
from pyFracAggregate.generators.tdcca import ThouyJullienGenerator


def get_generator(
    method: str,
    n_particles: int,
    df: float,
    kf: float,
    particle_dist: ParticleDistribution,
    overlap_tolerance: float = 0.0,
    **kwargs
) -> BaseGenerator:
    method = method.lower()

    if method == 'pca':
        return PCAGenerator(n_particles, df, kf, particle_dist, overlap_tolerance, **kwargs)
    elif method == 'cca':
        return CCAGenerator(n_particles, df, kf, particle_dist, overlap_tolerance, **kwargs)
    elif method == 'fracval':
        return FracVALGenerator(n_particles, df, kf, particle_dist, overlap_tolerance, **kwargs)
    elif method == 'tdcca':
        return ThouyJullienGenerator(n_particles, df, kf, particle_dist, overlap_tolerance, **kwargs)
    elif method in ('flage_pca', 'flage_cca'):
        raise ValueError(
            f"method='{method}' has been removed. Use method='{'pca' if method == 'flage_pca' else 'cca'}' "
            "(FLAGE is now the default placement strategy). "
            f"For the old random sampling behavior, use method='{'pca' if method == 'flage_pca' else 'cca'}', placement='random'."
        )
    else:
        raise ValueError(f"Unknown generation method: {method}")
```

- [ ] **Step 4: Update top-level `generate()` signature**

In `src/pyFracAggregate/__init__.py`, update the `generate` function:

```python
def generate(
    n_particles: int,
    df: float,
    kf: float,
    method: str = 'pca',
    particle_dist = None,
    overlap_tolerance: float = 1e-5,
    placement: str = 'algebraic',
    **kwargs
) -> Aggregate:
    """
    High-level API to generate a fractal aggregate.

    Args:
        n_particles (int): Target number of particles.
        df (float): Fractal dimension (typically 1.5 - 2.5).
        kf (float): Fractal prefactor (typically 1.0 - 2.0).
        method (str): Algorithm to use ('pca', 'cca', 'fracval', 'tdcca').
        particle_dist: Particle radius distribution (defaults to Monodisperse(1.0)).
        overlap_tolerance (float): Allowed overlap between spheres.
        placement (str): Placement strategy ('algebraic' (default) or 'random').
    """
    if particle_dist is None:
        particle_dist = Monodisperse(1.0)

    generator = get_generator(
        method=method,
        n_particles=n_particles,
        df=df,
        kf=kf,
        particle_dist=particle_dist,
        overlap_tolerance=overlap_tolerance,
        placement=placement,
        **kwargs
    )

    return generator.generate()
```

Also update `__all__` in the same file:

```python
__all__ = [
    "generate",
    "analyze",
    "Aggregate",
    "Monodisperse",
    "LognormalDistribution",
    "PCAGenerator",
    "CCAGenerator",
    "FracVALGenerator",
    "ThouyJullienGenerator",
    "AlgebraicPlacement",
    "RandomPlacement",
    "radius_of_gyration",
    "center_of_mass",
    "pair_correlation_function",
    "estimate_fractal_dimension",
    "plot_pair_correlation",
    "export_glb",
    "export_3mf",
    "export_vtm",
    "export_vtk",
    "export_to_json"
]
```

And add the imports at the top:

```python
from pyFracAggregate.generators.pca import PCAGenerator
from pyFracAggregate.generators.cca import CCAGenerator
from pyFracAggregate.generators.fracval import FracVALGenerator
from pyFracAggregate.generators.tdcca import ThouyJullienGenerator
from pyFracAggregate.generators.placement.algebraic import AlgebraicPlacement
from pyFracAggregate.generators.placement.random_ import RandomPlacement
```

- [ ] **Step 5: Run factory tests**

Run: `pytest tests/test_generators/test_factory.py -v`
Expected: All PASS.

- [ ] **Step 6: Commit**

```bash
git add src/pyFracAggregate/generators/factory.py \
        src/pyFracAggregate/__init__.py \
        tests/test_generators/test_factory.py
git commit -m "feat: update factory and API for placement strategy (6 methods -> 4)"
```

---

### Task 8: Rename FracVAL and tdCCA generators

**Files:**
- Create: `src/pyFracAggregate/generators/fracval.py` (rename from `cca_fracval.py`)
- Create: `src/pyFracAggregate/generators/tdcca.py` (rename from `tdcca_thouy.py`)
- Modify: `src/pyFracAggregate/generators/fracval.py` (update imports)
- Test: existing `test_cca_fracval.py` and `test_tdcca_thouy.py`

- [ ] **Step 1: Rename files**

```bash
git mv src/pyFracAggregate/generators/cca_fracval.py src/pyFracAggregate/generators/fracval.py
git mv src/pyFracAggregate/generators/tdcca_thouy.py src/pyFracAggregate/generators/tdcca.py
```

- [ ] **Step 2: Fix imports in fracval.py**

In `src/pyFracAggregate/generators/fracval.py`, change line 11:
```python
# Old:
from pyFracAggregate.generators.pca_flage import PCAFlageGenerator
# New:
from pyFracAggregate.generators.pca import PCAGenerator
```

And in `generate()` method (line 24 and line 51), change:
```python
# Old: PCAFlageGenerator(...)
# New: PCAGenerator(...)
```

- [ ] **Step 3: Run fracval and tdcca tests**

Run: `pytest tests/test_generators/test_cca_fracval.py tests/test_generators/test_tdcca_thouy.py -v`
Expected: All PASS.

- [ ] **Step 4: Commit**

```bash
git add src/pyFracAggregate/generators/fracval.py src/pyFracAggregate/generators/tdcca.py
git commit -m "refactor: rename cca_fracval -> fracval, tdcca_thouy -> tdcca"
```

---

### Task 9: Clean up old files and update optimizer_flage tests

**Files:**
- Delete: `src/pyFracAggregate/generators/pca_filippov.py`
- Delete: `src/pyFracAggregate/generators/pca_flage.py`
- Delete: `src/pyFracAggregate/generators/cca_filippov.py`
- Delete: `src/pyFracAggregate/generators/cca_flage.py`
- Delete: `tests/test_generators/test_pca_flage.py`
- Delete: `tests/test_generators/test_cca_flage.py`
- Delete: `tests/test_generators/test_cca_filippov.py`
- Modify: `tests/test_generators/test_optimizer_flage.py` (keep, optimizer still used by algebraic placement)
- Modify: `tests/test_generators/test_benchmark.py`

- [ ] **Step 1: Delete old generator files**

```bash
git rm src/pyFracAggregate/generators/pca_filippov.py \
       src/pyFracAggregate/generators/pca_flage.py \
       src/pyFracAggregate/generators/cca_filippov.py \
       src/pyFracAggregate/generators/cca_flage.py
```

- [ ] **Step 2: Delete old test files**

```bash
git rm tests/test_generators/test_pca_flage.py \
       tests/test_generators/test_cca_flage.py \
       tests/test_generators/test_cca_filippov.py
```

- [ ] **Step 3: Update benchmark tests**

Replace `tests/test_generators/test_benchmark.py`:

```python
import time
import pytest
import numpy as np
import pyFracAggregate as pfa


@pytest.mark.benchmark
def test_pca_default_is_fast():
    """PCA with default algebraic placement should complete in reasonable time."""
    n = 200
    df, kf = 1.8, 1.3

    np.random.seed(42)
    t0 = time.perf_counter()
    agg = pfa.generate(n_particles=n, df=df, kf=kf, method='pca')
    elapsed = time.perf_counter() - t0

    assert agg.current_size == n
    assert elapsed < 30.0  # Should be fast with algebraic placement


@pytest.mark.benchmark
def test_cca_default_is_fast():
    """CCA with default algebraic placement should complete in reasonable time."""
    n = 100
    df, kf = 1.8, 1.3

    np.random.seed(42)
    t0 = time.perf_counter()
    agg = pfa.generate(n_particles=n, df=df, kf=kf, method='cca')
    elapsed = time.perf_counter() - t0

    assert agg.current_size == n
    assert elapsed < 30.0
```

- [ ] **Step 4: Update optimizer_flage test imports (if needed)**

The `test_optimizer_flage.py` imports from `pyFracAggregate.generators.optimizer_flage` which still exists and is still used by `placement/algebraic.py`. No changes needed unless we later remove `optimizer_flage.py` (we won't in this refactor — it's an internal utility).

- [ ] **Step 5: Run all tests**

Run: `pytest tests/ -v --ignore=tests/test_io/`
Expected: All PASS.

- [ ] **Step 6: Commit**

```bash
git add -A
git commit -m "refactor: remove old generator files and update tests"
```

---

### Task 10: Update generators __init__.py and run full test suite

**Files:**
- Modify: `src/pyFracAggregate/generators/__init__.py`
- Modify: `docs/superpowers/specs/2026-04-09-placement-strategy-refactor-design.md` (update status)

- [ ] **Step 1: Update generators __init__.py**

```python
# src/pyFracAggregate/generators/__init__.py
"""Generators for fractal aggregates."""

from pyFracAggregate.generators.pca import PCAGenerator
from pyFracAggregate.generators.cca import CCAGenerator
from pyFracAggregate.generators.fracval import FracVALGenerator
from pyFracAggregate.generators.tdcca import ThouyJullienGenerator
from pyFracAggregate.generators.placement.base import PlacementStrategy
from pyFracAggregate.generators.placement.algebraic import AlgebraicPlacement
from pyFracAggregate.generators.placement.random_ import RandomPlacement

__all__ = [
    "BaseGenerator",
    "PCAGenerator",
    "CCAGenerator",
    "FracVALGenerator",
    "ThouyJullienGenerator",
    "PlacementStrategy",
    "AlgebraicPlacement",
    "RandomPlacement",
]
```

- [ ] **Step 2: Run full test suite**

Run: `pytest tests/ -v`
Expected: All PASS.

- [ ] **Step 3: Commit**

```bash
git add src/pyFracAggregate/generators/__init__.py
git commit -m "chore: update generators package exports"
```
