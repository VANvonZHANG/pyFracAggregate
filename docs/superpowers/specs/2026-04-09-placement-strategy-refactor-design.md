# Placement Strategy Refactor Design

## Problem

FLAGE (Skorupski et al., 2014) is a speed optimization technique that uses algebraic sphere-sphere intersection to compute exact touching points, replacing random Monte Carlo sampling. It is currently exposed as four separate generator methods (`flage_pca`, `flage_cca`) alongside the original Filippov methods (`pca`, `cca`). Since FLAGE produces the same type of fractal aggregate as Filippov (just faster), exposing it as a separate algorithm creates unnecessary cognitive burden for users.

## Goal

Refactor FLAGE from a top-level algorithm choice into a pluggable placement strategy. Users see 4 methods instead of 6. Advanced users can opt into the slower random placement via a `placement` parameter.

## Design

### 1. Placement Strategy Layer

New module `src/pyFracAggregate/generators/placement/`:

```
generators/placement/
├── __init__.py       # exports + get_placement() factory
├── base.py           # PlacementStrategy ABC
├── algebraic.py      # AlgebraicPlacement (FLAGE)
└── random.py         # RandomPlacement (Filippov)
```

**`PlacementStrategy` ABC** — two interfaces:

```python
class PlacementStrategy(ABC):
    @abstractmethod
    def place_particle(self, agg, candidate_radius, geom_center, L, mean_radius):
        """PCA stage: place a single particle onto the Gamma sphere.
        Returns (x, y, z) or None."""

    @abstractmethod
    def merge_clusters(self, pos1, r1, agg1, pos2_centered, r2, agg2, Gamma, mean_radius):
        """CCA stage: merge two sub-clusters.
        Returns pos2_final array or None."""
```

**`AlgebraicPlacement`** — migrated from `optimizer_flage.py` + `cca_flage.py`:
- `place_particle`: calls `build_particle_list_pca` -> `find_exact_touching_points_pca` -> `filter_overlapping_candidates`, with random sampling fallback
- `merge_clusters`: surface particle filtering + rotation placement, with random fallback

**`RandomPlacement`** — migrated from `pca_filippov.py` + `cca_filippov.py`:
- `place_particle`: random sampling on Gamma sphere with gradual tolerance relaxation
- `merge_clusters`: random rotation + collision detection + minimum gap tracking

**Factory**: `get_placement(name)` where `'algebraic'` (default) -> `AlgebraicPlacement`, `'random'` -> `RandomPlacement`.

### 2. Generator Consolidation

New file structure:

```
generators/
├── __init__.py
├── base.py              # BaseGenerator — add placement parameter
├── factory.py           # get_generator — remove flage_pca/flage_cca entries
├── pca.py               # merged from pca_filippov.py + pca_flage.py
├── cca.py               # merged from cca_filippov.py + cca_flippov.py
├── fracval.py           # renamed from cca_fracval.py
├── tdcca.py             # renamed from tdcca_thouy.py
└── placement/           # strategy modules from section 1
```

**`BaseGenerator` change**: new `placement` kwarg (default `'algebraic'`), resolved via `get_placement()`.

**`PCAGenerator` (pca.py)**: single class, delegates particle placement to `self.placement.place_particle()`. Extreme fallback (attach to random particle surface) stays in the generator.

**`CCAGenerator` (cca.py)**: single class with two phases:
- Phase 1 (sub-clusters): uses `PCAGenerator(..., placement=self.placement)`
- Phase 2 (merging): delegates to `self.placement.merge_clusters()`

**`FracVALGenerator` (fracval.py)**: imports `PCAGenerator` instead of `PCAFlageGenerator`. Its own three-stage merge logic (contact matrix, alignment rotation, CM2 constraint) stays internal — not routed through placement strategy, as this is unique to Moran 2019.

**`ThouyJullienGenerator` (tdcca.py)**: unchanged logic. Uses lattice hierarchy, unrelated to FLAGE/Filippov.

**Deleted files**: `pca_filippov.py`, `pca_flage.py`, `cca_filippov.py`, `cca_flage.py`, `optimizer_flage.py`.

### 3. User API Changes

**`generate()` signature**:

```python
def generate(n_particles, df, kf, method='pca', particle_dist=None,
             overlap_tolerance=0.0, placement='algebraic', ...):
```

**Factory mapping** (6 -> 4):

| Old method      | New method  | Notes                                |
|-----------------|-------------|--------------------------------------|
| `'pca'`         | `'pca'`     | kept, defaults to algebraic          |
| `'flage_pca'`   | removed     | see migration error                  |
| `'cca'`         | `'cca'`     | kept, defaults to algebraic          |
| `'flage_cca'`   | removed     | see migration error                  |
| `'fracval'`     | `'fracval'` | kept, ignores placement parameter    |
| `'tdcca'`       | `'tdcca'`   | kept, ignores placement parameter    |

**Migration error** for removed methods:

```python
raise ValueError(
    "method='flage_pca' has been removed. Use method='pca' "
    "(FLAGE is now the default placement strategy). "
    "For the old random sampling behavior, use method='pca', placement='random'."
)
```

**`__init__.py` exports**:
- Remove: `PCAFilippovGenerator`, `PCAFlageGenerator`, `CCAFilippovGenerator`, `CCAFlageGenerator`
- Add: `PCAGenerator`, `CCAGenerator`, `AlgebraicPlacement`, `RandomPlacement`
- Keep: `FracVALGenerator`, `ThouyJullienGenerator`

### 4. Test Changes

- All `method='flage_pca'` / `method='flage_cca'` tests -> `method='pca', placement='algebraic'`
- New tests for `placement='random'` to verify fallback path
- Factory tests: remove flage entries, add error test for invalid method names
- Placement unit tests: test `AlgebraicPlacement` and `RandomPlacement` independently

## Out of Scope

- FracVAL merge logic is not abstracted into a placement strategy (unique deterministic algorithm)
- tdCCA lattice logic is not abstracted (fundamentally different approach)
- No performance benchmarking in this refactor (purely structural)
