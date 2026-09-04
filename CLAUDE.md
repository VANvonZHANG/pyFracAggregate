# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build & Development Commands

```bash
pip install -e ".[dev]"        # Editable install with dev tools (pytest, ruff, mypy)
pip install -e ".[plot]"       # Include matplotlib for plotting
pytest                        # Run all tests
pytest tests/test_core/test_aggregate.py  # Run a single test file
pytest -k "test_pca"          # Run tests matching a name pattern
ruff check src/               # Lint source code
mypy src/                     # Static type checking
```

No CLI entry point exists — the package is a library used via `import pyFracAggregate as pfa`.

## Architecture

**Four-layer design**: core → generators → analysis → io

### Core (`src/pyFracAggregate/core/`)
- **Aggregate**: Central data structure. Pre-allocated `(max_particles, 5)` NumPy array storing `[x, y, z, radius, mass]`. Properties `positions`, `radii`, `masses` return zero-copy views into the backing array.
- **ParticleDistribution**: ABC with `Monodisperse(radius)`, `LognormalDistribution(mean, std)` and `FixedRadii(radii)` implementations. Generators call `sample(n, rng=None)` to get particle sizes.

### Generators (`src/pyFracAggregate/generators/`)
- **BaseGenerator**: ABC accepting `(n_particles, df, kf, particle_dist, overlap_tolerance=1e-5, placement='solved', scaling=None, seed=None, surface_beta=None, rng=None)`. Holds the seeded `np.random.Generator` (`self.rng`); all generators produce an `Aggregate` via `generate()`.
- **Factory**: `get_generator(method, ...)` validates the legality matrix (10 legal `method`×`scaling`×`placement` cells; `pca×constructed` raises) then dispatches:
  - `'pca'` → `PCAGenerator` — particle-cluster aggregation
  - `'cca'` → `CCAGenerator` — cluster-cluster aggregation
  - `'fracval'` → deprecated alias for `(cca, mass, constructed)`; `'tdcca'` removed in v0.4

#### Scaling laws (`src/pyFracAggregate/core/scaling.py`)
- **ScalingLaw** ABC: `weights()`, `char_radius()`, `target_rg_sq()`, `pca_step()`, `cca_gamma()`.
- **CountScaling** (Filippov 2000 count weights) / **MassScaling** (Morán 2019 mass weights; default). `get_scaling(name_or_law, df, kf)` accepts names or instances. Monodisperse: the two are mathematically equivalent.

#### Placement Strategy Layer (`src/pyFracAggregate/generators/placement/`)
- **PlacementStrategy** ABC with `place_particle()` (PCA stage) and `merge_clusters()` (CCA stage).
- **SolvedPlacement** (FLAGE, Skorupski et al., 2014): closed-form tangency solving with Monte Carlo fallback. Default.
- **SampledPlacement** (Filippov et al., 2000): pure Monte Carlo sampling with tolerance relaxation.
- **ConstructedPlacement** (Morán et al., 2019): specified contact pair + attitude construction + COM check (merge-only; the old FracVAL merge logic lives here).
- **`get_placement(name_or_strategy, ...)`** factory returns a `PlacementStrategy` by name (`'sampled'`/`'solved'`/`'constructed'`; deprecated aliases `'algebraic'`→solved, `'random'`→sampled) or passes instances through.
- **`solvers.py`**: shared contact primitives (`solve_tangency`, `filter_overlapping_candidates`, `build_particle_list_pca`, `mc_touch_place`, `mc_touch_merge`) used by all strategies.

### Analysis (`src/pyFracAggregate/analysis/`)
- `morphology`: `radius_of_gyration()`, `center_of_mass()`
- `correlation`: `pair_correlation_function()`, `estimate_fractal_dimension()`, `plot_pair_correlation()`

### IO (`src/pyFracAggregate/io/`)
- **data.py**: `export_yaml()` — full aggregate snapshot (data, generation params, analysis results)
- **visualization.py**: `save_screenshot()` (PNG) and `save_rotation_video()` (MP4) via off-screen pyvista; `color_by="radius"` maps monomer size to a colormap
- **vtk.py**: `export_vtm()` (MultiBlock) and `export_vtk()` (point cloud) via pyvista

### Top-level API (`__init__.py`)
`pfa.generate(n_particles, df, kf, method='pca', scaling='mass', placement='solved', particle_dist=None, overlap_tolerance=1e-5, seed=None)` and `pfa.analyze(aggregate)` (returns the `MorphologyReport` dataclass) are the primary entry points. `generate()` delegates to the factory after matrix validation. Generation is reproducible via `seed=` (private `np.random.Generator`; the global NumPy RNG is never consulted). IO exports include `export_yaml()` (full snapshot; accepts a `MorphologyReport` and serializes legacy key names), `export_vtk()` (point cloud), and `export_vtm()` (MultiBlock).

## Key Conventions
- Python 3.13+, type hints throughout; `ruff check src/` and `mypy src/` clean
- Physical units (`length_unit`, `mass_unit`, `density`) are part of `Aggregate` and `BaseGenerator`
- `mathutils` package used for 3D math (with scipy fallback)
- Tests live in `tests/` mirroring the source layout (`test_core/`, `test_generators/`, `test_analysis/`, `test_io/`)
