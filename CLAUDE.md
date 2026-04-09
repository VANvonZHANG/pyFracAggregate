# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build & Development Commands

```bash
pip install -e ".[dev]"        # Editable install with dev tools (pytest, ruff, mypy)
pip install -e ".[science]"    # Include pyvista for VTK visualization
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
- **ParticleDistribution**: ABC with `Monodisperse(radius)` and `LognormalDistribution(mean, std)` implementations. Generators call `sample(n)` to get particle sizes.

### Generators (`src/pyFracAggregate/generators/`)
- **BaseGenerator**: ABC accepting `(n_particles, df, kf, particle_dist, overlap_tolerance, placement='algebraic')`. All generators produce an `Aggregate` via `generate()`.
- **Factory**: `get_generator(method, ...)` dispatches to:
  - `'pca'` → `PCAGenerator` — particle-cluster aggregation
  - `'cca'` → `CCAGenerator` — cluster-cluster aggregation
  - `'fracval'` → `FracVALGenerator` — FracVAL algorithm
  - `'tdcca'` → `ThouyJullienGenerator` — Thouy & Jullien (2004) algorithm
- All generators share the same constructor signature from `BaseGenerator`.

#### Placement Strategy Layer (`src/pyFracAggregate/generators/placement/`)
- **PlacementStrategy** ABC with `place_particle()` (PCA stage) and `merge_clusters()` (CCA stage).
- **AlgebraicPlacement** (FLAGE, Skorupski et al., 2014): Algebraic touching-point computation with random Monte Carlo fallback. Default strategy.
- **RandomPlacement** (Filippov et al., 2000): Pure random Monte Carlo sampling with tolerance relaxation.
- **`get_placement(name)`** factory returns a `PlacementStrategy` by name (`'algebraic'` or `'random'`).
- **`_helpers.py`**: Shared Monte Carlo helpers (`random_monte_carlo_place`, `random_monte_carlo_merge`) used by both strategies.

### Analysis (`src/pyFracAggregate/analysis/`)
- `morphology`: `radius_of_gyration()`, `center_of_mass()`
- `correlation`: `pair_correlation_function()`, `estimate_fractal_dimension()`, `plot_pair_correlation()`

### IO (`src/pyFracAggregate/io/`)
- **mesh.py**: `export_glb()` (glTF binary) and `export_3mf()` via trimesh
- **vtk.py**: `export_vtm()` and `export_vtk()` via pyvista (optional dependency)
- **data.py**: `export_to_json()` for structured data output

### Top-level API (`__init__.py`)
`pfa.generate(n_particles, df, kf, method, placement='algebraic')` and `pfa.analyze(aggregate)` are the primary entry points. `generate()` delegates to the factory. The `placement` parameter selects the placement strategy (`'algebraic'` or `'random'`).

## Key Conventions
- Python 3.9+, type hints throughout
- Physical units (`length_unit`, `mass_unit`, `density`) are part of `Aggregate` and `BaseGenerator`
- `mathutils` package used for 3D math (with scipy fallback)
- Tests live in `tests/` mirroring the source layout (`test_core/`, `test_generators/`, `test_analysis/`, `test_io/`)
