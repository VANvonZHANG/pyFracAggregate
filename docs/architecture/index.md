# Architecture

pyFracAggregate is a four-layer library: a **core** data structure
(`Aggregate`, primary-particle distributions), a **generators** layer holding
the four aggregation algorithms and their shared placement sublayer, an
**analysis** layer computing morphological descriptors, and an **io** layer
exporting aggregates. A thin top-level facade (`pfa.generate` /
`pfa.analyze`) ties the layers together, and a factory function dispatches
generation requests to one of the four algorithm classes.

```{mermaid}
flowchart TB
    user(["user code"])

    facade["top-level facade<br/>generate() / analyze()"]
    factory["generators/factory.py<br/>get_generator(method, ...)"]

    subgraph GENLAYER ["generators/"]
        BASE["BaseGenerator (ABC)"]
        PCA["PCAGenerator"]
        CCA["CCAGenerator"]
        FRACVAL["FracVALGenerator"]
        TJ["ThouyJullienGenerator"]
        subgraph PLACE ["generators/placement/"]
            PABC["PlacementStrategy (ABC)"]
            ALG["AlgebraicPlacement<br/>(FLAGE, default)"]
            RND["RandomPlacement<br/>(Monte Carlo)"]
            MC["_helpers.py<br/>shared Monte Carlo"]
        end
    end

    subgraph CORELAYER ["core/"]
        AGG["Aggregate<br/>pre-allocated (max_particles, 5)<br/>[x, y, z, radius, mass]"]
        DIST["ParticleDistribution<br/>Monodisperse / LognormalDistribution"]
    end

    subgraph ANALYSIS ["analysis/"]
        MORPH["morphology.py<br/>radius_of_gyration / center_of_mass"]
        CORR["correlation.py<br/>pair correlation / Df estimation"]
    end

    subgraph IOLAYER ["io/"]
        YAMLIO["data.py<br/>export_yaml"]
        VTKIO["vtk.py<br/>export_vtk / export_vtm"]
        VISIO["visualization.py<br/>export_render / export_rotation_video"]
    end

    user --> facade
    facade -->|"'method' keyword"| factory
    factory --> PCA
    factory --> CCA
    factory --> FRACVAL
    factory --> TJ

    PCA & CCA & FRACVAL & TJ -.->|"subclass"| BASE
    DIST -->|"sample() radii"| BASE
    PCA -.->|"place_particle()"| PABC
    CCA -.->|"merge_clusters()"| PABC
    PABC --> ALG
    PABC --> RND
    ALG --> MC
    RND --> MC

    PCA & CCA & FRACVAL & TJ -->|"generate() returns"| AGG

    AGG --> MORPH
    AGG --> CORR
    facade -.->|"analyze()"| MORPH
    facade -.->|"analyze()"| CORR

    AGG --> YAMLIO
    AGG --> VTKIO
    AGG --> VISIO
```

Only `pca` and `cca` route through the placement sublayer; `fracval` and
`tdcca` embed their own contact logic (see below).

## The `Aggregate` data structure

Every layer produces or consumes one central type,
[`Aggregate`](/api-reference/index.md#core). It stores all particle data in a
**single pre-allocated NumPy array** of shape `(max_particles, 5)`, one row
per particle holding `[x, y, z, radius, mass]`:

- **Why pre-allocation.** The array is allocated once, contiguously, at
  construction. Growing the cluster is a row write plus a counter increment
  (`add_particle` is O(1)); there are no per-particle Python objects and no
  list reallocations or copies during generation. Data locality keeps the
  tight numeric loops in the generators and analysis fast.
- **Zero-copy views.** The properties `positions` (`(N, 3)`), `radii`
  (`(N,)`), and `masses` (`(N,)`) return NumPy *views* slicing that one
  backing array, not copies — mutating them mutates the aggregate.
  `to_numpy()` returns a copy of the valid `(N, 5)` block when ownership of
  the data must leave the aggregate.
- **`.current_size` semantics.** The backing array is over-provisioned to
  `max_particles` rows; `.current_size` counts how many are valid. The views
  always expose exactly the valid prefix, so `.current_size` is "the N of
  the cluster" (the analysis helper `pfa.analyze` returns it as `"N"`). The
  capacity is readable as `.max_size`.
- **Units.** `length_unit`, `mass_unit`, and `density` travel with the
  aggregate (set by the generator; defaults `'nm'`, `'g'`, `1.0`) so exports
  and analysis can label quantities.

Primary-particle sizes are supplied by a `ParticleDistribution` —
`Monodisperse(radius)` or `LognormalDistribution(mean, std)` — whose
`sample(n)` the generators call to draw radii.

## The generator contract

All four algorithms implement the abstract
[`BaseGenerator`](/api-reference/index.md#generators) with a single
constructor signature:

```python
BaseGenerator(
    n_particles,        # target number of primary particles
    df, kf,             # fractal dimension and prefactor
    particle_dist,      # ParticleDistribution for primary radii
    overlap_tolerance=0.0,
    placement='algebraic',
    # plus length_unit / mass_unit / density
)
```

`generate()` then returns a populated `Aggregate`. Because the contract is
identical, the factory `get_generator(method, ...)` (in
`generators/factory.py`) can dispatch on a string — `'pca'`, `'cca'`,
`'fracval'`, `'tdcca'` — and that factory is exactly what the top-level
`pfa.generate()` wraps. Keyword-only extras (`surface_beta`) are validated
there and rejected for methods that do not support them.

`pfa.analyze()` is the facade's read-side: it bundles
`radius_of_gyration`, `center_of_mass`, and the pair-correlation fit into
one summary dict (`Rg`, `CoM`, `N`, `Df_estimated`, `R2`).

## The placement strategy layer

The scaling law fixes *where* the center of each added particle or cluster
must sit (distance `L` or `Γ` from the cluster center) but not *which*
particles touch. Resolving that contact problem is delegated to a strategy
object, selected by name:

- [`PlacementStrategy`](/api-reference/index.md#placement) — the ABC. Its two
  abstract methods mirror the two aggregation stages: `place_particle()`
  (single particle onto a cluster, the PCA stage) and `merge_clusters()`
  (two clusters onto a common `Γ`, the CCA stage).
- `AlgebraicPlacement` (default) — FLAGE, Skorupski et al. (2014). The
  analytical solver in `generators/optimizer_flage.py` intersects the target
  sphere with a reference particle's contact sphere to get exact touching
  points; candidates are overlap-filtered, with a Monte Carlo fallback.
- `RandomPlacement` — Filippov et al. (2000). Pure Monte Carlo sampling on
  the target sphere with gradual tolerance relaxation until a candidate is
  accepted (typically several times slower).
- `get_placement(name)` — the factory. `BaseGenerator.__init__` resolves the
  `placement` string through it and stores the strategy instance, injecting
  the generator's `overlap_tolerance` into it.

Both implementations share their Monte Carlo machinery
(`random_monte_carlo_place`, `random_monte_carlo_merge`) through
`generators/placement/_helpers.py`, so the strategies differ only in their
deterministic precomputation, not in their fallbacks.

`FracVALGenerator` and `ThouyJullienGenerator` do **not** route through the
placement layer: FracVAL's merge has its own deterministic contact search
(sphere-sphere intersection with overlap-resolving rotations), and Thouy &
Jullien selects among lattice-seeded orientations directly. The `placement`
argument is accepted for constructor uniformity but ignored by both.

## Analysis

The analysis layer is a set of pure functions over an `Aggregate`:
`morphology.py` provides `radius_of_gyration` (mass-weighted, including each
sphere's intrinsic gyration, per Morán et al. 2019 Eq. (3)) and
`center_of_mass`; `correlation.py` provides `pair_correlation_function`,
`estimate_fractal_dimension` (log-log regression of `C(r)` over the fractal
regime), and the matplotlib-based `plot_pair_correlation` for diagnosing the
fit. No function mutates the aggregate.

## I/O

The io layer serializes an `Aggregate` for downstream use: `data.py` writes
a YAML snapshot bundling particle data with generation parameters and
analysis results; `vtk.py` builds the pyvista point cloud (`export_vtk`) and
MultiBlock dataset (`export_vtm`) for ParaView; `visualization.py` performs
off-screen pyvista rendering (`export_render`) and assembles MP4 rotation
videos (`export_rotation_video`). Rendering exporters need a working 3D
backend — see the [io guide](/user-guide/io.md) for headless-environment
notes.

## Tests

The test suite mirrors the source layout:

- `tests/test_core/` — `Aggregate`, distributions, 3D math helpers
- `tests/test_generators/` — the four algorithms, factory, placement
  strategies, FLAGE optimizer
- `tests/test_analysis/` — morphology and correlation functions
- `tests/test_io/` — YAML and visualization exports

Slow performance tests carry the `benchmark` pytest marker and can be
deselected with `-m "not benchmark"` (see
[Contributing](/contributing.md#running-the-tests)).
