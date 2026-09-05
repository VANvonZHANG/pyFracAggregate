# Architecture

pyFracAggregate is a four-layer library: a **core** data structure
(`Aggregate`, primary-particle distributions, scaling laws), a **generators**
layer holding the two aggregation algorithms, the shared scaling-law
strategy, and their placement sublayer, an **analysis** layer computing
morphological descriptors, and an **io** layer exporting aggregates. A thin
top-level facade (`pfa.generate` / `pfa.analyze`) ties the layers together;
a factory validates the three-axis coordinate
(`method` × `scaling` × `placement`) and dispatches to a generator class.
Every generator draws randomness from one seeded `numpy.random.Generator`,
so any legal coordinate is reproducible with `seed=`.

```{mermaid}
flowchart TB
    user(["user code"])

    facade["top-level facade<br/>generate() / analyze()"]
    factory["generators/factory.py<br/>get_generator: matrix validation<br/>method x scaling x placement"]

    subgraph GENLAYER ["generators/"]
        BASE["BaseGenerator (ABC)<br/>holds seeded np.random.Generator"]
        PCA["PCAGenerator"]
        CCA["CCAGenerator"]
        subgraph PLACE ["generators/placement/"]
            PABC["PlacementStrategy (ABC)"]
            SOLVED["SolvedPlacement<br/>(FLAGE, default)"]
            SMP["SampledPlacement<br/>(Monte Carlo)"]
            CONS["ConstructedPlacement<br/>(FracVAL contact pairs)"]
            SOLV["solvers.py<br/>closed-form + MC primitives"]
        end
    end

    subgraph CORELAYER ["core/"]
        AGG["Aggregate<br/>pre-allocated (max_particles, 5)<br/>[x, y, z, radius, mass]"]
        DIST["ParticleDistribution<br/>Monodisperse / Lognormal / FixedRadii"]
        SCALE["core/scaling.py<br/>ScalingLaw: Count | Mass"]
    end

    subgraph ANALYSIS ["analysis/"]
        MORPH["morphology.py<br/>radius_of_gyration / center_of_mass"]
        CORR["correlation.py<br/>pair correlation / Df estimation"]
    end

    subgraph IOLAYER ["io/"]
        YAMLIO["data.py<br/>export_yaml"]
        VTKIO["vtk.py<br/>export_vtk / export_vtm"]
        VISIO["visualization.py<br/>save_screenshot / save_rotation_video"]
    end

    user --> facade
    facade -->|"coordinate + seed"| factory
    factory --> PCA
    factory --> CCA

    PCA & CCA -.->|"subclass"| BASE
    DIST -->|"sample(n, rng) radii"| BASE
    SCALE -->|"pca_step / cca_gamma"| BASE
    PCA -.->|"place_particle()"| PABC
    CCA -.->|"merge_clusters()"| PABC
    PABC --> SOLVED
    PABC --> SMP
    PABC --> CONS
    SOLVED --> SOLV
    SMP --> SOLV
    CONS --> SOLV

    PCA & CCA -->|"generate() returns"| AGG

    AGG --> MORPH
    AGG --> CORR
    facade -.->|"analyze()"| MORPH
    facade -.->|"analyze()"| CORR

    AGG --> YAMLIO
    AGG --> VTKIO
    AGG --> VISIO
```

The scaling law (`core/scaling.py`) owns the parallel-axis target-distance
equations once; the placement strategies are thin recipes over the shared
contact primitives (`solvers.py`).

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

Both algorithms implement the abstract
[`BaseGenerator`](/api-reference/index.md#generators) with a single
constructor signature:

```python
BaseGenerator(
    n_particles,        # target number of primary particles
    df, kf,             # fractal dimension and prefactor
    particle_dist,      # ParticleDistribution for primary radii
    overlap_tolerance=1e-5,
    placement='solved', # 'sampled' | 'solved' | 'constructed' or instance
    scaling=None,       # ScalingLaw instance or 'count'/'mass' (default mass)
    seed=None,          # seed for the generator's np.random.Generator
    # plus length_unit / mass_unit / density / surface_beta
)
```

`generate()` then returns a populated `Aggregate`. Because the contract is
identical, the factory `get_generator(method, ...)` (in
`generators/factory.py`) can dispatch on a string — `'pca'` or `'cca'`,
with `'fracval'` a deprecated alias for `(cca, mass, constructed)` — after
validating the legality matrix (e.g. `pca` rejects `constructed`, which is
merge-only). That factory is exactly what the top-level `pfa.generate()`
wraps. Keyword-only extras (`surface_beta`) are validated there and
rejected for placements that do not support them.

`pfa.analyze()` is the facade's read-side: it bundles
`radius_of_gyration`, `center_of_mass`, and both per-measure
fractal-dimension estimators (sandbox by default, pair-correlation via
`estimator="pcf"`) into a typed `MorphologyReport` (`rg`, `com`, `n`,
`estimator`, per-measure `df_num_est`/`r2_num`/`r_num`/`num_correlation`
and `df_mass_est`/`r2_mass`/`r_mass`/`mass_correlation`).

## The placement strategy layer

The scaling law fixes *where* the center of each added particle or cluster
must sit (distance `L` or `Γ` from the cluster center) but not *which*
particles touch. Resolving that contact problem is delegated to a strategy
object, selected by name:

- [`PlacementStrategy`](/api-reference/index.md#placement) — the ABC. Its two
  abstract methods mirror the two aggregation stages: `place_particle()`
  (single particle onto a cluster, the PCA stage) and `merge_clusters()`
  (two clusters onto a common `Γ`, the CCA stage).
- `SolvedPlacement` (default) — FLAGE, Skorupski et al. (2014). The closed
  form `solve_tangency` intersects the target sphere with a reference
  particle's contact sphere to get exact touching points; candidates are
  overlap-filtered, with a Monte Carlo fallback.
- `SampledPlacement` — Filippov et al. (2000). Pure Monte Carlo sampling on
  the target sphere with gradual tolerance relaxation until a candidate is
  accepted (typically several times slower).
- `ConstructedPlacement` — Morán et al. (2019). A contact pair is selected
  from reachability across `Γ`; the incoming cluster is rotated into
  contact, residual overlaps spun out about the contact axis, and the COM
  separation verified. PCA-stage placement is unsupported (raises).
- `get_placement(name_or_strategy, ...)` — the factory, accepting names
  (with the deprecated `'algebraic'`/`'random'` aliases) or instances
  (pandas-style). `BaseGenerator.__init__` resolves through it, passing the
  generator's `overlap_tolerance` and its seeded `Generator`.

All strategies share their contact primitives through
`generators/placement/solvers.py` (`solve_tangency`, `mc_touch_place`,
`mc_touch_merge`), so they differ only in their recipe, not in their
primitives. The former `FracVALGenerator` merge *is*
`ConstructedPlacement`; there is no generator-specific contact logic left
outside the placement layer.

## Analysis

The analysis layer is a set of pure functions over an `Aggregate`:
`morphology.py` provides `radius_of_gyration` (mass-weighted, including each
sphere's intrinsic gyration, per Morán et al. 2019 Eq. (3)) and
`center_of_mass`; `correlation.py` provides `pair_correlation_function`
and its mass-weighted mirror `mass_pair_correlation_function`
(both differenced curves; `estimate_fractal_dimension` is the shared
power-law fitter), `plot_pair_correlation` for diagnosis;
`sandbox.py` provides the cumulative mass-radius family
(`number_radius_function`/`mass_radius_function`, one-shot
`number_sandbox_dimension`/`mass_sandbox_dimension`, `plot_sandbox`).
No function mutates the aggregate.

## I/O

The io layer serializes an `Aggregate` for downstream use: `data.py` writes
a YAML snapshot bundling particle data with generation parameters and
analysis results; `vtk.py` builds the pyvista point cloud (`export_vtk`) and
MultiBlock dataset (`export_vtm`) for ParaView; `visualization.py` performs
off-screen pyvista rendering (`save_screenshot`) and assembles MP4 rotation
videos (`save_rotation_video`). Rendering exporters need a working 3D
backend — see the [io guide](/user-guide/io.md) for headless-environment
notes.

## Tests

The test suite mirrors the source layout:

- `tests/test_core/` — `Aggregate`, distributions, 3D math helpers
- `tests/test_generators/` — both algorithms, factory + legality matrix,
  placement strategies, solvers, scaling laws, v0.3.0 regression anchors
- `tests/test_analysis/` — morphology, correlation, MorphologyReport
- `tests/test_io/` — YAML and visualization exports
- `tests/test_density.py` — root-level density test
- `tests/test_compat_aliases.py` — deprecated-alias equivalence (T4/T8)
- `tests/test_seed_reproducibility.py` — seed determinism (T5)
- `tests/fixtures/` — shared fixture data (v0.3.0 baseline snapshots)

Slow performance tests carry the `benchmark` pytest marker and can be
deselected with `-m "not benchmark"` (see
[Contributing](/contributing.md#running-the-tests)).
