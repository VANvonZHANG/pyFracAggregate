# Architecture

pyFracAggregate is organized around one idea: **every layer produces or
consumes a single central data structure**, the
[`Aggregate`](/api-reference/index.md#core), and generation is defined by
**three orthogonal choices** — the `method` axis (`pca` particle-cluster vs
`cca` cluster-cluster aggregation), the `scaling` axis (count weights after
Filippov 2000 vs mass weights after Morán 2019), and the `placement` axis
(`sampled` Monte Carlo, `solved` closed-form tangency, `constructed`
FracVAL contact pairs). A thin facade (`pfa.generate` / `pfa.analyze`)
validates the 10 legal combinations and dispatches; the analysis and I/O
layers are pure consumers of the finished aggregate.

The map below shows the whole library in one view — the two rows trace the
*write path* (a `generate()` call ending in an `Aggregate`) and the
collaborators each generator consults per growth step. It is fully
interactive — pan, zoom, toggle the theme, search nodes, or trace any
relationship — in the
[standalone viewer](../_static/pyfracaggregate-architecture.html){target=_blank},
which is also the best way to read it on a small screen.

```{raw} html
<iframe src="../_static/pyfracaggregate-architecture.html"
        style="width:100%;height:580px;border:1px solid var(--pst-color-border, #ccc);border-radius:8px;"
        title="Interactive pyFracAggregate architecture map"></iframe>
<p style="font-size:0.9em;">
<a href="../_static/pyfracaggregate-architecture.html" target="_blank">
Open the interactive architecture map in a full tab</a> — supports pan/zoom,
light/dark themes, search, focus, and relationship tracing.</p>
```

How to read the map:

- **Solid arrows are the main path**: user code → facade → factory →
  generator → `Aggregate` → analysis / export.
- **Dashed arrows are collaborator calls** the generator makes each step:
  the scaling law for the target distance, the distribution for radii,
  the placement strategy for contact.
- **Dashed regions are the four layers**: `generators/`, `core/`,
  `analysis/`, and `io/`.

## How a `generate()` call flows

1. **Facade.** `pfa.generate(n_particles, df, kf, ...)` defaults the
   particle distribution to `Monodisperse(1.0)` and forwards everything to
   the factory.
2. **Validation.** `get_generator()` (in `generators/factory.py`) resolves
   deprecated aliases (`'fracval'` → `(cca, mass, constructed)`,
   `'algebraic'`/`'random'` for placements) and rejects illegal
   method × placement combinations (`pca` has no `constructed` stage) and
   misplaced options such as `surface_beta` outside `'solved'`.
3. **Generator.** `PCAGenerator` or `CCAGenerator` is constructed holding
   one seeded `numpy.random.Generator`, the chosen `ScalingLaw` (resolved
   through `get_scaling`), and a `PlacementStrategy` (resolved through
   `get_placement`, which also passes in the generator's tolerance and
   RNG).
4. **Growth loop.** Each step asks two collaborators: the scaling law for
   *where* — the target center-to-center distance `L` (PCA) or `Γ` (CCA)
   from the parallel-axis theorem — and the placement strategy for *which
   particles touch* at that distance.
5. **Result.** `generate()` returns the populated `Aggregate`; the same
   `seed=` replays every draw, so any legal coordinate is reproducible.

The read path mirrors it: `pfa.analyze(aggregate, estimator=...)` bundles
morphology and per-measure fractal-dimension estimates into a
`MorphologyReport`, and the `io/` exporters serialize the aggregate (with
or without that report) to YAML, VTK/VTM, PNG, or MP4.

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

## Scaling and placement: the division of labor

Each growth step splits into two questions, owned by two separate strategy
layers:

- **Where** — the center of the added particle (PCA) or incoming cluster
  (CCA) must sit at a distance `L` or `Γ` from the cluster center. The
  [`ScalingLaw`](/api-reference/index.md#generators) in `core/scaling.py`
  owns these parallel-axis target-distance equations *once*:
  `CountScaling` uses particle-count weights (Filippov et al., 2000),
  `MassScaling` uses mass weights (Morán et al., 2019), and the two are
  mathematically equivalent for monodisperse primaries. Computing `Γ`
  needs the sub-cluster gyration radii, so the laws reuse
  `analysis.morphology.radius_of_gyration` — the one place `core/` reaches
  into `analysis/`.
- **Which particles touch** — resolving actual contact at that target
  distance is delegated to a `PlacementStrategy`. The strategies are thin
  recipes over the shared contact primitives in
  `generators/placement/solvers.py` (`solve_tangency`, Monte Carlo touch
  placement), so they differ in recipe, not in primitives.

## The placement strategy layer

The two questions above meet in a strategy object, selected by name:

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

The former `FracVALGenerator` merge *is* `ConstructedPlacement`; there is
no generator-specific contact logic left outside the placement layer.

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
