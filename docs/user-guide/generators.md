# Generating aggregates

All generation algorithms share one entry point, the factory function
[`pfa.generate()`](/api-reference/index.md#top-level-api):

```python
import pyFracAggregate as pfa

agg = pfa.generate(n_particles=64, df=1.8, kf=1.9)
```

`generate()` selects an algorithm on **three orthogonal axes** — `method`
(the aggregation schedule), `scaling` (how the parallel-axis target distances
are weighted), and `placement` (how contacts are found) — so every classical
algorithm from the literature is one coordinate in this system.

## Parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `n_particles` | `int` | — | Target number of primary particles. |
| `df` | `float` | — | Fractal dimension {math}`D_f`. |
| `kf` | `float` | — | Fractal prefactor {math}`k_f`. |
| `method` | `str` | `'pca'` | Algorithm family: `'pca'` (particle-cluster) or `'cca'` (cluster-cluster). |
| `scaling` | `str` | `'mass'` | Parallel-axis weighting: `'count'` (Filippov 2000) or `'mass'` (Morán 2019). |
| `placement` | `str` | `'solved'` | Contact strategy: `'sampled'`, `'solved'`, or `'constructed'` (cca only). |
| `particle_dist` | `ParticleDistribution` | `None` | Primary-particle radius distribution; `None` means `Monodisperse(1.0)`. |
| `overlap_tolerance` | `float` | `1e-5` | Maximum allowed interpenetration between sphere surfaces (in length units). |
| `seed` | `int` | `None` | Seed for bit-reproducible generation; `None` draws fresh entropy. |

For soot-like systems, typical values are {math}`D_f \approx 1.6`–`1.9` and
{math}`k_f \approx 1.2`–`2.4`; the science behind both parameters is in
[Morphology parameters](/background/index.md#morphology-parameters).

Keyword-only unit arguments are forwarded to the generator:
`length_unit='nm'`, `mass_unit='g'`, and `density=1.0` (see
[Units](analysis.md#units)).

## The coordinate system

| Literature method | Coordinate |
|---|---|
| DLA-style PCA | `(pca, count, solved)` |
| Filippov CCA (2000) | `(cca, count, sampled)` |
| FLAGE-style CCA (Skorupski 2014) | `(cca, count, solved)` |
| FracVAL (Morán 2019) | `(cca, mass, constructed)` |

```python
agg_dla      = pfa.generate(32, 1.78, 1.9, method="pca",  scaling="count", placement="solved")
agg_filippov = pfa.generate(32, 1.78, 1.9, method="cca",  scaling="count", placement="sampled")
agg_flage    = pfa.generate(32, 1.78, 1.9, method="cca",  scaling="count", placement="solved")
agg_fracval  = pfa.generate(32, 1.78, 1.9, method="cca",  scaling="mass",  placement="constructed")
```

As a rule of thumb: `pca` for fast scans, `(cca, mass, constructed)` for
polydisperse soot models (the FracVAL coordinate), `(cca, count, sampled)`
to reproduce Filippov-style generation. The trade-offs are compared in
[Choosing a method](/background/index.md#choosing-a-method).

```{note}
`method='fracval'` is a **deprecated alias** for
`(cca, mass, constructed)` — it emits a `DeprecationWarning` and will be
removed in 1.0. `method='tdcca'` (Thouy & Jullien) was **removed in v0.4**
without a replacement; pin `pyFracAggregate<0.4` if you need it.
```

## The scaling axis

The parallel-axis theorem splits every cluster by *weights*: counting each
primary as one (`scaling='count'`, the Filippov 2000 original) or weighting
by mass (`scaling='mass'`, physically correct for polydisperse primaries,
following Morán 2019). Both have scientific semantics — count reproduces the
original papers, mass handles size distributions honestly — so the axis is
exposed rather than hard-wired.

For a **monodisperse** distribution the two are mathematically equivalent
(masses are proportional to counts), so the `'mass'` default changes nothing
for monodisperse users. For **polydisperse** input, `'mass'` is the default
since v0.4 — this is a behavior change from v0.3, where `cca` used count
weighting (`'fracval'` was the only mass-weighted path).

## Particle size distributions

Primary-particle radii come from a `ParticleDistribution`. Two are provided:

- `Monodisperse(radius)` — all primaries share one radius.
- `LognormalDistribution(mean, std)` — `mean` is the *geometric* mean radius
  and `std` the *geometric* standard deviation (values at or below 1.0 collapse to
  monodisperse).

```python
import numpy as np

mono = pfa.Monodisperse(radius=15.0)
agg_mono = pfa.generate(32, 1.8, 1.9, particle_dist=mono)
print(np.unique(agg_mono.radii))    # [15.]

poly = pfa.LognormalDistribution(mean=15.0, std=1.6)
agg_poly = pfa.generate(32, 1.8, 1.9, particle_dist=poly)
print(agg_poly.radii.min(), agg_poly.radii.max())  # e.g. 4.67 ... 36.68
```

For the physics of polydisperse primaries see
[Primary particles](/background/index.md#primary-particles).

## Overlap tolerance

`overlap_tolerance` is the maximum allowed overlap between two sphere
*surfaces*: a contact is valid when
{math}`d_{ij} \geq r_i + r_j - \delta` with {math}`\delta` the tolerance. The
default of `1e-5` (in length units) produces essentially hard-sphere
contacts. Larger values let primaries interpenetrate — occasionally wanted to
mimic sintering necks, at the cost of biasing the realized morphology and any
fractal dimension measured from it:

```python
sintered = pfa.generate(32, 1.8, 1.9, overlap_tolerance=0.3)
```

## Placement strategy

The `placement` argument selects how each new particle or subcluster is
brought into contact with the existing aggregate. The names describe how the
contact comes to be: it is **solved** in closed form, **sampled** by Monte
Carlo, or **constructed** from a specified contact pair.

- `'solved'` (default) — FLAGE-style closed-form tangency computation with a
  Monte Carlo fallback (Skorupski et al., 2014). Contacts are near-exact and
  target distances are honored precisely.
- `'sampled'` — pure Monte Carlo sampling with gradual tolerance relaxation
  (Filippov et al., 2000). Contacts are slightly fuzzier, and generation is
  typically several times slower at moderate {math}`N`.
- `'constructed'` — FracVAL contact construction (Morán et al., 2019): a
  specified contact pair is selected and the merging cluster is rotated and
  translated into place, with a center-of-mass check. **Cluster merging
  only** — `method='pca'` with `placement='constructed'` raises
  `ValueError`.

```python
agg = pfa.generate(32, 1.8, 1.9, placement="sampled")
```

Practical guidance: keep the default — it is both more precise and faster.
Switch to `'sampled'` to reproduce older Filippov-style generation, and to
`'constructed'` for the FracVAL coordinate. The deprecated names
`'algebraic'` (→ `'solved'`) and `'random'` (→ `'sampled'`) still resolve
with a warning and will be removed in 1.0.

## Working with the `Aggregate`

`generate()` returns an [`Aggregate`](/api-reference/index.md#core): a
pre-allocated `(max_particles, 5)` NumPy array storing
`[x, y, z, radius, mass]`. The array-like accessors return **zero-copy
views** — mutating them mutates the aggregate:

```python
agg = pfa.generate(64, 1.8, 1.9)

agg.positions        # view, shape (N, 3)
agg.radii            # view, shape (N,)
agg.masses           # view, shape (N,)
agg.current_size     # number of valid particles (int) — the "N" of the cluster
agg.to_numpy()       # copy of the (N, 5) data block
```

Note the accessor for the particle count is `current_size` (the analysis
helper `pfa.analyze` returns it as `report.n`).

## Factory function or generator classes

`pfa.generate()` is a thin factory over two generator classes with an
identical constructor shape —
[`PCAGenerator`, `CCAGenerator`](/api-reference/index.md#generators).
Instantiating a class directly is equivalent but makes the full constructor
surface explicit, including the unit parameters that `generate()` only
forwards as keywords. The `scaling` and `placement` arguments also accept
**instances** (`CountScaling()`/`MassScaling()`, any `PlacementStrategy`)
pandas-style, for custom strategies:

```python
from pyFracAggregate.core.scaling import MassScaling

gen = pfa.CCAGenerator(
    n_particles=32, df=1.8, kf=1.9,
    particle_dist=pfa.Monodisperse(1.0),
    scaling=MassScaling(1.8, 1.9),
    length_unit="um", mass_unit="kg", density=1.8,
    seed=42,
)
agg = gen.generate()
```

## Reproducibility and seeding

Generation is stochastic. All generators draw randomness from a private
NumPy `Generator` created from the `seed` argument — the **global**
`numpy.random` state is never consulted, so `numpy.random.seed(...)` has no
effect on generation.

```python
first = pfa.generate(32, 1.8, 1.9, seed=42)
second = pfa.generate(32, 1.8, 1.9, seed=42)

print(np.array_equal(first.positions, second.positions))  # True
```

Same seed → bit-identical particle data (positions, radii, masses) for any
legal coordinate; different seed → a different realization; `seed=None` →
fresh entropy every call. Note that for `pca` the radius of gyration is
essentially fixed by `(n_particles, df, kf)` — each particle sits at exactly
the target distance from the running center, so the seed varies the
cluster's texture (orientations and contact choices), not its overall size.
