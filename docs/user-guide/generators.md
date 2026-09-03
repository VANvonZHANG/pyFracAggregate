# Generating aggregates

All four generation algorithms share one entry point, the factory function
[`pfa.generate()`](/api-reference/index.md#top-level-api):

```python
import pyFracAggregate as pfa

agg = pfa.generate(n_particles=64, df=1.8, kf=1.9)
```

## Parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `n_particles` | `int` | — | Target number of primary particles. |
| `df` | `float` | — | Fractal dimension {math}`D_f`. |
| `kf` | `float` | — | Fractal prefactor {math}`k_f`. |
| `method` | `str` | `'pca'` | Algorithm: `'pca'`, `'cca'`, `'fracval'`, or `'tdcca'`. |
| `particle_dist` | `ParticleDistribution` | `None` | Primary-particle radius distribution; `None` means `Monodisperse(1.0)`. |
| `overlap_tolerance` | `float` | `1e-5` | Maximum allowed interpenetration between sphere surfaces (in length units). |
| `placement` | `str` | `'algebraic'` | Contact-placement strategy: `'algebraic'` or `'random'`. |

For soot-like systems, typical values are {math}`D_f \approx 1.6`–`1.9` and
{math}`k_f \approx 1.2`–`2.4`; the science behind both parameters is in
[Morphology parameters](/background/index.md#morphology-parameters).

Keyword-only unit arguments are forwarded to the generator:
`length_unit='nm'`, `mass_unit='g'`, and `density=1.0` (see
[Units](analysis.md#units)).

## Choosing a method

The four `method` keywords cover the two classical algorithmic families —
particle-cluster (`'pca'`) and cluster-cluster
(`'cca'`, `'fracval'`, `'tdcca'`) aggregation. The trade-offs (polydispersity support, Df targeting,
typical use) are compared in [Choosing a method](/background/index.md#choosing-a-method);
as a rule of thumb: `pca` for fast scans, `fracval` for polydisperse soot
models, `cca` for monodisperse hierarchical structure, `tdcca` to reproduce
Thouy & Jullien structures.

```python
agg_pca    = pfa.generate(32, 1.78, 1.9, method="pca")
agg_cca    = pfa.generate(32, 1.78, 1.9, method="cca")
agg_fracval = pfa.generate(32, 1.78, 1.9, method="fracval")
agg_tdcca  = pfa.generate(32, 1.78, 1.9, method="tdcca")
```

```{warning}
`method="tdcca"` requires `n_particles` to be a power of two (the Thouy &
Jullien algorithm builds the cluster by hierarchically merging pairs of
equal-size subclusters). Any other value raises `ValueError`.
```

## Particle size distributions

Primary-particle radii come from a `ParticleDistribution`. Two are provided:

- `Monodisperse(radius)` — all primaries share one radius.
- `LognormalDistribution(mean, std)` — `mean` is the *geometric* mean radius
  and `std` the *geometric* standard deviation (values below 1.0 collapse to
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
brought into contact with the existing aggregate; it takes effect for
`method='pca'` and `method='cca'` (`fracval` and `tdcca` embed their own
contact logic). See [Placement strategies](/background/index.md#placement-strategies)
for the algorithmic detail.

- `'algebraic'` (default) — FLAGE-style exact touching-point computation with
  a Monte Carlo fallback. Contacts are near-exact and target distances are
  honored precisely.
- `'random'` — pure Monte Carlo sampling with gradual tolerance relaxation
  until a candidate satisfies the overlap tolerance. Contacts are slightly
  fuzzier, and generation is typically several times slower at moderate
  {math}`N` (the relaxed retries add up).

```python
agg = pfa.generate(32, 1.8, 1.9, placement="random")
```

Practical guidance: keep the default — it is both more precise and faster.
Switch to `'random'` mainly to reproduce older Filippov-style generation
behavior.

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
helper `pfa.analyze` returns it as `"N"`).

## Factory function or generator classes

`pfa.generate()` is a thin factory over four generator classes with an
identical constructor shape —
[`PCAGenerator`, `CCAGenerator`, `FracVALGenerator`, `ThouyJullienGenerator`](/api-reference/index.md#generators).
Instantiating a class directly is equivalent but makes the full constructor
surface explicit, including the unit parameters that `generate()` only
forwards as keywords:

```python
gen = pfa.PCAGenerator(
    n_particles=32, df=1.8, kf=1.9,
    particle_dist=pfa.Monodisperse(1.0),
    length_unit="um", mass_unit="kg", density=1.8,
)
agg = gen.generate()
```

## Reproducibility and seeding

Generation is stochastic. All generators draw randomness from NumPy's
**global** legacy random state (`numpy.random.*`); there is no `seed`
parameter in the API. Consequently:

- Seeding is done with `numpy.random.seed(...)` **immediately before each
  `generate()` call** (or before constructing and running a generator class —
  the draws happen inside `generate()`).
- `numpy.random.default_rng()` and other new-style `Generator` objects do
  **not** affect generation, because the code never consults them.

```python
import numpy as np

np.random.seed(42)
first = pfa.generate(32, 1.8, 1.9)

np.random.seed(42)
second = pfa.generate(32, 1.8, 1.9)

print(np.array_equal(first.positions, second.positions))  # True
```

This was verified for all four methods, both placement strategies, and
lognormal size distributions: same seed → bit-identical particle data;
different seed → a different realization. Note that for `pca` the radius of
gyration is fixed by `(n_particles, df, kf)` regardless of seed — the seed
varies the cluster's texture (orientations and contact choices), not its
overall size.
