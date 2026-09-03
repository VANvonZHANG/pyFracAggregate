# Tutorial: basic usage

This page walks you through the complete pyFracAggregate pipeline — generate,
analyze, visualize, and export — twice: first a monodisperse aggregate built
with particle-cluster aggregation (PCA), then a polydisperse aggregate built
with FracVAL. It is a static adaptation of the
[demo notebook](https://github.com/vanvonzhang/pyFracAggregate/blob/main/examples/pyFracAggregate_demo.ipynb),
which remains in `examples/` if you prefer an executable format.

Every code block below was executed in the order shown, with
pyFracAggregate 0.2.0 and `numpy.random.seed(0)` set before each generation
step. The printed values are the real measured outputs of that run, and the
embedded figures are the actual artifacts of those exact calls (saved under
different file names in the docs tree). If you run the snippets yourself you
will get the same numbers; with a different seed you get a different
realization, with a slightly different measured morphology.

```{note}
Plotting needs matplotlib, which is not installed by default:
`pip install "pyFracAggregate[plot]"` (see
[Installation](/user-guide/installation.md#install-from-pypi)). 3D rendering
uses pyvista, which is a base dependency.
```

## Part 1 — monodisperse PCA aggregate

### Generate the aggregate

Start with [`pfa.generate()`](/api-reference/index.md#top-level-api), the
single entry point for all four algorithms. We ask for 200 primary particles
with a fractal dimension {math}`D_f = 1.8` and prefactor {math}`k_f = 1.9` —
typical soot-like values (see
[Morphology parameters](/background/index.md#morphology-parameters)) — using
the `'pca'` method, a fast particle-cluster baseline.

Set the seed **before** calling `generate()`: generation is stochastic and
draws from NumPy's global random state, so the seed is what makes a run
reproducible (details in
[Reproducibility and seeding](/user-guide/generators.md#reproducibility-and-seeding)).

```python
import numpy as np
import pyFracAggregate as pfa

np.random.seed(0)
agg = pfa.generate(n_particles=200, df=1.8, kf=1.9, method="pca")

print(f"Particles: {agg.current_size}")
print(f"Radii:     {agg.radii.min():.3f} to {agg.radii.max():.3f}")
extent = agg.positions.max(axis=0) - agg.positions.min(axis=0)
print(f"Extent:    {extent[0]:.1f} x {extent[1]:.1f} x {extent[2]:.1f} {agg.length_unit}")
```

```text
Particles: 200
Radii:     1.000 to 1.000
Extent:    37.0 x 27.0 x 27.3 nm
```

With no `particle_dist` argument the primaries are monodisperse with radius
1.0 (in the default `length_unit` of nm), and 200 spheres at {math}`D_f = 1.8`
span roughly 37 nm — an open, branched object, not a compact ball.

### Analyze morphology

[`pfa.analyze()`](/api-reference/index.md#analysis) computes the radius of
gyration, center of mass, and — via the pair correlation function — an
estimate of the fractal dimension, all in one call:

```python
results = pfa.analyze(agg)

print(f"Rg:           {results['Rg']:.3f} {agg.length_unit}")
print(f"CoM:          [{results['CoM'][0]:.2f}, {results['CoM'][1]:.2f}, {results['CoM'][2]:.2f}]")
print(f"N:            {results['N']}")
print(f"Df_estimated: {results['Df_estimated']:.3f}")
print(f"R2:           {results['R2']:.4f}")
```

```text
Rg:           13.274 nm
CoM:          [13.42, -0.87, -3.09]
N:            200
Df_estimated: 1.714
R2:           0.9643
```

Read the last two lines with care. The estimated {math}`D_f` is 1.714 against
a request of 1.8, and {math}`R^2 = 0.96` says the log-log fit is good —
{math}`R^2` measures fit quality, not agreement with the target. A single
realization at moderate {math}`N` typically under-estimates the requested
{math}`D_f`; average over several seeded realizations before quoting ensemble
numbers ([Interpreting `Df_estimated` and `R2`](/user-guide/analysis.md#interpreting-df_estimated-and-r2)).

### Pair correlation function

The estimate above comes from the pair correlation function
{math}`C(r) \propto r^{D_f - 3}`, so it is worth looking at the underlying
curve. Compute it explicitly, then let
[`plot_pair_correlation()`](/api-reference/index.md#analysis) redraw it with
the fractal fit and the fit window — the fit runs from the mean primary radius
to {math}`R_g`:

```python
r_centers, c_r = pfa.pair_correlation_function(agg, bins=50)
print(f"bins: {len(r_centers)}, r up to {r_centers[-1]:.1f} {agg.length_unit}")

pfa.plot_pair_correlation(
    agg,
    bins=50,
    show_fit=True,
    reference_df=1.8,
    save_path="pcf_pca.png",
)
```

```text
bins: 50, r up to 38.0 nm
Plot saved to pcf_pca.png
```

```{figure} ../_static/tutorial_pca_pcf.png
:alt: Log-log plot of the pair correlation function C(r) with power-law fit and fit window

The pair correlation function of the 200-particle PCA aggregate. Blue dots:
measured {math}`C(r)`. Red line: power-law fit over the fractal regime
(slope {math}`= D_f - 3`), giving {math}`D_f = 1.71`. Green dashed line:
reference slope for the requested {math}`D_f = 1.8`. Vertical lines mark the
fit window (mean primary radius to {math}`R_g`).
```

The measured points follow a clean power law between the fit bounds and peel
off at both ends — below the primary-particle scale and beyond {math}`R_g`,
where finite-size effects dominate. That middle decade is exactly where the
fractal dimension lives.

### Render the aggregate

Numbers check out; now look at the object.
[`export_render()`](/api-reference/index.md#io) writes an off-screen 3D
screenshot as PNG — `window_size` is capped at a modest 960 px to keep the
docs light:

```python
pfa.export_render(agg, "pca_render.png", color="dimgray", window_size=(960, 720))
```

```{figure} ../_static/tutorial_pca_render.png
:alt: Rendered 3D view of the monodisperse PCA aggregate

The 200-particle monodisperse PCA aggregate, rendered with pyvista. Every
primary has radius 1.0 nm; the open, branched shape is the {math}`D_f = 1.8`
morphology.
```

```{note}
`export_render` needs an OpenGL context even though it renders off-screen.
On a headless server it may need `xvfb-run` or an OSMesa-built VTK — the
workarounds are collected in
[Rendered image](/user-guide/io.md#rendered-image).
```

### Export the data

Finally, persist the aggregate. `export_yaml` writes a full snapshot —
particle data, units, and the generation parameters and analysis results you
pass in (recording the seed makes the file traceable to an identical
aggregate). `export_vtk` writes a lightweight point cloud for ParaView:

```python
pfa.export_yaml(
    agg,
    "aggregate.yaml",
    generation_params={"method": "pca", "n_particles": 200,
                       "df": 1.8, "kf": 1.9, "seed": 0},
    analysis_results=results,
)
pfa.export_vtk(agg, "aggregate.vtk")
```

This run produced a 21 KB `aggregate.yaml` and an 11 KB `aggregate.vtk`.
Format guidance is in [Exporting data](/user-guide/io.md) — YAML for
reproducibility, VTK point cloud for quick inspection, VTM MultiBlock when
you need explicit sphere meshes.

## Part 2 — polydisperse FracVAL aggregate

Real soot primaries are not all the same size. Repeat the pipeline with a
`LognormalDistribution` and the `'fracval'` method, which is designed for
polydisperse primaries. Two things change in the call: the distribution and
the method — everything downstream (`analyze`, plotting, export) is identical.

```python
np.random.seed(0)
poly = pfa.LognormalDistribution(mean=1.0, std=1.6)
agg_poly = pfa.generate(
    n_particles=256, df=1.8, kf=1.9, method="fracval", particle_dist=poly
)

print(f"Particles:   {agg_poly.current_size}")
print(f"Radii:       {agg_poly.radii.min():.3f} to {agg_poly.radii.max():.3f}")
print(f"Mean radius: {agg_poly.radii.mean():.3f}")
```

```text
Particles:   256
Radii:       0.301 to 3.065
Mean radius: 1.134
```

Two details worth noting:

- `std` is the **geometric** standard deviation. A value of 1.6 spreads the
  radii from 0.30 to 3.07 nm around a geometric mean of 1.0 nm — roughly a
  factor of ten between the smallest and largest primary. Values at or below
  1.0 collapse to monodisperse
  ([Particle size distributions](/user-guide/generators.md#particle-size-distributions)).
- FracVAL accepts any particle count ({math}`N \leq 8` internally falls back
  to a PCA call). Only `'tdcca'` requires a power of two. 256 is simply a
  typical soot-aggregate size.

### Analyze and compare

```python
results_poly = pfa.analyze(agg_poly)

print(f"Rg:           {results_poly['Rg']:.3f} {agg_poly.length_unit}")
print(f"CoM:          [{results_poly['CoM'][0]:.2f}, {results_poly['CoM'][1]:.2f}, {results_poly['CoM'][2]:.2f}]")
print(f"N:            {results_poly['N']}")
print(f"Df_estimated: {results_poly['Df_estimated']:.3f}")
print(f"R2:           {results_poly['R2']:.4f}")
```

```text
Rg:           17.282 nm
CoM:          [5.98, 11.78, 4.39]
N:            256
Df_estimated: 1.676
R2:           0.9770
```

```python
pfa.export_render(agg_poly, "fracval_render.png", color="dimgray", window_size=(960, 720))
```

```{figure} ../_static/tutorial_fracval_render.png
:alt: Rendered 3D view of the polydisperse FracVAL aggregate

The 256-particle polydisperse FracVAL aggregate. Primary radii follow a
lognormal distribution with geometric mean 1.0 nm and geometric standard
deviation 1.6; the subcluster-merge construction gives a lumpier, more
branched texture than the PCA aggregate of Part 1.
```

Side by side, the two runs differ in three ways:

| | Part 1 (PCA) | Part 2 (FracVAL) |
|---|---|---|
| Primaries | 1.000 nm (monodisperse) | 0.301–3.065 nm, lognormal |
| {math}`R_g` | 13.274 nm | 17.282 nm |
| `Df_estimated` ({math}`R^2`) | 1.714 (0.9643) | 1.676 (0.9770) |

- **Polydispersity.** The radius spread is the whole point of Part 2: large
  primaries anchor the cluster while small ones fill the crevices, which is
  what real combustion soot looks like
  ([Primary particles](/background/index.md#primary-particles)).
- **Construction.** PCA accretes one particle at a time onto a growing
  cluster; FracVAL grows small subclusters first and merges them
  cluster-to-cluster, so the render shows a lumpier, more branched object
  ([The four methods](/background/index.md#the-four-methods)).
- **Size and Df.** With the same {math}`D_f` and {math}`k_f` requested (and a
  somewhat larger {math}`N`), the FracVAL cluster has a larger {math}`R_g`
  (17.3 vs 13.3 nm) — the scaling-law targets are statistical, not exact per
  realization — and
  its measured {math}`D_f` is again below the 1.8 request (1.68, with a tight
  fit at {math}`R^2 = 0.98`). Same lesson as Part 1: one realization is one
  sample; seed several and average when you report morphology.

## Where to go next

- [Generating aggregates](/user-guide/generators.md) — all `generate()`
  parameters, the other two methods, placement strategies, seeding.
- [Analyzing aggregates](/user-guide/analysis.md) — the individual analysis
  functions and choosing the fit window.
- [Exporting data](/user-guide/io.md) — the five export formats, ParaView
  workflows, headless-rendering workarounds.
- [Background](/background/index.md) — what {math}`D_f` and {math}`k_f`
  mean and how each algorithm approximates them.
- [API reference](/api-reference/index.md) — full signatures.
