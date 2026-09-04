# Analyzing aggregates

The one-call entry point,
[`pfa.analyze()`](/api-reference/index.md#analysis), computes the core
morphological properties of an aggregate:

```python
import pyFracAggregate as pfa

agg = pfa.generate(256, df=1.8, kf=1.9, method="pca", seed=0)

report = pfa.analyze(agg)   # MorphologyReport
print(report.rg, report.df_est, report.r2)
```

which for this seed gives (values rounded):

```text
rg      = 15.230
com     = [-3.788, 5.321, -0.262]
n       = 256
df_est  = 1.493
r2      = 0.981
```

`analyze()` returns a typed `MorphologyReport` (a dataclass; pass it through
`dataclasses.asdict` when you need a plain dict):

| Attribute | Meaning |
|---|---|
| `rg` | Radius of gyration (float, in `length_unit`). |
| `com` | Center of mass, shape `(3,)` array. |
| `n` | Number of particles (`aggregate.current_size`). |
| `df_est` | Fractal dimension estimated from the pair correlation function. |
| `r2` | Coefficient of determination of the underlying log-log fit. |
| `r_centers` | Bin centers of the pair correlation `r` grid. |
| `pair_correlation` | The `C(r)` values on that grid. |

`export_yaml(agg, path, analysis_results=report)` serializes the report
under the legacy snapshot key names (`Rg`, `CoM`, `N`, `Df_estimated`,
`R2`, plus `r_centers`/`pair_correlation`), keeping YAML snapshots
field-compatible with earlier versions. (Until v0.3 the report was a plain
dict with those capitalized keys.)

## Interpreting `df_est` and `r2`

The estimate comes from a log-log linear fit of the pair correlation function
{math}`C(r)`. Because {math}`C(r) \propto r^{D_f - 3}` in the fractal regime,
the fitted slope equals {math}`D_f - 3` and the reported
`df_est` is `slope + 3` — see
[the background chapter](/background/index.md#morphology-parameters) for the
theory. The fit window used by `analyze()` runs from the mean primary radius
to {math}`R_g`.

Two caveats learned from the example above: a single realization at moderate
{math}`N` under-estimates the requested {math}`D_f` (here 1.65 for a request
of 1.8), and `R2` measures the fit quality, not the agreement with the
requested `df`. Average `Df_estimated` over several seeded realizations before
quoting ensemble numbers.

## Individual functions

All helpers accept an `Aggregate` and are re-exported at the top level (full
signatures in the [API reference](/api-reference/index.md#analysis)).

### Morphology

```python
rg = pfa.radius_of_gyration(agg)   # float
com = pfa.center_of_mass(agg)      # (3,) array
```

`radius_of_gyration` includes the finite size of the primaries via the
parallel-axis theorem ({math}`3/5\,r^2` per solid sphere).

### Pair correlation function

```python
r_centers, c_r = pfa.pair_correlation_function(agg, bins=50)
```

`bins` sets the number of equal-width bins between 0 and `r_max`; when `r_max`
is `None` (default) it defaults to twice the largest distance of any particle
center from the cluster centroid. Distances are in `length_unit`.

### Fractal-dimension estimation

```python
df_est, r2, fit = pfa.estimate_fractal_dimension(
    r_centers, c_r, r_min=np.mean(agg.radii), r_max=rg
)
```

`r_min`/`r_max` bound the regression window — restrict them to the fractal
regime (between roughly the mean primary radius and {math}`R_g`) rather than
fitting the whole curve. `fit` is a dict with the `slope`, `intercept`, and
the fitted points (`x_fit`, `y_fit`).

### Plotting

`plot_pair_correlation` renders {math}`C(r)` on log-log axes with the fractal
fit and fit window; pass `save_path="pcf.png"` to write a file instead of
opening a window.

```python
pfa.plot_pair_correlation(agg, save_path="pcf.png")
```

```{note}
Plotting requires matplotlib, which is not installed by default: run
`pip install "pyFracAggregate[plot]"` (see
[Installation](installation.md#install-from-pypi)). On servers without a
display, matplotlib automatically selects its non-interactive backend at
import time, and `save_path` writes the figure to disk without opening a
window.
```

## Units

An `Aggregate` carries its units as plain attributes, set at construction
time by the generator: `length_unit` (default `'nm'`), `mass_unit`
(default `'g'`), and `density` (default `1.0`). Particle masses are computed
as {math}`\rho \cdot \frac{4}{3}\pi r^3`, i.e. density times sphere volume in
the chosen units. `pfa.generate()` forwards these as keyword-only arguments:

```python
agg = pfa.generate(
    32, df=1.8, kf=1.9,
    length_unit="um", mass_unit="kg", density=1.8,
    particle_dist=pfa.Monodisperse(15.0),  # radii now in um
)
```

`pfa.analyze()` and `radius_of_gyration` return values in `length_unit`;
nothing converts units for you — if you generate with `length_unit="um"` and
radii of 15.0, `Rg` is in um as well. The units are carried through verbatim
by the YAML export (see [Exporting data](io.md)).
