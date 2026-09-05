# Analyzing aggregates

The one-call entry point,
[`pfa.analyze()`](/api-reference/index.md#analysis), computes the core
morphological properties of an aggregate:

```python
import pyFracAggregate as pfa

agg = pfa.generate(256, df=1.8, kf=1.9, method="pca", seed=0)

report = pfa.analyze(agg)   # MorphologyReport
print(f"Rg:       {report.rg:.3f} {agg.length_unit}")
print(f"Df,num:   {report.df_num_est:.3f}  (R2={report.r2_num:.4f})")
print(f"Df,mass:  {report.df_mass_est:.3f}  (R2={report.r2_mass:.4f})")
```

which for this seed prints:

```text
Rg:       15.230 nm
Df,num:   1.889  (R2=0.9788)
Df,mass:  1.889  (R2=0.9788)
```

`analyze()` returns a typed `MorphologyReport` (a dataclass; pass it through
`dataclasses.asdict` when you need a plain dict):

| Field | Meaning |
|---|---|
| `rg` | Radius of gyration (float, in `length_unit`; mass-weighted). |
| `com` | Center of mass, shape `(3,)` array. |
| `n` | Number of particles (`aggregate.current_size`). |
| `estimator` | Which family produced the report: `"sandbox"` or `"pcf"`. |
| `df_num_est` | Number-based fractal dimension (counting measure). |
| `r2_num` | Fit quality of the number-measure fit. |
| `r_num`, `num_correlation` | Number-measure curve (sandbox: cumulative ⟨N(r)⟩; pcf: C(r)). |
| `df_mass_est` | Mass-based fractal dimension (≡ volume-based; constant density). |
| `r2_mass` | Fit quality of the mass-measure fit. |
| `r_mass`, `mass_correlation` | Mass-measure curve (sandbox: cumulative ⟨M(r)⟩; pcf: C_m(r)). |

`export_yaml(agg, path, analysis_results=report)` serializes the report
under the v0.6 snapshot key names (`Rg`, `CoM`, `N`, `estimator`,
`Df_num_estimated`, `R2_num`, `r_num`, `num_correlation`,
`Df_mass_estimated`, `R2_mass`, `r_mass`, `mass_correlation`). The
`estimator` key records which family produced the numbers, so snapshots
stay traceable. Consumers must read the `estimator` field before
interpreting `r_num`/`num_correlation` (sandbox mode holds cumulative
curves; pcf mode holds differenced curves).

## Interpreting `df_num_est`, `df_mass_est` and the r2 fields

`analyze()` reports both measures (see
[the background chapter](/background/index.md#counting-versus-mass-measure)
for what the two measures mean). For monodisperse primaries the mass
measure is the counting measure times a constant, so `df_mass_est`
equals `df_num_est` exactly — a useful sanity check. For polydisperse
primaries the two track the same underlying arrangement unless radii
correlate with position.

Two caveats: a single realization at moderate `N` deviates from the
requested `df` by a few tenths, and the `r2` fields measure fit quality,
not agreement with the request. Average over several seeded
realizations before quoting ensemble numbers.

`estimator="pcf"` switches the classic pair-correlation path (differenced
C(r)/C_m(r) curves); the mass PCF curve is noisy on single realizations —
prefer the default sandbox estimator for `df_mass_est`.

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

### Sandbox functions

```python
r_num, n_r = pfa.number_radius_function(agg)      # <N(r)>, 15 log-spaced points
r_mass, m_r = pfa.mass_radius_function(agg)       # <M(r)>, weights r_i^3

dfn, r2n, fitn = pfa.number_sandbox_dimension(agg)
dfm, r2m, fitm = pfa.mass_sandbox_dimension(agg)
```

`bins` is the number of log-spaced grid points; the default window runs
from the mean primary radius to `Rg` (pass `r_min`/`r_max` to override).
`plot_sandbox(agg, measure="both")` overlays both curves with their fits —
the measure-comparison figure.

### Mass pair correlation

```python
r, c_m = pfa.mass_pair_correlation_function(agg, bins=50)
```

Pairs are weighted by {math}`m_i m_j` (volume cubed; constant density makes
mass- and volume-weighting identical). Noisy on single realizations —
intended for ensemble-averaged curves.

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

`plot_pair_correlation` accepts `measure="num"` (default), `"mass"`, or
`"both"`. `plot_sandbox` accepts the same values with `"both"` as default.

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
