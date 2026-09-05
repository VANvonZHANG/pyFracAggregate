# Background

pyFracAggregate generates synthetic nanoparticle aggregates: clusters of
spherical primary particles with a prescribed fractal morphology. This chapter
is the theory companion to the [API reference](/api-reference/index.md). It
answers one practical question: **given a target morphology** — particle count
{math}`N`, fractal dimension {math}`D_f`, prefactor {math}`k_f`, and a size
distribution of primaries — **and a downstream use, which generation
algorithm should be used, what is its principle, and what are its guarantees
and boundaries?**

The organizing fact: the scaling law that defines the target is a statistical
statement about *ensembles* of aggregates, and no single finite cluster of
hard spheres satisfies it exactly. Every algorithm here is a heuristic that
approaches the target. The through-line below is: what the target is, how
achievement of it is measured, and how each algorithm approximates it.

## Why generate fractal aggregates

Soot, aerosols, and colloids form mass-fractal clusters: many nanoscale
primary particles coagulated into low-density, self-similar structures. For
downstream computations the arrangement of the primaries is often the quantity
of interest, and experiments rarely deliver controlled, reproducible geometry.
Synthetic generation fills the gap:

- **Light-scattering simulations.** T-matrix, Rayleigh-Debye-Gans, and
  discrete-dipole codes consume primary-particle coordinates directly;
  morphology drives cross sections and angular patterns.
- **Numerical TEM images.** Projected synthetic aggregates yield images
  comparable to real micrographs, validating sizing and morphology
  diagnostics.
- **Population-balance initial conditions.** Coagulation and sintering models
  need structured (not point-mass) aggregates as inputs.
- **Drag and mobility.** Translational and rotational drag, diffusion, and
  thermophoresis of low-density aggregates depend on where the primaries sit.

With {math}`(N, D_f, k_f)` and the size distribution as explicit inputs,
morphology becomes a controlled variable rather than a case-by-case
reconstruction.

## Morphology parameters

A mass-fractal aggregate is characterized by the scaling law

```{math}
:label: eq-scaling-law

N = k_f \left( \frac{R_g}{a_p} \right)^{D_f},
```

where {math}`R_g` is the radius of gyration and {math}`a_p` a characteristic
primary-particle radius.

- **{math}`D_f`, the fractal dimension**, states how mass grows with linear
  size: {math}`D_f = 3` is a compact sphere, {math}`D_f = 1` a chain.
  Diffusion-limited cluster aggregation gives {math}`D_f \approx 1.75`–
  {math}`1.8`, reaction-limited aggregation {math}`D_f \approx 2.1`; flame
  soot is typically reported at {math}`1.6`–{math}`1.9`.
- **{math}`k_f`, the fractal prefactor**, quantifies packing density at fixed
  {math}`N`; soot-like values are {math}`1.2`–{math}`2.4`. Since {math}`k_f`
  correlates with {math}`D_f` and with the measurement method, the pair
  should be quoted together.

Both are ensemble-level properties, measured by fitting {eq}`eq-scaling-law`
across aggregates spanning {math}`N`, or within one aggregate from its pair
correlation function. Two consequences follow. The law is asymptotic: at the
small {math}`N` typical of simulations a single aggregate scatters
appreciably around it, and the effective {math}`k_f` drifts with {math}`N`.
And requesting {math}`(D_f, k_f)` does not return them exactly — the realized
dimension of one aggregate is a random variable, so average over realizations
for ensemble statements.

**How this library measures quality.**
[analyze()](/api-reference/index.md#analysis) reports both measures (see
the next section): `df_num_est` and `df_mass_est`, computed by the sandbox
estimator by default (`estimator="pcf"` selects the classic
pair-correlation path). The intended workflow is: generate, `analyze()`,
compare `df_num_est` with the requested `df` — and `df_mass_est` too when
primaries are polydisperse — averaging over several seeded realizations
before quoting ensemble numbers.

## Counting versus mass measure

The scaling law counts primaries; every primary contributes "1". That is
the **number (counting) measure**, and it is what `df` in
{math}`N = k_f (R_g/a)^{D_f}` refers to. Real primaries have different
sizes, and a second, equally valid question exists: how does the
**mass** of primaries within distance {math}`r` of a primary grow? Two
definitions that differ by one word:

- **number Df** — the *number* of primary centres within radius r of a
  primary scales as {math}`N(r) \propto r^{D_f}`;
- **mass Df** — the *summed mass* of primaries within radius r of a
  primary scales as {math}`M(r) \propto r^{D_{f,m}}`.

Because the material density is constant, mass weights {math}`m_i`
differ from volume weights {math}`r_i^3` only by a constant factor:
mass-based and volume-based weighting are identical, and the fitted
exponent is unaffected.

### Which estimator, and why

| Estimator | Result on the tutorial Part-2 aggregate (N=256, {math}`\sigma_g=1.6`, seed 0) | Verdict |
|---|---|---|
| number sandbox {math}`\langle N(r)\rangle` | {math}`D_f = 1.782` ({math}`R^2 = 0.994`) | robust |
| mass sandbox {math}`\langle M(r)\rangle` | {math}`D_{f,m} = 1.879` ({math}`R^2 = 0.976`) | robust — library default |
| differenced mass PCF {math}`C_m(r)` | {math}`R^2 \approx 0.125` | noisy on single realizations |
| mass-weighted seeds (strict) | outside (1, 3) on 5/5 seeds | do not use |

The failure modes are statistical, not conceptual: mass weighting is
dominated by the few largest primaries (participation ratio
{math}`(\sum m)^2 / \sum m^2 \approx 43` of 256 primaries here), and
differencing amplifies that noise. `C_m(r)` remains useful for
ensemble-averaged curves.

### Interpreting {math}`D_{f,m} - D_f`

Both measures live on the same support (the set of primary centres), so
the two exponents coincide when radii are uncorrelated with position
(measured Spearman rank correlation {math}`\approx -0.05` on the tutorial
aggregate). A systematic difference indicates size-position
correlation — e.g. large primaries anchoring a dense core — and is the
physical situation where reporting both numbers matters (compacted or
coated soot).

## Primary particles

Primary sizes enter through a distribution object:
[Monodisperse](/api-reference/index.md#core) gives every particle the same
radius (the default); [LognormalDistribution](/api-reference/index.md#core)
samples from a lognormal with geometric mean `mean` and geometric standard
deviation `std` ({math}`\sigma_g \geq 1`; values at or below 1.0 collapse to the
monodisperse case).

The scaling law presumes a single {math}`a_p`; with polydisperse primaries two
things change. The characteristic radius becomes a convention — the PCA and
CCA target equations here use the mean primary radius — and {math}`R_g` must
use mass weights and the intrinsic gyration of each solid sphere, via the
parallel-axis theorem (Morán et al., 2019, Eq. (3)):

```{math}
:label: eq-rg-mass-weighted

R_g^2 = \frac{1}{\sum_i m_i} \sum_i m_i \left[ \left| \mathbf{R}_i - \mathbf{R}_c \right|^2 + \frac{3}{5} r_i^2 \right],
```

where {math}`\mathbf{R}_c` is the center of mass and {math}`\tfrac{3}{5} r_i^2`
the self-gyration of sphere {math}`i`. This library's `radius_of_gyration`
implements exactly this definition, and the FracVAL merge criterion below
extends it to combining two clusters.

## Two algorithmic families

**Particle-cluster aggregation (PCA)** grows an aggregate one primary at a
time, each addition placed at the distance that keeps the cluster on the
scaling-law trajectory and then brought into contact. Any {math}`N` is
natural, generation is cheap, and per-step control is exact in the
monodisperse limit — but the texture is that of single-particle attachment,
smoother and more compact than real soot, which forms by cluster-cluster
collisions.

**Cluster-cluster aggregation (CCA)** builds small clusters first and merges
them pairwise, each merge choosing the center-to-center separation that makes
the *merged* cluster satisfy the scaling law; merging repeats hierarchically
until one aggregate remains. This mirrors diffusion-limited coagulation and
its texture, at the cost of a more delicate merge criterion and, in some
formulations, constraints on {math}`N`.

The two `method` values select the family: `pca` is the first; `cca` the
second. Everything else — weighting and contact strategy — is the `scaling`
and `placement` axes, so the classical named algorithms (Filippov CCA, FLAGE,
FracVAL) are *coordinates* `(method, scaling, placement)` rather than separate
methods.

## The coordinate system

| Literature method | Coordinate |
|---|---|
| DLA-style PCA | `(pca, count, solved)` |
| Filippov CCA (2000) | `(cca, count, sampled)` |
| FLAGE-style CCA (Skorupski 2014) | `(cca, count, solved)` |
| FracVAL (Morán 2019) | `(cca, mass, constructed)` |

### PCA — particle-cluster aggregation

Selected with `pfa.generate(method='pca')`. The `scaling` axis weights the
per-step target: the count form (Filippov's original) is shown below, the
mass form is its mass-weighted counterpart.

**Principle.** From a seed particle, each subsequent particle {math}`n` is
assigned a target sphere of radius {math}`L` about the cluster's geometric
center and placed in contact with an existing primary on that sphere
(Filippov et al., 2000; optimized in Skorupski et al., 2014). {math}`L`
inverts the scaling law for the aggregate after the addition (Filippov et
al., Eq. (10)):

```{math}
:label: eq-pca-L

L^2 = \frac{n^2 a^2}{n - 1} \left( \frac{n}{k_f} \right)^{2/D_f}
      - \frac{n a^2}{n - 1}
      - n a^2 \left( \frac{n - 1}{k_f} \right)^{2/D_f},
```

with {math}`a` the mean primary radius.

**Strengths.** Any {math}`N \geq 1`; cheapest method; exact per-step target
for monodisperse primaries.

**Limits.** The count-weighted form of the target degrades as
{math}`\sigma_g` grows (choose `scaling='mass'` for polydisperse primaries),
and sequential attachment cannot build cluster-cluster texture.

### CCA — cluster-cluster aggregation

Selected with `pfa.generate(method='cca')`; the default `scaling='mass'`
targets polydisperse primaries, `scaling='count'` reproduces Filippov's
original number-weighted form shown here.

**Principle** (Filippov et al., 2000). Small PCA clusters of five primaries
are built first (the last absorbs any remainder), then merged pairwise;
targets with {math}`N \leq 8` are simply produced by PCA. For merging
clusters of {math}`N_1` and {math}`N_2` primaries ({math}`N = N_1 + N_2`),
the separation {math}`\Gamma` makes the merged radius of gyration satisfy the
scaling law:

```{math}
:label: eq-cca-gamma

\Gamma^2 = \frac{N^2 a^2}{N_1 N_2} \left( \frac{N}{k_f} \right)^{2/D_f}
           - \frac{N}{N_2} R_{g,1}^2 - \frac{N}{N_1} R_{g,2}^2.
```

**Strengths.** Hierarchical structure closer to diffusion-limited
coagulation; any {math}`N`; both {math}`D_f` and {math}`k_f` enter.

**Limits.** {math}`\Gamma` is the number-weighted (monodisperse) form, so
polydispersity is approximate; merge orientations are Monte Carlo samples.

### The FracVAL coordinate — mass scaling + constructed contacts

Selected with `pfa.generate(method='cca', scaling='mass',
placement='constructed')` (the keyword `method='fracval'` is a deprecated
alias for this coordinate).

**Principle** (Morán et al., 2019). A tunable CCA redesigned for polydisperse
primaries. The initial cluster size adapts to {math}`N` (5 primaries for
small aggregates, about {math}`N/10` up to {math}`N = 500`, then 50;
{math}`N \leq 8` delegates to PCA), and the merge criterion replaces number
weights with mass weights (Morán et al., Eqs. (3) and (6)):

```{math}
:label: eq-fracval-gamma

R_g = \bar{r}_p \left( \frac{N}{k_f} \right)^{1/D_f},
\qquad
\Gamma^2 = \frac{m^2 R_g^2 - m \left( m_1 R_{g,1}^2 + m_2 R_{g,2}^2 \right)}{m_1 m_2},
```

with {math}`m = m_1 + m_2` and mass-weighted gyration radii as in
{eq}`eq-rg-mass-weighted`. Placement at each merge is deterministic: contact
pairs are found from reachability across {math}`\Gamma`; the exact contact
point comes from a sphere-sphere intersection; the incoming cluster is
rotated to bring its particle into contact, with residual overlaps resolved
by rotating about the contact axis; acceptance requires the center-of-mass
separation to match {math}`\Gamma`, with a Monte Carlo fallback otherwise.

**Strengths.** Polydispersity handled natively; tighter {math}`(N, D_f, k_f)`
targeting; co-designed with numerical TEM image synthesis.

**Limits.** The most compute-intensive method; occasionally falls back to
random placement in tight merges.

### TDCCA — removed in v0.4

The Thouy & Jullien (1994) tunable CCA was **removed in v0.4**. The shipped
implementation was a lattice-seeding/continuous-rotation hybrid that could
not reproduce the paper's coordinates, and it ignored the `placement` axis
entirely — an unfaithful port is worse than none. Pin `pyFracAggregate<0.4`
if you depended on it, or reimplement from the paper (Thouy & Jullien, Eqs.
(11)–(12)).

## Placement strategies

The scaling law fixes where the center of the new object must lie (distance
{math}`L` or {math}`\Gamma`) but not which particles touch. Resolving this
contact problem without unwanted overlaps is the placement layer's job,
selected with the `placement` argument of `pfa.generate` (valid for both
methods; `constructed` is merge-only and rejected for `pca`). The names
describe how the contact comes to be:

- **`placement='solved'` (default)** — FLAGE (Skorupski et al., 2014). The
  contact is *solved* in closed form: intersecting the target sphere with a
  reference particle's contact sphere yields a circle of exact touching
  points; candidates are overlap-filtered and sampled, with a Monte Carlo
  fallback when no closed-form solution exists. Contacts are near-exact and
  target distances are honored precisely.
- **`placement='sampled'`** — Monte Carlo placement (Filippov et al., 2000).
  The contact is *sampled*: random directions and cluster orientations are
  tried on the target sphere; the first candidate whose minimum gap falls
  within tolerance is accepted. The tolerance starts at {math}`10^{-3} a`
  and relaxes in {math}`0.05\,a` steps so generation always terminates; the
  best candidate is kept.
- **`placement='constructed'`** — FracVAL contact construction (Morán et
  al., 2019). The contact is *constructed*: a specific contact pair is
  chosen from reachability across {math}`\Gamma`, the incoming cluster is
  rotated into contact, residual overlaps are spun out about the contact
  axis, and the center-of-mass separation is verified against
  {math}`\Gamma` (Monte Carlo fallback otherwise). Cluster merging only.

The pre-v0.4 names `'algebraic'` (→ `'solved'`) and `'random'` (→
`'sampled'`) still resolve with a `DeprecationWarning` and will be removed
in 1.0.

`overlap_tolerance` sets the admissible interpenetration {math}`\delta`:
contact becomes {math}`d_{ij} \geq r_i + r_j - \delta`, with default
{math}`10^{-5}` (essentially hard spheres). Larger values produce overlapping
primaries — occasionally wanted to mimic sintering necks — but bias the
realized morphology and any fractal dimension measured from it.

## Choosing a method

The table below summarizes the trade-offs; see
[generate()](/api-reference/index.md#top-level-api) in the API reference for
the full signature.

```{list-table}
:header-rows: 1

* - Coordinate
  - Family
  - N flexibility
  - Polydispersity
  - Df targeting
  - Typical use
* - `(pca, ·, ·)`
  - particle-cluster
  - any N ≥ 1
  - count approximate / mass exact per-step
  - per-step L, Filippov Eq. (10)
  - fast baselines; parameter scans
* - `(cca, count, ·)`
  - cluster-cluster
  - any N ≥ 1 (N ≤ 8 via PCA)
  - approximate (number-weighted Γ)
  - per-merge Γ, Filippov et al.
  - monodisperse hierarchical structure
* - `(cca, mass, constructed)`
  - cluster-cluster
  - any N ≥ 1 (N ≤ 8 via PCA)
  - native (mass-weighted)
  - per-merge Γ, Morán Eqs. (3), (6)
  - polydisperse soot models; numerical TEM
```

By downstream use:

- **Light-scattering input:** `pca` for parameter scans at moderate
  {math}`N`; `cca` when texture must be representative, mass scaling if
  primaries are polydisperse.
- **Numerical TEM images:** `(cca, mass, constructed)`, designed for exactly
  this pipeline.
- **Population-balance initial conditions:** `pca` for cheap
  small-{math}`N` samples; `cca` when morphology statistics feed downstream
  kernels.
- **Drag and mobility:** any coordinate; keep `overlap_tolerance` small so
  the geometry is a genuine hard-sphere packing.

Whatever the choice, close the loop with
[analyze()](/api-reference/index.md#analysis): compare `Df_estimated` with
the requested `df`, check `R2`, and average over realizations before quoting
ensemble numbers.

## References

1. A. V. Filippov, M. Zurita, and D. E. Rosner. *Fractal-like aggregates:
   Relation between morphology and physical properties.* Journal of Colloid
   and Interface Science, 229(1):261–273, 2000.
   [doi:10.1006/jcis.2000.7027](https://doi.org/10.1006/jcis.2000.7027)
2. J. Morán, A. Fuentes, F. Liu, and J. Yon. *FracVAL: An improved tunable
   algorithm of cluster-cluster aggregation for generation of fractal
   structures formed by polydisperse primary particles.* Computer Physics
   Communications, 239:225–237, 2019.
   [doi:10.1016/j.cpc.2019.01.015](https://doi.org/10.1016/j.cpc.2019.01.015)
3. J. Morán, A. Fuentes, F. Liu, and J. Yon. *FracVAL: An improved tunable
   algorithm for generation of fractal aggregates formed by polydisperse
   primary particles and subsequent numerical tilt series TEM images
   generation.* 1st Franco-AMSUD Energy and Environment Meeting, Marseille,
   France, March 2019. HAL: hal-02144789. (Companion conference paper on the
   TEM-image extension.)
4. K. Skorupski, J. Mroczka, T. Wriedt, and N. Riefler. *A fast and accurate
   implementation of tunable algorithms used for generation of fractal-like
   aggregate models.* Physica A: Statistical Mechanics and its Applications,
   404:106–117, 2014.
   [doi:10.1016/j.physa.2014.02.072](https://doi.org/10.1016/j.physa.2014.02.072)
5. R. Thouy and R. Jullien. *A cluster-cluster aggregation model with tunable
   fractal dimension.* Journal of Physics A: Mathematical and General,
   27(9):2953–2963, 1994.
   [doi:10.1088/0305-4470/27/9/012](https://doi.org/10.1088/0305-4470/27/9/012)

Further reading:

6. C. M. Sorensen. *Light scattering by fractal aggregates: A review.* Aerosol
   Science and Technology, 35(2):648–687, 2001.
   [doi:10.1080/02786820117868](https://doi.org/10.1080/02786820117868)
7. C. M. Sorensen. *The mobility of fractal aggregates: A review.* Aerosol
   Science and Technology, 45(7):765–779, 2011.
   [doi:10.1080/02786826.2011.560909](https://doi.org/10.1080/02786826.2011.560909)
