# pyFracAggregate

[![CI](https://github.com/vanvonzhang/pyFracAggregate/actions/workflows/test.yml/badge.svg)](https://github.com/vanvonzhang/pyFracAggregate/actions/workflows/test.yml)
[![PyPI version](https://img.shields.io/pypi/v/pyFracAggregate)](https://pypi.org/project/pyFracAggregate/)
[![Python versions](https://img.shields.io/pypi/pyversions/pyFracAggregate)](https://pypi.org/project/pyFracAggregate/)
[![Docs](https://img.shields.io/badge/docs-online-blue)](https://vanvonzhang.github.io/pyFracAggregate/)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue)](LICENSE)

A Python library for generating synthetic fractal aggregates — clusters of
spherical primary particles with a tunable morphology, such as soot and other
aerosols — behind one coordinate-system API: **method** × **scaling** ×
**placement** select among the classical generation algorithms
(particle-cluster and cluster-cluster aggregation, count- or mass-weighted
scaling, three contact-placement strategies), with built-in morphological
analysis and export to common scientific formats.

## Features

- **Three orthogonal axes, one API** — `method` (`'pca'` | `'cca'`) ×
  `scaling` (`'count'` | `'mass'`) × `placement` (`'solved'` | `'sampled'` |
  `'constructed'`); every classical algorithm is a coordinate in this system.
- **Three placement strategies** — closed-form tangency solving (default),
  Monte Carlo sampling, or FracVAL-style contact construction.
- **Reproducible generation** — pass `seed=` for bit-identical reruns of any
  legal coordinate.
- **Monodisperse and lognormal primary particles** — `Monodisperse` and
  `LognormalDistribution` size distributions feed any generator.
- **Built-in morphology analysis** — radius of gyration, center of mass, pair
  correlation function, and fractal-dimension estimation with fit quality
  (`pfa.analyze`).
- **Rich exports** — YAML snapshot, VTK point cloud and VTM multiblock (via
  pyvista, ready for ParaView), off-screen static render, and rotation video.
- **Fully typed library with tests** — type hints throughout the source, and a
  pytest suite mirroring the package layout.

## Installation

```console
$ pip install pyFracAggregate
```

> **Requires Python ≥ 3.13.** The 3D math dependency `mathutils` only has
> usable wheels for the 3.13 ABI on several platforms; older interpreters can
> fail at compile time. See the
> [installation guide](https://vanvonzhang.github.io/pyFracAggregate/user-guide/installation.html)
> for details and platform notes.

To install from source for development:

```console
$ pip install -e ".[dev]"
```

## Quick start

```python
import pyFracAggregate as pfa

agg = pfa.generate(200, 1.8, 1.9, method='pca', seed=0)  # N=200, Df=1.8, kf=1.9

summary = pfa.analyze(agg)   # MorphologyReport
print(agg.current_size, summary.df_est)   # 200 1.6126416651056448

pfa.export_yaml(agg, 'aggregate.yaml')
pfa.export_vtk(agg, 'aggregate.vtk')
```

Generation is stochastic; pass `seed=` for reproducible aggregates (the
global `numpy.random` state is never consulted). The single-realization
`df_est` scatters around the requested `df`; average over realizations for
ensemble statements.

## The coordinate system

Every classical aggregate algorithm is a coordinate in a three-axis system
(`method`, `scaling`, `placement`):

| Literature method | `pyFracAggregate` coordinate |
|---|---|
| DLA-style PCA | `(pca, count, solved)` |
| Filippov CCA (2000) | `(cca, count, sampled)` |
| FLAGE-style CCA (Skorupski 2014) | `(cca, count, solved)` |
| FracVAL (Morán 2019) | `(cca, mass, constructed)` |

The two `method` families —
[`pca`](https://vanvonzhang.github.io/pyFracAggregate/background/index.html#pca-particle-cluster-aggregation)
(particle-cluster) and
[`cca`](https://vanvonzhang.github.io/pyFracAggregate/background/index.html#cca-cluster-cluster-aggregation)
(cluster-cluster) — are introduced in the
[background chapter](https://vanvonzhang.github.io/pyFracAggregate/background/index.html)
on the documentation site, which derives each algorithm's principle,
guarantees, and limits. `'fracval'` remains a deprecated alias for
`(cca, mass, constructed)` until 1.0; `'tdcca'` was removed in v0.4.

## Documentation

Full documentation — background theory, user guide, tutorial, API reference,
architecture notes, and contributing instructions — is hosted at:

[Documentation](https://vanvonzhang.github.io/pyFracAggregate/)

## License

[MIT](LICENSE)
