# pyFracAggregate

[![CI](https://github.com/vanvonzhang/pyFracAggregate/actions/workflows/test.yml/badge.svg)](https://github.com/vanvonzhang/pyFracAggregate/actions/workflows/test.yml)
[![PyPI version](https://img.shields.io/pypi/v/pyFracAggregate)](https://pypi.org/project/pyFracAggregate/)
[![Python versions](https://img.shields.io/pypi/pyversions/pyFracAggregate)](https://pypi.org/project/pyFracAggregate/)
[![Docs](https://img.shields.io/badge/docs-online-blue)](https://vanvonzhang.github.io/pyFracAggregate/)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue)](LICENSE)

A Python library for generating synthetic fractal aggregates — clusters of
spherical primary particles with a tunable morphology, such as soot and other
aerosols — unified across four classical generation algorithms behind one API,
with built-in morphological analysis and export to common scientific formats.

## Features

- **Four generation algorithms, one API** — particle-cluster aggregation
  (`'pca'`), cluster-cluster aggregation (`'cca'`), FracVAL (`'fracval'`), and
  the Thouy & Jullien tunable CCA (`'tdcca'`), all selected with a single
  `method=` keyword.
- **Two placement strategies** — FLAGE-style algebraic touching-point
  computation (default) or Monte Carlo random placement with tolerance
  relaxation.
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
import numpy as np
import pyFracAggregate as pfa

np.random.seed(0)
agg = pfa.generate(200, 1.8, 1.9, method='pca')  # N=200, Df=1.8, kf=1.9

summary = pfa.analyze(agg)   # Rg=13.274 nm, Df_estimated=1.714, R2=0.964
print(agg.current_size, summary['Df_estimated'])   # 200 1.714241520287105

pfa.export_yaml(agg, 'aggregate.yaml')
pfa.export_vtk(agg, 'aggregate.vtk')
```

Generation is stochastic and draws from NumPy's global legacy random state:
call `np.random.seed(...)` immediately before `pfa.generate(...)` for
reproducible aggregates. The single-realization `Df_estimated` scatters
around the requested `df`; average over realizations for ensemble statements.

## Methods

| Keyword | Algorithm | Family | Polydispersity | Reference |
|---|---|---|---|---|
| [`pca`](https://vanvonzhang.github.io/pyFracAggregate/background/index.html#pca-particle-cluster-aggregation) | Particle-cluster aggregation | particle-cluster | approximate (mean radius) | Skorupski et al., 2014 |
| [`cca`](https://vanvonzhang.github.io/pyFracAggregate/background/index.html#cca-cluster-cluster-aggregation) | Cluster-cluster aggregation | cluster-cluster | approximate (number-weighted) | Filippov et al., 2000 |
| [`fracval`](https://vanvonzhang.github.io/pyFracAggregate/background/index.html#fracval-tunable-cca-for-polydisperse-primaries) | FracVAL tunable CCA | cluster-cluster | native (mass-weighted) | Morán et al., 2019 |
| [`tdcca`](https://vanvonzhang.github.io/pyFracAggregate/background/index.html#tdcca-thouy-jullien-tunable-cca) | Thouy & Jullien tunable CCA | cluster-cluster | supported (mass-weighted Rg) | Thouy & Jullien, 1994 |

Each keyword links to the corresponding section of the
[background chapter](https://vanvonzhang.github.io/pyFracAggregate/background/index.html)
on the documentation site, which derives each algorithm's principle,
guarantees, and limits.

## Documentation

Full documentation — background theory, user guide, tutorial, API reference,
architecture notes, and contributing instructions — is hosted at:

**https://vanvonzhang.github.io/pyFracAggregate/**

## License

[MIT](LICENSE)
