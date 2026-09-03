# API Reference

Complete reference for the 21 public symbols exported by `pyFracAggregate`
(the package's `__all__`), grouped by layer: the top-level facade, the core
data structures, the four generation algorithms, the analysis functions, the
I/O exporters, and the placement strategies.

## Top-level API

The facade functions cover the common workflow: build an aggregate with
`generate()` (which dispatches to one of the four generation methods), then
summarize its morphology with `analyze()`.

```{eval-rst}
.. autofunction:: pyFracAggregate.generate

.. autofunction:: pyFracAggregate.analyze
```

## Core

`Aggregate` is the central data structure every other layer produces or
consumes: a pre-allocated `(max_particles, 5)` NumPy array of
`[x, y, z, radius, mass]` rows whose `positions`, `radii`, and `masses`
properties are zero-copy views. The two distribution classes describe
primary-particle sizes and are passed to generators via the `particle_dist`
argument.

```{eval-rst}
.. autoclass:: pyFracAggregate.core.aggregate.Aggregate
   :members:

.. autoclass:: pyFracAggregate.core.distributions.Monodisperse
   :members:

.. autoclass:: pyFracAggregate.core.distributions.LognormalDistribution
   :members:
```

## Generators

Four algorithms share the `BaseGenerator` constructor contract
`(n_particles, df, kf, particle_dist, overlap_tolerance, placement)` and each
returns an `Aggregate` from its `generate()` method. Users normally reach them
through `generate(method=...)`; the classes are public for direct use and
subclassing.

```{eval-rst}
.. autoclass:: pyFracAggregate.generators.pca.PCAGenerator
   :members:

.. autoclass:: pyFracAggregate.generators.cca.CCAGenerator
   :members:

.. autoclass:: pyFracAggregate.generators.fracval.FracVALGenerator
   :members:

.. autoclass:: pyFracAggregate.generators.tdcca.ThouyJullienGenerator
   :members:
```

## Analysis

Morphological descriptors computed from an `Aggregate`: global quantities
(radius of gyration, center of mass) and the two-point pair correlation
function `C(r)`, from which the fractal dimension is estimated by log-log
regression. `analyze()` bundles the main ones into a summary dict, and
`plot_pair_correlation()` visualizes the fit (requires matplotlib).

```{eval-rst}
.. autofunction:: pyFracAggregate.analysis.morphology.radius_of_gyration

.. autofunction:: pyFracAggregate.analysis.morphology.center_of_mass

.. autofunction:: pyFracAggregate.analysis.correlation.pair_correlation_function

.. autofunction:: pyFracAggregate.analysis.correlation.estimate_fractal_dimension

.. autofunction:: pyFracAggregate.analysis.correlation.plot_pair_correlation
```

## I/O

Export an `Aggregate` for downstream use: a YAML snapshot bundling the particle
data with generation parameters and analysis results, VTK/VTM files built with
pyvista for ParaView and other tools, and off-screen rendered PNG images or
MP4 rotation videos. The render and video exporters require a working pyvista
3D backend (see the user guide for headless-environment notes).

```{eval-rst}
.. autofunction:: pyFracAggregate.io.data.export_yaml

.. autofunction:: pyFracAggregate.io.visualization.export_render

.. autofunction:: pyFracAggregate.io.visualization.export_rotation_video

.. autofunction:: pyFracAggregate.io.vtk.export_vtm

.. autofunction:: pyFracAggregate.io.vtk.export_vtk
```

## Placement

Placement strategies decide where a new particle or cluster touches the
existing structure while respecting the overlap tolerance; every generator
selects one via `placement=`. Both classes implement the same two entry
points: `place_particle()` for particle-cluster stages and `merge_clusters()`
for cluster-cluster stages.

```{eval-rst}
.. autoclass:: pyFracAggregate.generators.placement.algebraic.AlgebraicPlacement
   :members:

.. autoclass:: pyFracAggregate.generators.placement.random_.RandomPlacement
   :members:
```
