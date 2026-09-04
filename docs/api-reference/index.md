# API Reference

Complete reference for the public symbols exported by `pyFracAggregate`
(the package's `__all__`), grouped by layer: the top-level facade, the core
data structures and scaling laws, the two generation algorithms, the
analysis functions, the I/O exporters, and the placement strategies.

## Top-level API

The facade functions cover the common workflow: build an aggregate with
`generate()` (which validates the `method` × `scaling` × `placement`
coordinate and dispatches), then summarize its morphology with `analyze()`
(returning a `MorphologyReport`).

```{eval-rst}
.. autofunction:: pyFracAggregate.generate

.. autofunction:: pyFracAggregate.analyze
```

## Core

`Aggregate` is the central data structure every other layer produces or
consumes: a pre-allocated `(max_particles, 5)` NumPy array of
`[x, y, z, radius, mass]` rows whose `positions`, `radii`, and `masses`
properties are zero-copy views. The distribution classes describe
primary-particle sizes and are passed to generators via the `particle_dist`
argument; the scaling laws own the parallel-axis target-distance equations
(count- vs mass-weighted).

```{eval-rst}
.. autoclass:: pyFracAggregate.core.aggregate.Aggregate
   :members:

.. autoclass:: pyFracAggregate.core.distributions.Monodisperse
   :members:

.. autoclass:: pyFracAggregate.core.distributions.LognormalDistribution
   :members:

.. autoclass:: pyFracAggregate.core.distributions.FixedRadii
   :members:

.. autoclass:: pyFracAggregate.core.scaling.ScalingLaw
   :members:

.. autoclass:: pyFracAggregate.core.scaling.CountScaling
   :members:

.. autoclass:: pyFracAggregate.core.scaling.MassScaling
   :members:
```

## Generators

Both algorithms share the `BaseGenerator` constructor contract
`(n_particles, df, kf, particle_dist, overlap_tolerance, scaling,
placement, seed)` and each returns an `Aggregate` from its `generate()`
method. Users normally reach them through `generate(method=...)`; the
classes are public for direct use and subclassing.

```{eval-rst}
.. autoclass:: pyFracAggregate.generators.pca.PCAGenerator
   :members:

.. autoclass:: pyFracAggregate.generators.cca.CCAGenerator
   :members:
```

## Analysis

Morphological descriptors computed from an `Aggregate`: global quantities
(radius of gyration, center of mass) and the two-point pair correlation
function `C(r)`, from which the fractal dimension is estimated by log-log
regression. `analyze()` bundles the main ones into a `MorphologyReport`, and
`plot_pair_correlation()` visualizes the fit (requires matplotlib).

```{eval-rst}
.. autoclass:: pyFracAggregate.MorphologyReport
   :members:
```

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

.. autofunction:: pyFracAggregate.io.visualization.save_screenshot

.. autofunction:: pyFracAggregate.io.visualization.save_rotation_video

.. autofunction:: pyFracAggregate.io.vtk.export_vtm

.. autofunction:: pyFracAggregate.io.vtk.export_vtk
```

## Placement

Placement strategies decide where a new particle or cluster touches the
existing structure while respecting the overlap tolerance; every generator
selects one via `placement=` (name or instance). All classes implement the
same two entry points: `place_particle()` for particle-cluster stages and
`merge_clusters()` for cluster-cluster stages.

```{eval-rst}
.. autoclass:: pyFracAggregate.generators.placement.solved.SolvedPlacement
   :members:

.. autoclass:: pyFracAggregate.generators.placement.sampled.SampledPlacement
   :members:

.. autoclass:: pyFracAggregate.generators.placement.constructed.ConstructedPlacement
   :members:
```
