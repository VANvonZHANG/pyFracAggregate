# Exporting data

An [`Aggregate`](/api-reference/index.md#core) lives in memory; five exporters
(see the [I/O API reference](/api-reference/index.md#io)) write it to disk in
formats suited to different downstream uses. All snippets below assume:

```python
import pyFracAggregate as pfa

agg = pfa.generate(64, df=1.8, kf=1.9)
results = pfa.analyze(agg)
```

## YAML snapshot

`export_yaml` bundles everything about one aggregate in a single human-readable
file: particle data, the aggregate's units and density, and — optionally — the
generation parameters and analysis results you pass in:

```python
pfa.export_yaml(
    agg, "aggregate.yaml",
    generation_params={"method": "pca", "n_particles": 64,
                       "df": 1.8, "kf": 1.9, "seed": 42},
    analysis_results=results,
)
```

The written file has three top-level keys — `generation` (only if
`generation_params` is given), `aggregate` (`n_particles`, `length_unit`,
`mass_unit`, `density`, `positions`, `radii`, `masses`), and `analysis` (only
if `analysis_results` is given). Recording the seed you used (see
[Reproducibility](generators.md#reproducibility-and-seeding)) makes the
snapshot traceable back to an identical aggregate.

## VTK point cloud

`export_vtk` writes a lightweight VTK PolyData **point cloud**: one point per
primary, with `radius` and `mass` as point attributes:

```python
pfa.export_vtk(agg, "aggregate.vtk")
```

## VTM MultiBlock

`export_vtm` writes each primary as an explicit sphere mesh, grouped in a
pyvista MultiBlock. Heavier on disk than the point cloud, but ready to render
with no extra filters:

```python
pfa.export_vtm(agg, "aggregate.vtm")
```

## Rendered image

`save_screenshot` renders an off-screen 3D screenshot of the sphere mesh and
saves it as PNG. Sphere `color`, `opacity`, `background`, camera position,
and `window_size` are configurable:

```python
pfa.save_screenshot(agg, "render.png", color="dimgray", window_size=(512, 384))
```

## Rotation video

`save_rotation_video` animates a full 360° turn and writes an MP4
(`n_frames` frames at `fps`, camera at `elevation` degrees). Keep `n_frames`
modest while prototyping — rendering time scales with it:

```python
pfa.save_rotation_video(agg, "rotation.mp4", n_frames=72, fps=24)
```

MP4 writing goes through imageio's ffmpeg backend. The
`imageio[ffmpeg]` dependency installed with pyFracAggregate provides a
bundled ffmpeg; if you replaced it with plain `imageio`, install ffmpeg
yourself or re-add the extra (`pip install "imageio[ffmpeg]"`).

## Viewing VTK/VTM in ParaView

Both `aggregate.vtk` and `aggregate.vtm` open directly in
[ParaView](https://www.paraview.org/). The `.vtk` point cloud needs one extra
step to look like soot: select the source, apply the **Glyph** filter with
glyph type **Sphere**, scale by the `radius` array (scale factor 2 with the default
0.5-radius sphere source), and set
the number of theta/phi resolution to taste. The `.vtm` MultiBlock already
contains explicit sphere meshes, so it displays as-is after loading — at the
cost of a larger file.

````{warning}
**Rendering and video on headless servers.** `save_screenshot` and
`save_rotation_video` need a working OpenGL context even though they render
off-screen. On a machine without a GPU or display, pyvista/VTK may fall back
to software EGL successfully — or crash with EGL/segfault errors, depending
on the VTK build. The reliable fixes are a virtual display or software GL:

```console
$ xvfb-run -a python make_renders.py
```

or set `PYVISTA_OFF_SCREEN=true` with an OSMesa-built VTK (`libosmesa6-dev`
on Debian/Ubuntu, or `vtk-osmesa` from conda-forge). This is the same
workaround this project's CI needed; do not debug your physics first — check
the GL backend first.
````
