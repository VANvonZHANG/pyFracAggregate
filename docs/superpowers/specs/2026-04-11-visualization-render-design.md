# Visualization: Render & Rotation Video Export

## Summary

Add two new IO functions to pyFracAggregate:

1. **`export_render()`** — render a 3D aggregate screenshot (PNG) using pyvista
2. **`export_rotation_video()`** — generate a 360-degree rotation animation (MP4) using pyvista + imageio

Both use offscreen rendering and share a common sphere mesh builder.

## Architecture

New file: `src/pyFracAggregate/io/visualization.py`

```
src/pyFracAggregate/io/
├── __init__.py
├── data.py            # YAML export (existing)
├── vtk.py             # VTK/VTM export (existing)
└── visualization.py   # NEW: screenshot + rotation video
```

### Shared Helper

Extract `_build_sphere_mesh(aggregate) -> pyvista.PolyData` from `vtk.py` into a shared location. This function converts an `Aggregate` into a pyvista mesh of spheres, used by both existing VTK exports and the new visualization functions.

## API Design

### `export_render()`

```python
def export_render(
    aggregate: Aggregate,
    path: str,
    color: str = "lightblue",
    opacity: float = 0.8,
    background: str = "white",
    camera_position: str | tuple = "iso",
    window_size: tuple[int, int] = (1024, 768),
) -> None:
```

- Renders the aggregate as 3D spheres using pyvista
- Preset camera positions: `"iso"`, `"xy"`, `"xz"`, `"yz"`, `"front"` (pyvista built-ins)
- Custom camera via `(position, focal_point, up)` tuple
- Offscreen rendering, no window popup
- Output format determined by `path` extension (PNG)

### `export_rotation_video()`

```python
def export_rotation_video(
    aggregate: Aggregate,
    path: str,
    color: str = "lightblue",
    opacity: float = 0.8,
    background: str = "white",
    window_size: tuple[int, int] = (1024, 768),
    n_frames: int = 72,
    fps: int = 24,
    elevation: float = 30.0,
) -> None:
```

- Camera orbits 360 degrees around the aggregate center
- `n_frames` total frames at `fps` frame rate
- `elevation` controls camera pitch (default 30 degrees)
- Offscreen rendering, frame-by-frame capture
- Encoded to MP4 via `imageio` with `imageio-ffmpeg` plugin
- Path must end in `.mp4` or raises `ValueError`
- Raises `ImportError` with install instructions if ffmpeg unavailable

### Public API

Both functions exported from `pyFracAggregate.__init__`:

```python
import pyFracAggregate as pfa
pfa.export_render(aggregate, "output.png")
pfa.export_rotation_video(aggregate, "output.mp4", n_frames=120, fps=30)
```

## Dependencies

```toml
[project.optional-dependencies]
plot = ["matplotlib>=3.5.0", "imageio[ffmpeg]>=2.9.0"]
```

- `imageio[ffmpeg]` only needed for MP4 video output
- `pyvista` is already a core dependency (no change)

## Tests

File: `tests/test_io/test_visualization.py`

- `test_export_render_creates_png` — verify file creation, non-empty
- `test_export_render_camera_presets` — test preset camera positions
- `test_export_rotation_video_creates_mp4` — verify MP4 generation (use low frame count)
- `test_export_rotation_video_invalid_path` — non-`.mp4` path raises `ValueError`
- `test_export_rotation_video_no_ffmpeg` — missing ffmpeg raises `ImportError`

## Implementation Notes

- Reuse sphere mesh building logic from `vtk.py` via shared helper
- Both functions use `pyvista.Plotter(off_screen=True)` for headless rendering
- Video pipeline: render frame → numpy array → imageio writer → MP4
- The shared helper placement should be decided during implementation (either in `visualization.py` and imported by `vtk.py`, or in a small shared module)
