# Visualization Render & Rotation Video Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `export_render()` (3D screenshot PNG) and `export_rotation_video()` (360° rotation MP4) to pyFracAggregate's IO layer.

**Architecture:** New `visualization.py` module in `io/` with a shared `_build_sphere_mesh()` helper. Both functions use pyvista offscreen rendering. Video encoding uses `imageio[ffmpeg]`. The helper is also used by `vtk.py` to eliminate duplicate sphere-building code.

**Tech Stack:** pyvista (already a dependency), imageio[ffmpeg] (new optional dependency), numpy

---

### Task 1: Add `imageio[ffmpeg]` to `[plot]` optional dependencies

**Files:**
- Modify: `pyproject.toml:22-25`

- [ ] **Step 1: Update pyproject.toml**

In `pyproject.toml`, change the `[project.optional-dependencies]` plot section:

```toml
plot = [
    "matplotlib>=3.5.0",
    "imageio[ffmpeg]>=2.9.0",
]
```

- [ ] **Step 2: Install updated dependencies**

Run: `pip install -e ".[plot,dev]"`
Expected: Successful install with imageio and imageio-ffmpeg

- [ ] **Step 3: Commit**

```bash
git add pyproject.toml
git commit -m "chore: add imageio[ffmpeg] to plot optional dependencies"
```

---

### Task 2: Create shared sphere mesh helper and `export_render()`

**Files:**
- Create: `src/pyFracAggregate/io/visualization.py`
- Modify: `src/pyFracAggregate/io/vtk.py` (refactor to use shared helper)
- Test: `tests/test_io/test_visualization.py`

- [ ] **Step 1: Write failing test for `export_render()`**

Create `tests/test_io/test_visualization.py`:

```python
import os
import pytest
import numpy as np
import pyFracAggregate as pfa


@pytest.fixture
def small_aggregate():
    return pfa.generate(n_particles=20, df=1.8, kf=1.2, method="pca")


class TestExportRender:
    def test_export_render_creates_png(self, small_aggregate, tmp_path):
        path = str(tmp_path / "render.png")
        pfa.export_render(small_aggregate, path)
        assert os.path.exists(path)
        assert os.path.getsize(path) > 0

    def test_export_render_custom_color(self, small_aggregate, tmp_path):
        path = str(tmp_path / "render_red.png")
        pfa.export_render(small_aggregate, path, color="red", opacity=0.5)
        assert os.path.exists(path)

    def test_export_render_camera_presets(self, small_aggregate, tmp_path):
        for preset in ["iso", "xy", "xz", "yz"]:
            path = str(tmp_path / f"render_{preset}.png")
            pfa.export_render(small_aggregate, path, camera_position=preset)
            assert os.path.exists(path)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_io/test_visualization.py -v`
Expected: FAIL with `ImportError: cannot import name 'export_render' from 'pyFracAggregate'`

- [ ] **Step 3: Create `visualization.py` with `_build_sphere_mesh()` and `export_render()`**

Create `src/pyFracAggregate/io/visualization.py`:

```python
import pyvista as pv
from pyFracAggregate.core.aggregate import Aggregate


def _build_sphere_mesh(aggregate: Aggregate) -> pv.MultiBlock:
    """Build a MultiBlock of sphere meshes from an Aggregate."""
    blocks = pv.MultiBlock()
    positions = aggregate.positions
    radii = aggregate.radii
    for i in range(len(positions)):
        sphere = pv.Sphere(radius=radii[i], center=positions[i])
        blocks.append(sphere)
    return blocks


def export_render(
    aggregate: Aggregate,
    path: str,
    color: str = "lightblue",
    opacity: float = 0.8,
    background: str = "white",
    camera_position: str | tuple = "iso",
    window_size: tuple[int, int] = (1024, 768),
) -> None:
    """Render a 3D screenshot of the aggregate and save as PNG.

    Args:
        aggregate: The fractal aggregate to render.
        path: Output file path (should end in .png).
        color: Sphere color name or hex.
        opacity: Sphere opacity (0.0 to 1.0).
        background: Background color name or hex.
        camera_position: Preset name ("iso", "xy", "xz", "yz", "front")
            or a (position, focal_point, up) tuple.
        window_size: (width, height) in pixels.
    """
    plotter = pv.Plotter(off_screen=True, window_size=window_size)
    plotter.set_background(background)
    mesh = _build_sphere_mesh(aggregate)
    plotter.add_mesh(mesh, color=color, opacity=opacity)
    if isinstance(camera_position, str):
        plotter.camera_position = camera_position
    else:
        plotter.camera_position = camera_position
    plotter.screenshot(path)
    plotter.close()
```

- [ ] **Step 4: Refactor `vtk.py` to use shared helper**

Replace the sphere-building loop in `export_vtm()` with the shared helper. Edit `src/pyFracAggregate/io/vtk.py`:

```python
import pyvista as pv
from pyFracAggregate.core.aggregate import Aggregate
from pyFracAggregate.io.visualization import _build_sphere_mesh


def export_vtm(aggregate: Aggregate, path: str):
    """Export the aggregate as a VTM (MultiBlock) file.

    Each monomer is represented as a separate sphere block.
    """
    blocks = _build_sphere_mesh(aggregate)
    blocks.save(path)


def export_vtk(aggregate: Aggregate, path: str):
    """Export the aggregate as a VTK PolyData file (point cloud with attributes).

    In ParaView, use the 'Glyph' filter with 'Sphere' type to visualize.
    """
    positions = aggregate.positions
    radii = aggregate.radii
    masses = aggregate.masses

    point_cloud = pv.PolyData(positions)
    point_cloud["radius"] = radii
    point_cloud["mass"] = masses

    point_cloud.save(path)
```

- [ ] **Step 5: Export new functions from `__init__.py`**

In `src/pyFracAggregate/__init__.py`, add import at line 11 (after the vtk import):

```python
from pyFracAggregate.io.vtk import export_vtm, export_vtk
from pyFracAggregate.io.data import export_yaml
from pyFracAggregate.io.visualization import export_render, export_rotation_video
```

And add to `__all__` list:

```python
    "export_render",
    "export_rotation_video",
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `pytest tests/test_io/test_visualization.py -v`
Expected: All 3 tests PASS

Run: `pytest tests/test_io/ -v` (existing tests still pass)
Expected: All tests PASS

- [ ] **Step 7: Commit**

```bash
git add src/pyFracAggregate/io/visualization.py src/pyFracAggregate/io/vtk.py src/pyFracAggregate/__init__.py tests/test_io/test_visualization.py
git commit -m "feat: add export_render() for 3D aggregate screenshots"
```

---

### Task 3: Implement `export_rotation_video()`

**Files:**
- Modify: `src/pyFracAggregate/io/visualization.py`
- Modify: `tests/test_io/test_visualization.py`

- [ ] **Step 1: Write failing tests for `export_rotation_video()`**

Add to `tests/test_io/test_visualization.py`:

```python
class TestExportRotationVideo:
    def test_creates_mp4(self, small_aggregate, tmp_path):
        path = str(tmp_path / "rotation.mp4")
        pfa.export_rotation_video(small_aggregate, path, n_frames=4, fps=4)
        assert os.path.exists(path)
        assert os.path.getsize(path) > 0

    def test_invalid_path_raises(self, small_aggregate, tmp_path):
        path = str(tmp_path / "rotation.gif")
        with pytest.raises(ValueError, match="mp4"):
            pfa.export_rotation_video(small_aggregate, path)

    def test_no_ffmpeg_raises(self, small_aggregate, tmp_path, monkeypatch):
        import pyFracAggregate.io.visualization as vis

        # Patch imageio.get_writer to raise ModuleNotFoundError
        original_get_writer = vis.imageio.get_writer

        def mock_get_writer(*args, **kwargs):
            raise ModuleNotFoundError("No module named 'imageio_ffmpeg'")

        monkeypatch.setattr(vis.imageio, "get_writer", mock_get_writer)
        path = str(tmp_path / "rotation.mp4")
        with pytest.raises(ImportError, match="imageio-ffmpeg"):
            pfa.export_rotation_video(small_aggregate, path)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_io/test_visualization.py::TestExportRotationVideo -v`
Expected: FAIL with `AttributeError` or `ImportError` (function not yet implemented)

- [ ] **Step 3: Implement `export_rotation_video()` in `visualization.py`**

Add import at top of `visualization.py`:

```python
import numpy as np
```

Add function after `export_render()`:

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
    """Generate a 360-degree rotation animation of the aggregate and save as MP4.

    Args:
        aggregate: The fractal aggregate to animate.
        path: Output file path (must end in .mp4).
        color: Sphere color name or hex.
        opacity: Sphere opacity (0.0 to 1.0).
        background: Background color name or hex.
        window_size: (width, height) in pixels.
        n_frames: Total number of frames for a full 360° rotation.
        fps: Frames per second in the output video.
        elevation: Camera elevation angle in degrees.

    Raises:
        ValueError: If path does not end in .mp4.
        ImportError: If imageio-ffmpeg is not installed.
    """
    if not path.lower().endswith(".mp4"):
        raise ValueError(f"Output path must end in .mp4, got: {path}")

    try:
        import imageio
    except ImportError:
        raise ImportError(
            "imageio is required for video export. "
            "Install with: pip install imageio[ffmpeg]"
        )

    try:
        imageio.get_writer("test.mp4", format="FFMPEG")
    except Exception:
        pass  # imageio-ffmpeg available, will create real writer below

    mesh = _build_sphere_mesh(aggregate)
    plotter = pv.Plotter(off_screen=True, window_size=window_size)
    plotter.set_background(background)
    plotter.add_mesh(mesh, color=color, opacity=opacity)

    center = np.mean(aggregate.positions, axis=0)
    rg = np.max(np.linalg.norm(aggregate.positions - center, axis=1))
    distance = max(rg * 4, 10.0)

    writer = imageio.get_writer(path, fps=fps)

    for i in range(n_frames):
        angle = 360.0 * i / n_frames
        azimuth_rad = np.radians(angle)
        elevation_rad = np.radians(elevation)

        pos = center + distance * np.array([
            np.cos(elevation_rad) * np.cos(azimuth_rad),
            np.cos(elevation_rad) * np.sin(azimuth_rad),
            np.sin(elevation_rad),
        ])

        plotter.camera.position = pos
        plotter.camera.focal_point = center
        plotter.camera.up = (0, 0, 1)
        plotter.reset_clipping_range()

        frame = plotter.screenshot(return_img=True)  # type: ignore[assignment]
        writer.append_data(frame)

    writer.close()
    plotter.close()
```

Also update the `test_no_ffmpeg_raises` test — the mock needs to target `imageio` at import time, not `imageio.get_writer`. Simplify the test:

```python
    def test_no_ffmpeg_raises(self, small_aggregate, tmp_path, monkeypatch):
        """Missing imageio should raise ImportError."""
        import pyFracAggregate.io.visualization as vis

        # Force imageio import to fail
        monkeypatch.setattr(vis, "imageio", None, raising=False)
        path = str(tmp_path / "rotation.mp4")
        with pytest.raises(ImportError, match="imageio"):
            pfa.export_rotation_video(small_aggregate, path)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_io/test_visualization.py -v`
Expected: All tests PASS (6 tests total)

- [ ] **Step 5: Commit**

```bash
git add src/pyFracAggregate/io/visualization.py tests/test_io/test_visualization.py
git commit -m "feat: add export_rotation_video() for 360° rotation MP4 export"
```

---

### Task 4: Full test suite and example

**Files:**
- Modify: `examples/generate_and_export.py` (add visualization examples)
- Modify: `pyproject.toml` (lock file not needed — just verify)

- [ ] **Step 1: Run full test suite**

Run: `pytest -v`
Expected: All tests PASS (existing + new visualization tests)

- [ ] **Step 2: Run linter**

Run: `ruff check src/pyFracAggregate/io/visualization.py`
Expected: No errors

- [ ] **Step 3: Update example file**

Append to `examples/generate_and_export.py`:

```python
# --- Visualization export ---
pfa.export_render(agg, "aggregate_render.png", camera_position="iso")
pfa.export_rotation_video(agg, "aggregate_rotation.mp4", n_frames=72, fps=24)
print(f"Render saved to aggregate_render.png")
print(f"Rotation video saved to aggregate_rotation.mp4")
```

- [ ] **Step 4: Commit**

```bash
git add examples/generate_and_export.py
git commit -m "docs: add visualization examples to generate_and_export.py"
```
