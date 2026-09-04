from __future__ import annotations

from typing import Any, Sequence, cast

import numpy as np
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


def _orbit_camera_position(
    center: np.ndarray,
    distance: float,
    azimuth_deg: float,
    elevation_deg: float,
) -> list[list[float]]:
    """Camera tuple [pos, focal, up] for a full pyvista camera_position assignment.

    Pure function (no rendering side effects) so the orbit geometry stays
    unit-testable without a plotter.
    """
    az = np.radians(azimuth_deg)
    el = np.radians(elevation_deg)
    pos = [
        float(center[0] + distance * np.cos(el) * np.cos(az)),
        float(center[1] + distance * np.cos(el) * np.sin(az)),
        float(center[2] + distance * np.sin(el)),
    ]
    return [pos, [float(c) for c in center], [0.0, 0.0, 1.0]]


def _framing_distance(bounds: Sequence[float]) -> float:
    """aerosol3d framing rule: 1.5x the largest extent, floored at 1.0."""
    length = max(bounds[1] - bounds[0], bounds[3] - bounds[2], bounds[5] - bounds[4])
    return max(length, 1.0) * 1.5


def _build_plotter(
    aggregate: Aggregate,
    *,
    color: str,
    opacity: float,
    color_by: str | None,
    cmap: str,
    background: str,
    window_size: tuple[int, int],
) -> pv.Plotter:
    """Assemble an off-screen Plotter with the aggregate mesh and styling.

    Args:
        aggregate: The fractal aggregate to render.
        color: Solid sphere color; ignored when color_by="radius" (pyvista
            scalars take precedence over a solid color).
        opacity: Sphere opacity (0.0 to 1.0).
        color_by: None (solid color) or "radius" (colormap over monomer radii).
        cmap: Colormap name for color_by="radius".
        background: Background color name or hex.
        window_size: (width, height) in pixels.

    Raises:
        ValueError: If the aggregate is empty or color_by is invalid.
    """
    if aggregate.current_size == 0:
        raise ValueError("Cannot render an empty aggregate (current_size == 0).")
    if color_by not in (None, "radius"):
        raise ValueError(f"color_by must be None or 'radius', got: {color_by!r}")

    plotter = pv.Plotter(off_screen=True, window_size=list(window_size))
    try:
        plotter.set_background(background)  # type: ignore[arg-type]  # pyvista stub quirk
        mesh = _build_sphere_mesh(aggregate)
        if color_by == "radius":
            # MultiBlock scalars need a same-named array on every block; the
            # combined mesh gets one actor and one shared color scale.
            for block, radius in zip(mesh, aggregate.radii):
                block.cell_data["radius"] = np.full(block.n_cells, float(radius))
            combined = mesh.combine(merge_points=False)
            r_min = float(aggregate.radii.min())
            r_max = float(aggregate.radii.max())
            plotter.add_mesh(
                combined, scalars="radius",
                cmap=cmap,  # type: ignore[arg-type]  # pyvista stub quirk
                clim=(r_min, r_max), opacity=opacity,
            )
        else:
            plotter.add_mesh(mesh, color=color, opacity=opacity)
        return plotter
    except Exception:
        plotter.close()
        raise


def save_screenshot(
    aggregate: Aggregate,
    path: str,
    color: str = "lightblue",
    opacity: float = 0.8,
    color_by: str | None = None,
    cmap: str = "viridis",
    background: str = "white",
    camera_position: str | tuple = "iso",
    window_size: tuple[int, int] = (1024, 768),
) -> None:
    """Render an off-screen 3D screenshot of the aggregate and save as PNG.

    Args:
        aggregate: The fractal aggregate to render.
        path: Output file path (must end in .png).
        color: Sphere color name or hex; ignored when color_by="radius".
        opacity: Sphere opacity (0.0 to 1.0).
        color_by: None (solid color) or "radius" (colormap over monomer radii).
        cmap: Colormap name for color_by="radius".
        background: Background color name or hex.
        camera_position: Preset name ("iso", "xy", "xz", "yz") or a
            (position, focal_point, up) tuple.
        window_size: (width, height) in pixels.

    Raises:
        ValueError: If path does not end in .png, the aggregate is empty,
            or color_by is invalid.
    """
    if not path.lower().endswith(".png"):
        raise ValueError(f"Output path must end in .png, got: {path}")

    plotter = _build_plotter(
        aggregate, color=color, opacity=opacity, color_by=color_by,
        cmap=cmap, background=background, window_size=window_size,
    )
    try:
        plotter.camera_position = cast(Any, camera_position)
        plotter.screenshot(path)
    finally:
        plotter.close()


def save_rotation_video(
    aggregate: Aggregate,
    path: str,
    color: str = "lightblue",
    opacity: float = 0.8,
    color_by: str | None = None,
    cmap: str = "viridis",
    background: str = "white",
    window_size: tuple[int, int] = (1024, 768),
    n_frames: int = 72,
    fps: int = 24,
    elevation: float = 30.0,
) -> None:
    """Generate a 360-degree rotation animation of the aggregate and save as MP4.

    The camera is auto-framed from the mesh bounds (aerosol3d rule) and
    orbits once around the aggregate at the given elevation.

    Args:
        aggregate: The fractal aggregate to animate.
        path: Output file path (must end in .mp4).
        color: Sphere color name or hex; ignored when color_by="radius".
        opacity: Sphere opacity (0.0 to 1.0).
        color_by: None (solid color) or "radius" (colormap over monomer radii).
        cmap: Colormap name for color_by="radius".
        background: Background color name or hex.
        window_size: (width, height) in pixels.
        n_frames: Total frames for the full 360 degree rotation.
        fps: Frames per second in the output video.
        elevation: Camera elevation angle in degrees.

    Raises:
        ValueError: If path does not end in .mp4, the aggregate is empty,
            or color_by is invalid.
        ImportError: If imageio is not installed.
    """
    if not path.lower().endswith(".mp4"):
        raise ValueError(f"Output path must end in .mp4, got: {path}")

    import imageio.v2 as imageio

    plotter = _build_plotter(
        aggregate, color=color, opacity=opacity, color_by=color_by,
        cmap=cmap, background=background, window_size=window_size,
    )
    writer = imageio.get_writer(
        path, fps=fps,
        format="FFMPEG",  # type: ignore[arg-type]  # imageio stub quirk
    )
    try:
        bounds = plotter.bounds
        center = np.array([
            (bounds[0] + bounds[1]) / 2.0,
            (bounds[2] + bounds[3]) / 2.0,
            (bounds[4] + bounds[5]) / 2.0,
        ])
        distance = _framing_distance(bounds)

        for i in range(n_frames):
            azimuth = 360.0 * i / n_frames
            # camera_position tuple + render() required in off_screen mode;
            # individual camera attribute assignments are ignored by pyvista
            plotter.camera_position = cast(
                Any, _orbit_camera_position(center, distance, azimuth, elevation)
            )
            plotter.render()
            frame = plotter.screenshot(return_img=True)
            if frame is not None:
                writer.append_data(frame)
    finally:
        writer.close()
        plotter.close()


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
    plotter = pv.Plotter(off_screen=True, window_size=list(window_size))
    try:
        plotter.set_background(background)  # type: ignore[arg-type]  # pyvista stub quirk
        mesh = _build_sphere_mesh(aggregate)
        plotter.add_mesh(mesh, color=color, opacity=opacity)
        plotter.camera_position = cast(Any, camera_position)
        plotter.screenshot(path)
    finally:
        plotter.close()


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
        n_frames: Total number of frames for a full 360 degree rotation.
        fps: Frames per second in the output video.
        elevation: Camera elevation angle in degrees.

    Raises:
        ValueError: If path does not end in .mp4.
        ImportError: If imageio is not installed.
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

    writer = imageio.get_writer(path, fps=fps)
    try:
        mesh = _build_sphere_mesh(aggregate)
        plotter = pv.Plotter(off_screen=True, window_size=list(window_size))
        plotter.set_background(background)  # type: ignore[arg-type]  # pyvista stub quirk
        plotter.add_mesh(mesh, color=color, opacity=opacity)

        center = np.mean(aggregate.positions, axis=0)
        rg = np.max(np.linalg.norm(aggregate.positions - center, axis=1))
        distance = max(rg * 4, 10.0)

        try:
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
                plotter.reset_camera_clipping_range()

                frame = plotter.screenshot(return_img=True)
                if frame is not None:
                    writer.append_data(frame)
        finally:
            plotter.close()
    finally:
        writer.close()
