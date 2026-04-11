from __future__ import annotations

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
    try:
        plotter.set_background(background)
        mesh = _build_sphere_mesh(aggregate)
        plotter.add_mesh(mesh, color=color, opacity=opacity)
        plotter.camera_position = camera_position
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

    mesh = _build_sphere_mesh(aggregate)
    plotter = pv.Plotter(off_screen=True, window_size=window_size)
    plotter.set_background(background)
    plotter.add_mesh(mesh, color=color, opacity=opacity)

    center = np.mean(aggregate.positions, axis=0)
    rg = np.max(np.linalg.norm(aggregate.positions - center, axis=1))
    distance = max(rg * 4, 10.0)

    writer = imageio.get_writer(path, fps=fps)

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
            writer.append_data(frame)
    finally:
        writer.close()
        plotter.close()
