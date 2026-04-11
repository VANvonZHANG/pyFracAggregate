from __future__ import annotations

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
