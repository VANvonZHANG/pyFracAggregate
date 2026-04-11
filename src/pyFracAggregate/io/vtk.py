import numpy as np
import pyvista as pv
from pyFracAggregate.core.aggregate import Aggregate


def export_vtm(aggregate: Aggregate, path: str):
    """Export the aggregate as a VTM (MultiBlock) file.

    Each monomer is represented as a separate sphere block.
    """
    blocks = pv.MultiBlock()
    positions = aggregate.positions
    radii = aggregate.radii

    for i in range(len(positions)):
        sphere = pv.Sphere(radius=radii[i], center=positions[i])
        blocks.append(sphere)

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