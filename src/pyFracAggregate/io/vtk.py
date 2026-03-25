import numpy as np
import os
from ..core.aggregate import Aggregate

def export_vtm(aggregate: Aggregate, path: str):
    """
    Export the aggregate as a VTM (MultiBlock) file using pyvista.
    Each monomer is represented by a separate sphere block.
    
    Requires pyvista to be installed.
    
    Args:
        aggregate (Aggregate): The aggregate to export.
        path (str): The output file path (.vtm).
    """
    try:
        import pyvista as pv
    except ImportError:
        raise ImportError("pyvista is required for VTM export. Install it with: pip install pyvista")

    blocks = pv.MultiBlock()
    positions = aggregate.positions
    radii = aggregate.radii
    
    for i in range(len(positions)):
        sphere = pv.Sphere(radius=radii[i], center=positions[i])
        blocks.append(sphere)
        
    blocks.save(path)

def export_vtk(aggregate: Aggregate, path: str):
    """
    Export the aggregate as a VTK PolyData file representing points with radii.
    This is much more lightweight than VTM as it stores points instead of meshes.
    In ParaView, use the 'Glyph' filter with 'Sphere' type to visualize.
    
    Requires pyvista to be installed.
    
    Args:
        aggregate (Aggregate): The aggregate to export.
        path (str): The output file path (.vtk or .vtp).
    """
    try:
        import pyvista as pv
    except ImportError:
        raise ImportError("pyvista is required for VTK export. Install it with: pip install pyvista")

    positions = aggregate.positions
    radii = aggregate.radii
    masses = aggregate.masses
    
    # Create a point cloud
    point_cloud = pv.PolyData(positions)
    point_cloud["radius"] = radii
    point_cloud["mass"] = masses
    
    point_cloud.save(path)
