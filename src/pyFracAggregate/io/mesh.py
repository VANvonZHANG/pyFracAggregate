import trimesh
import numpy as np
from ..core.aggregate import Aggregate

def export_glb(aggregate: Aggregate, path: str):
    """
    Export the aggregate as a GLB file using trimesh.
    Each monomer is represented by a sphere.
    
    Args:
        aggregate (Aggregate): The aggregate to export.
        path (str): The output file path.
    """
    scene = trimesh.Scene()
    
    positions = aggregate.positions
    radii = aggregate.radii
    
    for i in range(len(positions)):
        # Create a sphere for each monomer
        # Using a reasonable subdivision level for smoothness
        sphere = trimesh.creation.uv_sphere(radius=radii[i], count=[16, 16])
        
        # Move the sphere to its position
        translation = np.eye(4)
        translation[:3, 3] = positions[i]
        sphere.apply_transform(translation)
        
        # Add to scene
        scene.add_geometry(sphere, node_name=f"monomer_{i}")
        
    scene.export(path)

def export_3mf(aggregate: Aggregate, path: str):
    """
    Export the aggregate as a 3MF (3D Manufacturing Format) file using trimesh.
    Requires 'lxml' to be installed.
    
    Args:
        aggregate (Aggregate): The aggregate to export.
        path (str): The output file path (.3mf).
    """
    scene = trimesh.Scene()
    
    positions = aggregate.positions
    radii = aggregate.radii
    
    for i in range(len(positions)):
        sphere = trimesh.creation.uv_sphere(radius=radii[i], count=[16, 16])
        translation = np.eye(4)
        translation[:3, 3] = positions[i]
        sphere.apply_transform(translation)
        scene.add_geometry(sphere, node_name=f"monomer_{i}")
        
    scene.export(path, file_type='3mf')
