import numpy as np
import bpy
from pyFracAggregate.core.aggregate import Aggregate
import os

def build_aggregate_in_blender(
    aggregate: Aggregate,
    collection_name: str = "Fractal_Aggregate",
    use_random_color: bool = True
) -> None:
    """
    Builds the aggregate directly in the current Blender scene using low-level API 
    for maximum performance. Assumes `bpy` is available in the environment.
    
    Args:
        aggregate (Aggregate): The fractal aggregate object to build.
        collection_name (str): The name of the collection to store the particles.
        use_random_color (bool): If True, assigns a random color to the aggregate's material.
    """
    # 1. Setup Collection
    if collection_name in bpy.data.collections:
        collection = bpy.data.collections[collection_name]
    else:
        collection = bpy.data.collections.new(collection_name)
        bpy.context.scene.collection.children.link(collection)

    # 2. Setup Material
    mat = bpy.data.materials.new(name=f"{collection_name}_Material")
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    principled = nodes.get("Principled BSDF")
    
    if principled:
        if use_random_color:
            import random
            principled.inputs[0].default_value = (random.random(), random.random(), random.random(), 1.0)
        else:
            principled.inputs[0].default_value = (0.05, 0.05, 0.05, 1.0) # Default dark soot
        principled.inputs[7].default_value = 0.9 # Roughness

    # 3. Create a master sphere mesh
    # We use a primitive operator once, then duplicate its data for performance
    bpy.ops.mesh.primitive_uv_sphere_add(segments=16, ring_count=8, radius=1.0, location=(0, 0, 0))
    master_obj = bpy.context.active_object
    master_mesh = master_obj.data
    
    # Shade smooth the master mesh
    for poly in master_mesh.polygons:
        poly.use_smooth = True
        
    bpy.data.objects.remove(master_obj) # Remove the master object, keep the mesh data

    # 4. Batch create particles using low-level API
    positions = aggregate.positions
    radii = aggregate.radii
    
    for i, (pos, r) in enumerate(zip(positions, radii)):
        # Create new object sharing the master mesh
        obj = bpy.data.objects.new(f"{collection_name}_P{i:04d}", master_mesh)
        
        # Set transform
        obj.location = pos
        obj.scale = (r, r, r)
        
        # Assign material
        obj.data.materials.append(mat)
        
        # Link to collection
        collection.objects.link(obj)

    print(f"Successfully built {len(positions)} particles into collection '{collection_name}'.")


def save_blend_file(output_path: str) -> None:
    """
    Saves the current Blender scene state to a .blend file.
    
    Args:
        output_path (str): The path where the .blend file will be saved.
    """
    abs_path = os.path.abspath(output_path)
    bpy.ops.wm.save_as_mainfile(filepath=abs_path)
    print(f"Blender scene successfully saved to: {abs_path}")
