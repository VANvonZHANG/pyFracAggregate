import numpy as np
import datetime
from pyFracAggregate.core.aggregate import Aggregate

def export_to_blender_script(
    aggregate: Aggregate,
    output_path: str,
    object_name: str = "Fractal_Aggregate",
    use_random_color: bool = True
) -> None:
    """
    Exports the aggregate structure to a standalone Python script that can be 
    executed inside Blender's scripting tab.
    
    The generated script will create a collection and spheres for each particle 
    at the specified coordinates and radii.
    
    Args:
        aggregate (Aggregate): The fractal aggregate object to export.
        output_path (str): The path to save the generated .py script.
        object_name (str): The name prefix for the generated objects in Blender.
        use_random_color (bool): If True, assigns a random color to the aggregate's material.
    """
    positions = aggregate.positions
    radii = aggregate.radii
    
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    script_content = f"""# Blender Import Script for pyFracAggregate
# Generated on: {now}
# Number of particles: {len(positions)}

import bpy
import random

def create_aggregate():
    # Create a new collection for the aggregate
    collection_name = "{object_name}"
    if collection_name in bpy.data.collections:
        # If exists, link to the existing one or create a sub-collection
        collection = bpy.data.collections[collection_name]
    else:
        collection = bpy.data.collections.new(collection_name)
        bpy.context.scene.collection.children.link(collection)

    # Create a material
    mat = bpy.data.materials.new(name="{object_name}_Material")
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    principled = nodes.get("Principled BSDF")
    
    if principled:
        if {use_random_color}:
            principled.inputs[0].default_value = (random.random(), random.random(), random.random(), 1.0)
        else:
            principled.inputs[0].default_value = (0.1, 0.1, 0.1, 1.0) # Default dark soot-like color
        principled.inputs[7].default_value = 0.8 # Roughness

    # Sphere data
    particles = {positions.tolist()}
    radii = {radii.tolist()}

    # Batch create spheres
    for i, (pos, r) in enumerate(zip(particles, radii)):
        # Create a sphere mesh
        bpy.ops.mesh.primitive_uv_sphere_add(
            segments=32, 
            ring_count=16, 
            radius=r, 
            location=pos
        )
        obj = bpy.context.active_object
        obj.name = f"{object_name}_P{{i:04d}}"
        obj.data.materials.append(mat)
        
        # Move to our collection
        for old_col in obj.users_collection:
            old_col.objects.unlink(obj)
        collection.objects.link(obj)
        
        # Shade smooth
        bpy.ops.object.shade_smooth()

    print(f"Imported {{len(particles)}} particles into collection '{{collection_name}}'.")

if __name__ == "__main__":
    create_aggregate()
"""
    
    with open(output_path, 'w') as f:
        f.write(script_content)
    
    print(f"Blender script successfully exported to: {output_path}")
