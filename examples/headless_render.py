"""
Headless Rendering Script using `bpy` directly in Python.
This script demonstrates an end-to-end pipeline:
1. Generate an aggregate using pyFracAggregate.
2. Build the geometry directly in the Blender background instance.
3. Automatically set up lighting and camera bounds.
4. Render the scene to a PNG image.
"""
import sys
import os

# Append the required directories to sys.path
sys.path.append(os.path.abspath('examples'))

import pyFracAggregate as pfa
import blender_render_setup

def main():
    # 1. Generate Aggregate
    print("Generating aggregate...")
    size_dist = pfa.LognormalDistribution(mean=1.0, std=1.2)
    aggregate = pfa.generate(
        n_particles=100, df=1.8, kf=1.3,
        method='fracval', particle_dist=size_dist, overlap_tolerance=0.01
    )

    # 2. Build in Blender
    print("Building aggregate into Blender scene...")
    collection_name = "Soot_Cluster_Final"
    pfa.build_aggregate_in_blender(
        aggregate,
        collection_name=collection_name,
        use_random_color=True
    )
    
    # 3. Setup lighting and camera
    print("Setting up lighting and camera...")
    blender_render_setup.setup_lighting_and_camera(collection_name)
    
    # 4. Render
    output_dir = "output"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    render_path = os.path.join(output_dir, "my_render.png")
    print("Starting render...")
    blender_render_setup.render_scene(render_path)
    print("Render complete!")

if __name__ == "__main__":
    main()
