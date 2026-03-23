"""
Headless Rendering Script using `bpy` directly in Python.
"""
import sys
import os

# Append the required directories to sys.path
sys.path.append(os.path.abspath('examples'))
sys.path.append(os.path.abspath('output'))

import bpy
import load_aggregate
import blender_render_setup

def main():
    # 1. Load aggregate geometries
    print("Loading aggregate into Blender scene...")
    load_aggregate.load_aggregate()
    
    # 2. Setup lighting and camera
    print("Setting up lighting and camera...")
    blender_render_setup.setup_lighting_and_camera("Soot_Cluster_Final")
    
    # 3. Render
    print("Starting render...")
    blender_render_setup.render_scene("output/my_render.png")
    print("Render complete!")

if __name__ == "__main__":
    main()
