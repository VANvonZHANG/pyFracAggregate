"""
[Complete Example] Soot Aggregate Generation, Analysis, and Blender Rendering
----------------------------------------------------------------------------
This script demonstrates the full end-to-end pipeline of pyFracAggregate:
1.  **Generate** a complex soot aggregate using the FracVAL method.
2.  **Analyze** its morphology (Radius of Gyration, CoM).
3.  **Export** numerical data to a structured JSON file.
4.  **Build** the 3D geometry directly inside Blender's memory.
5.  **Configure** a high-quality rendering setup (Camera, Sun Light).
6.  **Save** the current scene as a `.blend` file for manual inspection.
7.  **Render** a high-resolution PNG image (Headless Mode supported).

To run this:
    export PYTHONPATH=src:$PYTHONPATH
    python examples/headless_render.py
"""

import sys
import os
import datetime

# Add local directories to path for imports
sys.path.append(os.path.abspath('examples'))

import pyFracAggregate as pfa
import blender_render_setup

def run_pipeline():
    # --- 1. Generation ---
    print(f"[{datetime.datetime.now():%H:%M:%S}] 1. Generating complex polydisperse aggregate (N=100)...")
    size_dist = pfa.LognormalDistribution(mean=1.0, std=1.2)
    aggregate = pfa.generate(
        n_particles=100, 
        df=1.8, 
        kf=1.3,
        method='fracval', 
        particle_dist=size_dist, 
        overlap_tolerance=0.01
    )

    # --- 2. Morphology Analysis ---
    print(f"[{datetime.datetime.now():%H:%M:%S}] 2. Performing morphology analysis...")
    stats = pfa.analyze(aggregate)
    print(f"   -> Radius of Gyration (Rg): {stats['Rg']:.4f}")

    # --- 3. Data Export ---
    output_dir = "output"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    json_path = os.path.join(output_dir, "aggregate_data.json")
    pfa.export_to_json(aggregate, json_path)
    print(f"   -> Numerical data saved to: {json_path}")

    # --- 4. Blender Scene Building ---
    print(f"[{datetime.datetime.now():%H:%M:%S}] 3. Building 3D scene in Blender (High Performance Mode)...")
    collection_name = "Soot_Cluster_Master"
    pfa.build_aggregate_in_blender(
        aggregate,
        collection_name=collection_name,
        use_random_color=True
    )
    
    # --- 5. Scene Setup (Camera/Lighting) ---
    print(f"[{datetime.datetime.now():%H:%M:%S}] 4. Configuring camera and lighting...")
    blender_render_setup.setup_lighting_and_camera(collection_name)
    
    # --- 6. Save .blend Project ---
    blend_path = os.path.join(output_dir, "soot_project.blend")
    pfa.save_blend_file(blend_path)
    print(f"   -> Blender project file saved to: {blend_path}")

    # --- 7. Rendering ---
    render_path = os.path.join(output_dir, "final_render.png")
    print(f"[{datetime.datetime.now():%H:%M:%S}] 5. Starting high-quality render...")
    blender_render_setup.render_scene(render_path)
    print(f"   -> Final render saved to: {render_path}")

    print(f"\n[{datetime.datetime.now():%H:%M:%S}] Full pipeline completed successfully!")

if __name__ == "__main__":
    run_pipeline()
