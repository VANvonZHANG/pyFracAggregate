"""
Example: Generating and Exporting a Complex Soot Aggregate
----------------------------------------------------------
This script demonstrates how to generate a fractal-like aggregate using 
the FracVAL method (multi-disperse), calculate its physical properties, 
and export pure data (JSON) and a basic Blender loading script.

Author: Fan Zhang
"""

import os
import numpy as np
import pyFracAggregate as pfa

def main():
    # 1. Configuration
    N = 100            # Total number of particles
    DF = 1.8           # Target fractal dimension
    KF = 1.3           # Fractal prefactor
    
    # Define a log-normal size distribution for polydispersity
    size_dist = pfa.LognormalDistribution(mean=1.0, std=1.2)

    print(f"--- Generating Aggregate (N={N}, Df={DF}, Kf={KF}) ---")
    
    # 2. Generation using FracVAL algorithm
    aggregate = pfa.generate(
        n_particles=N,
        df=DF,
        kf=KF,
        method='fracval',
        particle_dist=size_dist,
        overlap_tolerance=0.01
    )
    
    # 3. Analysis
    stats = pfa.analyze(aggregate)
    print("\n--- Analysis Results ---")
    print(f"Radius of Gyration (Rg): {stats['Rg']:.4f}")
    
    # 4. Export Pure Data
    output_dir = "output"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    blender_script_path = os.path.join(output_dir, "load_aggregate.py")
    json_path = os.path.join(output_dir, "aggregate_data.json")
    
    # Export basic Blender script (only loads meshes, no rendering logic)
    pfa.export_to_blender_script(
        aggregate, 
        output_path=blender_script_path,
        object_name="Soot_Cluster_Final",
        use_random_color=True
    )
    
    # Export pure JSON data
    pfa.export_to_json(aggregate, json_path)
    
    print(f"\nExample completed.")
    print(f"Load script: {blender_script_path}")
    print(f"JSON Data: {json_path}")
    print("\nTo visualize in Blender:")
    print("1. Open Blender -> Scripting Tab -> Run `load_aggregate.py`")
    print("2. Run the `examples/blender_render_setup.py` script to setup lights and camera.")

if __name__ == "__main__":
    main()
