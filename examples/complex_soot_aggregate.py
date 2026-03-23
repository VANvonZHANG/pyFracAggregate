"""
Example: Generating and Analyzing a Complex Soot Aggregate
----------------------------------------------------------
This script demonstrates how to generate a fractal-like aggregate using 
the FracVAL method (multi-disperse), calculate its physical properties, 
and export it for visualization in Blender.

Author: Fan Zhang
Date: 2026-03-23
"""

import os
import numpy as np
import pyFracAggregate as pfa

def main():
    # 1. Configuration
    # N is the target number of particles
    N = 100            
    # DF is the target fractal dimension
    DF = 1.8           
    # KF is the target fractal prefactor
    KF = 1.3           
    
    # Define a log-normal size distribution for polydispersity
    # mean radius = 1.0, geometric standard deviation = 1.2
    size_dist = pfa.LognormalDistribution(mean=1.0, std=1.2)

    print(f"--- Generating Aggregate (N={N}, Df={DF}, Kf={KF}) ---")
    
    # 2. Generation using FracVAL algorithm (Advanced Hierarchical CCA)
    # The generation uses the FracVAL method for more accurate multi-disperse structures
    aggregate = pfa.generate(
        n_particles=N,
        df=DF,
        kf=KF,
        method='fracval',
        particle_dist=size_dist,
        overlap_tolerance=0.01
    )
    
    # 3. Analysis
    # Calculate basic morphological properties (Rg, CoM, N)
    stats = pfa.analyze(aggregate)
    rg = stats['Rg']
    com = stats['CoM']
    
    # Calculate pair correlation function C(r) for structural analysis
    r_centers, c_r = pfa.pair_correlation_function(aggregate, bins=50)
    
    print("\n--- Analysis Results ---")
    print(f"Radius of Gyration (Rg): {rg:.4f}")
    print(f"Center of Mass (CoM): {com}")
    print(f"Mean Particle Radius: {np.mean(aggregate.radii):.4f}")
    
    # 4. Export to Blender
    # This creates a Python script that can be run inside Blender's scripting tab
    output_dir = "output"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    blender_script_path = os.path.join(output_dir, "render_aggregate.py")
    pfa.export_to_blender_script(
        aggregate, 
        output_path=blender_script_path,
        object_name="Soot_Cluster_001",
        use_random_color=True
    )
    
    print(f"\nExample completed successfully.")
    print(f"You can find the Blender import script at: {blender_script_path}")

if __name__ == "__main__":
    main()
