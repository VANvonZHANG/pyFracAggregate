import pyFracAggregate as pfa
import os
import numpy as np

def main():
    # 1. Generate a larger fractal aggregate for better statistics
    print("Generating large fractal aggregate (N=500, Df=1.8)...")
    aggregate = pfa.generate(
        n_particles=500,
        df=1.8,
        kf=1.2,
        method='pca'
    )
    
    # 2. Perform automated analysis
    results = pfa.analyze(aggregate)
    print(f"\nAnalysis Results:")
    print(f"  Particles: {results['N']}")
    print(f"  Rg: {results['Rg']:.2f}")
    print(f"  Estimated Df: {results['Df_estimated']:.2f}")
    print(f"  R^2 of fit: {results['R2']:.4f}")
    
    # Create output directory
    os.makedirs("output", exist_ok=True)
    
    # 3. Test Plotting (save to file)
    print("\nGenerating and saving PCF plot...")
    save_path = "output/pcf_analysis.png"
    pfa.plot_pair_correlation(
        aggregate, 
        bins=100, 
        show_fit=True, 
        reference_df=1.8,
        save_path=save_path
    )
    
    # 4. Also export other formats to keep them updated
    print("Exporting to GLB and VTK...")
    pfa.export_glb(aggregate, "output/large_agg.glb")
    pfa.export_vtk(aggregate, "output/large_agg.vtk")

    print(f"\nVerification complete. Please check '{save_path}' for the analysis plot.")

if __name__ == "__main__":
    main()
