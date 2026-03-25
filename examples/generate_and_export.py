import pyFracAggregate as pfa
import os

def main():
    # 1. Generate a fractal aggregate
    print("Generating fractal aggregate...")
    aggregate = pfa.generate(
        n_particles=100,
        df=1.8,
        kf=1.2,
        method='pca'
    )
    
    # 2. Analyze properties
    results = pfa.analyze(aggregate)
    print(f"Generated aggregate with {results['N']} particles")
    print(f"Radius of Gyration (Rg): {results['Rg']:.2f}")
    
    # Create output directory if it doesn't exist
    os.makedirs("output", exist_ok=True)
    
    # 3. Export to GLB (using trimesh)
    print("Exporting to GLB (mesh.glb)...")
    pfa.export_glb(aggregate, "output/mesh.glb")
    
    # 4. Export to VTK (using pyvista)
    try:
        print("Exporting to VTK (points.vtk)...")
        pfa.export_vtk(aggregate, "output/points.vtk")
        
        print("Exporting to VTM (blocks.vtm)...")
        pfa.export_vtm(aggregate, "output/blocks.vtm")
    except ImportError as e:
        print(f"Note: {e}")

    print("\nAll tasks completed. Check the 'output' directory.")

if __name__ == "__main__":
    main()
