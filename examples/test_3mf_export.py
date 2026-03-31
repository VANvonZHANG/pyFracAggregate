import pyFracAggregate as pfa
import os

def main():
    print("Generating fractal aggregate...")
    aggregate = pfa.generate(
        n_particles=50,
        df=1.8,
        kf=1.2,
        method='pca'
    )
    
    # Create output directory
    os.makedirs("output", exist_ok=True)
    
    # Export to 3MF
    print("Exporting to 3MF (mesh.3mf)...")
    try:
        pfa.export_3mf(aggregate, "output/mesh.3mf")
        print("3MF export successful.")
    except Exception as e:
        print(f"3MF export failed: {e}")

if __name__ == "__main__":
    main()
