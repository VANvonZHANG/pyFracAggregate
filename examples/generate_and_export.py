import pyFracAggregate as pfa
import os

def main():
    # 1. Generate a fractal aggregate
    print("Generating fractal aggregate...")
    agg = pfa.generate(
        n_particles=100,
        df=1.8,
        kf=1.2,
        method='pca'
    )

    # 2. Analyze properties
    results = pfa.analyze(agg)
    print(f"Generated aggregate with {results['N']} particles")
    print(f"Radius of Gyration (Rg): {results['Rg']:.2f}")

    os.makedirs("output", exist_ok=True)

    # 3. Export to YAML
    print("Exporting to YAML...")
    pfa.export_yaml(
        agg, "output/aggregate.yaml",
        generation_params={"method": "pca", "n_particles": 100, "df": 1.8, "kf": 1.2},
        analysis_results=results,
    )

    # 4. Export to VTK and VTM
    print("Exporting to VTK...")
    pfa.export_vtk(agg, "output/points.vtk")
    print("Exporting to VTM...")
    pfa.export_vtm(agg, "output/blocks.vtm")

    # 5. Visualization export
    print("Rendering 3D screenshot...")
    pfa.export_render(agg, "output/aggregate_render.png", camera_position="iso")
    print("Generating rotation video...")
    pfa.export_rotation_video(agg, "output/aggregate_rotation.mp4", n_frames=72, fps=24)

    print("\nAll tasks completed. Check the 'output' directory.")

if __name__ == "__main__":
    main()
