# Comprehensive Jupyter Notebook Example

Replace `examples/generate_and_export.py` and `examples/test_pcf_plot.py` with a single `examples/pyFracAggregate_demo.ipynb` that covers the full API.

## Audience

- New users needing a guided tutorial (Section 1-2)
- Researchers exploring methods, parameters, and analysis (Section 3-5)

Language: English for all Markdown cells and code comments.

## Structure

### Section 1: Introduction & Setup

Two cells:

1. **Markdown** — What pyFracAggregate is, what this notebook covers, table of contents.
2. **Code** — `import pyFracAggregate as pfa`, optional-dependency checks (pyvista, imageio), print `pfa.__version__`.

### Section 2: Core Workflow — End-to-End

One complete pipeline that a new user can run top-to-bottom:

1. **Markdown** — Explain we'll generate a PCA aggregate (N=100, Df=1.8), analyze, export all formats.
2. **Code** — `pfa.generate(n_particles=100, df=1.8, kf=1.2, method='pca')`, print particle count and radius range.
3. **Code** — `pfa.analyze(agg)`, print Rg, CoM, N, Df_estimated, R².
4. **Code** — Data export: `export_yaml`, `export_vtk`, `export_vtm`.
5. **Code** — Visualization export: `export_render` (PNG, camera_position='iso'), `export_rotation_video` (MP4, n_frames=72, fps=24).
6. **Markdown** — Recap of what was done.

### Section 3: Generation Methods — Comparing Four Algorithms

Show all generation methods, placement strategies, and particle distributions:

1. **Markdown** — Brief description of PCA, CCA, FracVAL, TDCCA algorithms and when to use each.
2. **Code** — Generate with all four methods (small N for speed), compare results (N, Rg) in a summary table.
3. **Code** — Placement strategies: `algebraic` vs `random`, same parameters, compare Rg.
4. **Code** — Particle distributions: `Monodisperse(1.0)` vs `LognormalDistribution(mean=1.0, std=0.3)`.
5. **Markdown** — Summary of method selection guidance.

### Section 4: Analysis & Visualization — Morphology and Fractal Dimension

Deeper analysis for research use:

1. **Markdown** — Analysis tools overview: morphology, pair correlation, fractal dimension estimation.
2. **Code** — Generate larger aggregate (N=500, Df=1.8) for statistical analysis.
3. **Code** — `radius_of_gyration()` and `center_of_mass()`, print results.
4. **Code** — `pair_correlation_function()` → `estimate_fractal_dimension()`, print Df_estimated and R², compare to input Df.
5. **Code** — `plot_pair_correlation()` with `show_fit=True`, `reference_df=1.8`, save PNG and display inline.
6. **Markdown** — Interpretation of PCF plot and fit quality.

### Section 5: Export & Rendering — Format Details

Detailed look at each export format:

1. **Markdown** — Format purposes: YAML (full snapshot), VTK (point cloud), VTM (multi-block), PNG (render), MP4 (rotation video).
2. **Code** — `export_yaml` with full params (`generation_params`, `analysis_results`), read back and display YAML structure.
3. **Code** — `export_vtk` and `export_vtm`, print file sizes.
4. **Code** — `export_render` with different `camera_position` values (`'iso'`, `'xy'`, `'xz'`), generate multi-angle renders.
5. **Code** — `export_rotation_video` with custom params (`n_frames=120`, `fps=30`, `elevation=20`).
6. **Markdown** — Export format selection guidance.

## File Changes

- **Create**: `examples/pyFracAggregate_demo.ipynb`
- **Delete**: `examples/generate_and_export.py`
- **Delete**: `examples/test_pcf_plot.py`
- **Clean up**: `examples/__pycache__/` (no longer needed)

## Design Decisions

- Single notebook replaces two .py files for a richer, interactive experience.
- Section 2 (Core Workflow) is self-contained — new users can stop there.
- Sections 3-5 build on Section 2 concepts for research exploration.
- N=100 for quick demos, N=500 for statistical analysis — balances runtime and quality.
- Optional dependencies (pyvista, imageio) are checked in Section 1 with graceful fallback messages.
