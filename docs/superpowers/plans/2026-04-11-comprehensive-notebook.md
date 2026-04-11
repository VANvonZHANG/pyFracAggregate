# Comprehensive Jupyter Notebook Example — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace two `.py` example files with a single structured Jupyter notebook covering the full pyFracAggregate API.

**Architecture:** Single `.ipynb` file with 5 sections progressing from core workflow (new users) to detailed API exploration (researchers). All cells contain runnable code with Markdown explanations. Old `.py` examples are deleted.

**Tech Stack:** Jupyter Notebook, pyFracAggregate, pyvista (optional), imageio[ffmpeg] (optional), matplotlib

---

## File Structure

| Action | File | Responsibility |
|--------|------|----------------|
| Create | `examples/pyFracAggregate_demo.ipynb` | Single comprehensive demo notebook |
| Delete | `examples/generate_and_export.py` | Old example (replaced by notebook) |
| Delete | `examples/test_pcf_plot.py` | Old example (replaced by notebook) |

---

### Task 1: Create notebook with Section 1 (Introduction & Setup)

**Files:**
- Create: `examples/pyFracAggregate_demo.ipynb`

- [ ] **Step 1: Create the notebook file**

Create `examples/pyFracAggregate_demo.ipynb` with the following content — a notebook with 3 initial cells (metadata, markdown intro, setup code):

Cell 0 (markdown):
```markdown
# pyFracAggregate — Comprehensive Demo

This notebook demonstrates the full capabilities of **pyFracAggregate**, a Python library for generating and analyzing fractal aggregates.

**Contents:**
1. [Introduction & Setup](#Introduction-&-Setup)
2. [Core Workflow — End-to-End](#Core-Workflow)
3. [Generation Methods — Comparing Four Algorithms](#Generation-Methods)
4. [Analysis & Visualization — Morphology and Fractal Dimension](#Analysis-&-Visualization)
5. [Export & Rendering — Format Details](#Export-&-Rendering)
```

Cell 1 (code):
```python
import pyFracAggregate as pfa
import numpy as np
import os

print(f"pyFracAggregate version: {pfa.__version__}")

# Check optional visualization dependencies
try:
    import pyvista
    print(f"pyvista {pyvista.__version__} — 3D rendering available")
except ImportError:
    print("pyvista not installed — 3D rendering (export_render, export_rotation_video) will not work")

try:
    import imageio
    print(f"imageio {imageio.__version__} — video export available")
except ImportError:
    print("imageio not installed — export_rotation_video will not work")
```

Cell 2 (markdown):
```markdown
---
```

- [ ] **Step 2: Verify notebook loads**

Run: `python -c "import json; nb = json.load(open('examples/pyFracAggregate_demo.ipynb')); print(f'{len(nb[\"cells\"])} cells')"`
Expected: `3 cells`

- [ ] **Step 3: Commit**

```bash
git add examples/pyFracAggregate_demo.ipynb
git commit -m "feat: add Jupyter notebook example — Section 1 (Introduction & Setup)"
```

---

### Task 2: Add Section 2 (Core Workflow)

**Files:**
- Modify: `examples/pyFracAggregate_demo.ipynb` — append 5 cells after existing content

- [ ] **Step 1: Append Section 2 cells to the notebook**

Append these 5 cells to the notebook:

Cell 3 (markdown):
```markdown
## Core Workflow — End-to-End

This section walks through the complete pipeline: **generate → analyze → export**. By the end, you'll have a full set of output files from a single PCA aggregate.
```

Cell 4 (code):
```python
# Generate a fractal aggregate using Particle-Cluster Aggregation (PCA)
agg = pfa.generate(
    n_particles=100,
    df=1.8,       # Fractal dimension
    kf=1.2,       # Fractal prefactor
    method='pca'
)

print(f"Particles: {agg.current_size}")
print(f"Radius range: [{agg.radii.min():.3f}, {agg.radii.max():.3f}]")
print(f"Position bounds: x[{agg.positions[:,0].min():.1f}, {agg.positions[:,0].max():.1f}]"
      f" y[{agg.positions[:,1].min():.1f}, {agg.positions[:,1].max():.1f}]"
      f" z[{agg.positions[:,2].min():.1f}, {agg.positions[:,2].max():.1f}]")
```

Cell 5 (code):
```python
# Analyze morphological properties
results = pfa.analyze(agg)

print("Analysis Results:")
print(f"  Number of particles (N): {results['N']}")
print(f"  Radius of gyration (Rg): {results['Rg']:.2f}")
print(f"  Center of mass (CoM):    [{results['CoM'][0]:.2f}, {results['CoM'][1]:.2f}, {results['CoM'][2]:.2f}]")
print(f"  Estimated Df:            {results['Df_estimated']:.2f} (input: 1.80)")
print(f"  Fit quality (R²):        {results['R2']:.4f}")
```

Cell 6 (code):
```python
os.makedirs("output", exist_ok=True)

# Export to YAML — full snapshot with metadata
pfa.export_yaml(
    agg, "output/aggregate.yaml",
    generation_params={"method": "pca", "n_particles": 100, "df": 1.8, "kf": 1.2},
    analysis_results=results,
)
print("✓ YAML exported")

# Export to VTK — point cloud
pfa.export_vtk(agg, "output/points.vtk")
print("✓ VTK point cloud exported")

# Export to VTM — MultiBlock (spheres as structured data)
pfa.export_vtm(agg, "output/blocks.vtm")
print("✓ VTM MultiBlock exported")
```

Cell 7 (code):
```python
# Render a 3D screenshot
pfa.export_render(agg, "output/aggregate_render.png", camera_position="iso")
print("✓ Rendered PNG saved")

# Generate a 360° rotation video
pfa.export_rotation_video(agg, "output/aggregate_rotation.mp4", n_frames=72, fps=24)
print("✓ Rotation video saved")

print("\nAll exports complete. Check the 'output' directory.")
```

- [ ] **Step 2: Verify cell count**

Run: `python -c "import json; nb = json.load(open('examples/pyFracAggregate_demo.ipynb')); print(f'{len(nb[\"cells\"])} cells')"`
Expected: `8 cells`

- [ ] **Step 3: Commit**

```bash
git add examples/pyFracAggregate_demo.ipynb
git commit -m "feat: add Core Workflow section to notebook (generate→analyze→export)"
```

---

### Task 3: Add Section 3 (Generation Methods)

**Files:**
- Modify: `examples/pyFracAggregate_demo.ipynb` — append 5 cells

- [ ] **Step 1: Append Section 3 cells**

Cell 8 (markdown):
```markdown
---

## Generation Methods — Comparing Four Algorithms

pyFracAggregate provides four generation algorithms:

| Method | Key | Description |
|--------|-----|-------------|
| **PCA** | `'pca'` | Particle-Cluster Aggregation — adds one particle at a time |
| **CCA** | `'cca'` | Cluster-Cluster Aggregation — merges sub-clusters |
| **FracVAL** | `'fracval''` | FracVAL algorithm — tunable fractal dimension |
| **TDCCA** | `'tdcca'` | Thouy & Jullien (2004) — cluster-cluster with strict fractal scaling |
```

Cell 9 (code):
```python
# Compare all four generation methods with the same fractal parameters
methods = ['pca', 'cca', 'fracval', 'tdcca']
aggregates = {}

for method in methods:
    print(f"Generating with {method.upper()}...")
    aggregates[method] = pfa.generate(
        n_particles=50,   # Small for speed
        df=1.8,
        kf=1.2,
        method=method,
    )

print("\n--- Comparison ---")
print(f"{'Method':<10} {'N':>5} {'Rg':>8}")
print("-" * 26)
for method, agg_m in aggregates.items():
    rg = pfa.radius_of_gyration(agg_m)
    print(f"{method:<10} {agg_m.current_size:>5} {rg:>8.2f}")
```

Cell 10 (markdown):
```markdown
### Placement Strategies

Two placement strategies control how particles are positioned during generation:

- **`'algebraic'`** (default) — Analytical touching-point computation with Monte Carlo fallback. More accurate.
- **`'random'`** — Pure Monte Carlo sampling. Faster for large aggregates.
```

Cell 11 (code):
```python
# Compare placement strategies
for placement in ['algebraic', 'random']:
    agg_p = pfa.generate(n_particles=50, df=1.8, kf=1.2, method='pca', placement=placement)
    rg = pfa.radius_of_gyration(agg_p)
    print(f"placement={placement:<10} → N={agg_p.current_size}, Rg={rg:.2f}")
```

Cell 12 (markdown):
```markdown
### Particle Distributions

By default, particles have uniform radius (`Monodisperse`). You can also use a `LognormalDistribution` for polydisperse systems:
```

Cell 13 (code):
```python
# Monodisperse vs Lognormal distribution
mono = pfa.Monodisperse(radius=1.0)
lognorm = pfa.LognormalDistribution(mean=1.0, std=0.3)

for name, dist in [("Monodisperse(1.0)", mono), ("Lognormal(1.0, 0.3)", lognorm)]:
    agg_d = pfa.generate(n_particles=50, df=1.8, kf=1.2, method='pca', particle_dist=dist)
    print(f"{name:<25} → N={agg_d.current_size}, Rg={pfa.radius_of_gyration(agg_d):.2f}")
    print(f"  Radii: min={agg_d.radii.min():.3f}, max={agg_d.radii.max():.3f}, mean={agg_d.radii.mean():.3f}")
```

- [ ] **Step 2: Verify cell count**

Run: `python -c "import json; nb = json.load(open('examples/pyFracAggregate_demo.ipynb')); print(f'{len(nb[\"cells\"])} cells')"`
Expected: `14 cells`

- [ ] **Step 3: Commit**

```bash
git add examples/pyFracAggregate_demo.ipynb
git commit -m "feat: add Generation Methods section — all algorithms, placements, distributions"
```

---

### Task 4: Add Section 4 (Analysis & Visualization)

**Files:**
- Modify: `examples/pyFracAggregate_demo.ipynb` — append 5 cells

- [ ] **Step 1: Append Section 4 cells**

Cell 14 (markdown):
```markdown
---

## Analysis & Visualization — Morphology and Fractal Dimension

This section takes a deeper look at aggregate analysis: morphological properties, the pair correlation function (PCF), and fractal dimension estimation.
```

Cell 15 (code):
```python
# Generate a larger aggregate for better statistical analysis
agg_large = pfa.generate(n_particles=500, df=1.8, kf=1.2, method='pca')
print(f"Generated: N={agg_large.current_size}")
```

Cell 16 (code):
```python
# Morphological analysis
rg = pfa.radius_of_gyration(agg_large)
com = pfa.center_of_mass(agg_large)

print(f"Radius of gyration (Rg): {rg:.2f}")
print(f"Center of mass (CoM):    [{com[0]:.2f}, {com[1]:.2f}, {com[2]:.2f}]")
```

Cell 17 (code):
```python
# Pair correlation function and fractal dimension estimation
r_centers, c_r = pfa.pair_correlation_function(agg_large, bins=100)
df_est, r2, fit_info = pfa.estimate_fractal_dimension(
    r_centers, c_r,
    r_min=np.mean(agg_large.radii),
    r_max=rg,
)

print(f"Input Df:         1.80")
print(f"Estimated Df:     {df_est:.2f}")
print(f"Fit R²:           {r2:.4f}")
print(f"Fit range:        r ∈ [{fit_info.get('r_min', 'N/A'):.2f}, {fit_info.get('r_max', 'N/A'):.2f}]")
```

Cell 18 (code):
```python
# Plot the pair correlation function with fractal fit
pfa.plot_pair_correlation(
    agg_large,
    bins=100,
    show_fit=True,
    reference_df=1.8,
    save_path="output/pcf_analysis.png",
)
print("PCF plot saved to output/pcf_analysis.png")
```

Cell 19 (markdown):
```markdown
The PCF plot shows:
- **Blue dots**: Measured pair correlation function C(r)
- **Red dashed line**: Power-law fit in the fractal regime — slope = Df − 3
- **Green dotted line**: Reference slope for the input Df

A good fit (R² > 0.95) confirms the aggregate exhibits the expected fractal scaling.
```

- [ ] **Step 2: Verify cell count**

Run: `python -c "import json; nb = json.load(open('examples/pyFracAggregate_demo.ipynb')); print(f'{len(nb[\"cells\"])} cells')"`
Expected: `20 cells`

- [ ] **Step 3: Commit**

```bash
git add examples/pyFracAggregate_demo.ipynb
git commit -m "feat: add Analysis & Visualization section — morphology, PCF, Df estimation"
```

---

### Task 5: Add Section 5 (Export & Rendering)

**Files:**
- Modify: `examples/pyFracAggregate_demo.ipynb` — append 5 cells

- [ ] **Step 1: Append Section 5 cells**

Cell 20 (markdown):
```markdown
---

## Export & Rendering — Format Details

pyFracAggregate supports five export formats:

| Format | Function | Description |
|--------|----------|-------------|
| YAML | `export_yaml()` | Full snapshot: data + generation params + analysis results |
| VTK | `export_vtk()` | Point cloud — positions and radii as scalar field |
| VTM | `export_vtm()` | MultiBlock dataset — spheres as structured mesh blocks |
| PNG | `export_render()` | 3D rendered screenshot with configurable camera |
| MP4 | `export_rotation_video()` | 360° rotation animation |
```

Cell 21 (code):
```python
# YAML export — full snapshot with metadata
pfa.export_yaml(
    agg_large, "output/full_aggregate.yaml",
    generation_params={"method": "pca", "n_particles": 500, "df": 1.8, "kf": 1.2},
    analysis_results=pfa.analyze(agg_large),
)

# Read back and inspect structure
import yaml
with open("output/full_aggregate.yaml") as f:
    data = yaml.safe_load(f)

print("YAML top-level keys:", list(data.keys()))
print(f"  Particles: {len(data['data']['positions'])}")
if 'generation_params' in data:
    print(f"  Generation params: {data['generation_params']}")
if 'analysis' in data:
    print(f"  Analysis keys: {list(data['analysis'].keys())}")
```

Cell 22 (code):
```python
# VTK and VTM export — compare file sizes
pfa.export_vtk(agg_large, "output/points_500.vtk")
pfa.export_vtm(agg_large, "output/blocks_500.vtm")

vtk_size = os.path.getsize("output/points_500.vtk")
vtm_size = os.path.getsize("output/blocks_500.vtm")

print(f"VTK (point cloud): {vtk_size / 1024:.1f} KB")
print(f"VTM (MultiBlock):  {vtm_size / 1024:.1f} KB")
```

Cell 23 (code):
```python
# Multi-angle rendering with different camera positions
cameras = {
    'iso': 'Isometric (default)',
    'xy': 'XY plane — top-down view',
    'xz': 'XZ plane — side view',
}

for cam, desc in cameras.items():
    path = f"output/render_{cam}.png"
    pfa.export_render(agg_large, path, camera_position=cam)
    print(f"  {cam}: {desc} → {path}")
```

Cell 24 (code):
```python
# Custom rotation video — higher quality settings
pfa.export_rotation_video(
    agg_large,
    "output/aggregate_rotation_hq.mp4",
    n_frames=120,
    fps=30,
    elevation=20,
)
print("High-quality rotation video saved (120 frames, 30 fps, elevation=20°)")
```

Cell 25 (markdown):
```markdown
### Export Format Selection Guide

- **YAML** — For reproducibility: store the complete state to reload later.
- **VTK** — For lightweight point cloud visualization in ParaView or similar tools.
- **VTM** — For full 3D sphere representation (larger files, richer visualization).
- **PNG** — For static figures in papers and presentations.
- **MP4** — For supplementary materials and presentations.

**Tip:** Use VTK for quick inspection and VTM when you need actual sphere geometries.
```

- [ ] **Step 2: Verify cell count**

Run: `python -c "import json; nb = json.load(open('examples/pyFracAggregate_demo.ipynb')); print(f'{len(nb[\"cells\"])} cells')"`
Expected: `26 cells`

- [ ] **Step 3: Commit**

```bash
git add examples/pyFracAggregate_demo.ipynb
git commit -m "feat: add Export & Rendering section — all formats with parameter details"
```

---

### Task 6: Delete old examples and verify

**Files:**
- Delete: `examples/generate_and_export.py`
- Delete: `examples/test_pcf_plot.py`

- [ ] **Step 1: Delete old example files**

```bash
git rm examples/generate_and_export.py examples/test_pcf_plot.py
```

- [ ] **Step 2: Verify notebook structure**

Run:
```bash
python -c "
import json
nb = json.load(open('examples/pyFracAggregate_demo.ipynb'))
md_cells = sum(1 for c in nb['cells'] if c['cell_type'] == 'markdown')
code_cells = sum(1 for c in nb['cells'] if c['cell_type'] == 'code')
print(f'Total: {len(nb[\"cells\"])} cells ({md_cells} markdown, {code_cells} code)')
assert len(nb['cells']) == 26, f'Expected 26 cells, got {len(nb[\"cells\"])}'
print('OK')
"
```
Expected: `Total: 26 cells (13 markdown, 13 code)` then `OK`

- [ ] **Step 3: Run ruff on notebook code cells**

Run: `python -c "
import json, subprocess, tempfile, os
nb = json.load(open('examples/pyFracAggregate_demo.ipynb'))
code = '\n'.join(''.join(c['source']) for c in nb['cells'] if c['cell_type'] == 'code')
with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
    f.write(code)
    tmp = f.name
result = subprocess.run(['ruff', 'check', tmp], capture_output=True, text=True)
os.unlink(tmp)
if result.returncode != 0:
    print(result.stdout)
    print(result.stderr)
else:
    print('Ruff: all clean')
"`
Expected: `Ruff: all clean`

- [ ] **Step 4: Commit**

```bash
git add -A
git commit -m "chore: replace .py examples with comprehensive Jupyter notebook"
```
