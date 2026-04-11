# IO Simplification Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Simplify export to YAML + VTK + VTM only; remove trimesh/lxml, promote pyvista to required dependency.

**Architecture:** Rewrite `io/data.py` from JSON to YAML with generation params and analysis results. Delete `io/mesh.py`. Update `vtk.py` to remove optional-import guards. Update `__init__.py` public API and `pyproject.toml` dependencies.

**Tech Stack:** PyYAML, pyvista, numpy

---

### Task 1: Rewrite `io/data.py` — JSON → YAML

**Files:**
- Modify: `src/pyFracAggregate/io/data.py`
- Test: `tests/test_io/test_data.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_io/test_data.py`:

```python
import os
import yaml
import numpy as np
import pytest
from pyFracAggregate.core.aggregate import Aggregate
from pyFracAggregate.io.data import export_yaml


def _make_aggregate(n=3):
    agg = Aggregate(max_particles=n)
    agg.add_particle(0.0, 0.0, 0.0, 1.0, 1.0)
    agg.add_particle(2.0, 0.0, 0.0, 1.0, 1.0)
    if n >= 3:
        agg.add_particle(4.0, 0.0, 0.0, 1.0, 1.0)
    return agg


def test_export_yaml_basic(tmp_path):
    agg = _make_aggregate(3)
    path = str(tmp_path / "basic.yaml")
    export_yaml(agg, path)

    with open(path) as f:
        data = yaml.safe_load(f)

    assert "aggregate" in data
    assert data["aggregate"]["n_particles"] == 3
    assert len(data["aggregate"]["positions"]) == 3
    assert isinstance(data["aggregate"]["positions"][0], list)
    assert data["aggregate"]["density"] == 1.0
    assert data["aggregate"]["length_unit"] == "nm"


def test_export_yaml_with_generation_params(tmp_path):
    agg = _make_aggregate(2)
    params = {"method": "pca", "n_particles": 2, "df": 1.8, "kf": 1.2, "placement": "algebraic"}
    path = str(tmp_path / "with_params.yaml")
    export_yaml(agg, path, generation_params=params)

    with open(path) as f:
        data = yaml.safe_load(f)

    assert data["generation"]["method"] == "pca"
    assert data["generation"]["df"] == 1.8
    assert "aggregate" in data
    assert "analysis" not in data


def test_export_yaml_with_analysis_results(tmp_path):
    agg = _make_aggregate(2)
    analysis = {"Rg": 5.0, "center_of_mass": [1.0, 0.0, 0.0], "fractal_dimension": 1.8}
    path = str(tmp_path / "with_analysis.yaml")
    export_yaml(agg, path, analysis_results=analysis)

    with open(path) as f:
        data = yaml.safe_load(f)

    assert data["analysis"]["Rg"] == 5.0
    assert data["analysis"]["center_of_mass"] == [1.0, 0.0, 0.0]
    assert "aggregate" in data
    assert "generation" not in data


def test_export_yaml_all_sections(tmp_path):
    agg = _make_aggregate(2)
    params = {"method": "cca", "n_particles": 2, "df": 2.0, "kf": 1.0}
    analysis = {"Rg": 3.0}
    path = str(tmp_path / "all.yaml")
    export_yaml(agg, path, generation_params=params, analysis_results=analysis)

    with open(path) as f:
        data = yaml.safe_load(f)

    assert "generation" in data
    assert "aggregate" in data
    assert "analysis" in data


def test_export_yaml_numpy_arrays_converted(tmp_path):
    """NumPy types must be serializable (no numpy.float64 in output)."""
    agg = _make_aggregate(2)
    path = str(tmp_path / "numpy_safe.yaml")
    # Should not raise
    export_yaml(agg, path)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_io/test_data.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'pyFracAggregate.io.data'` or `ImportError: cannot import name 'export_yaml'`

- [ ] **Step 3: Write minimal implementation**

Replace `src/pyFracAggregate/io/data.py` with:

```python
import numpy as np
import yaml
from pyFracAggregate.core.aggregate import Aggregate


def _to_native(obj):
    """Recursively convert NumPy types to native Python types for YAML serialization."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, dict):
        return {k: _to_native(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_native(v) for v in obj]
    return obj


def export_yaml(
    aggregate: Aggregate,
    output_path: str,
    *,
    generation_params: dict | None = None,
    analysis_results: dict | None = None,
) -> None:
    """Export aggregate to YAML with optional generation params and analysis results.

    Args:
        aggregate: The fractal aggregate object to export.
        output_path: Path to save the YAML file.
        generation_params: Optional dict of generation parameters (method, df, kf, etc.).
        analysis_results: Optional dict of analysis results (Rg, center_of_mass, etc.).
    """
    data = {}

    if generation_params is not None:
        data["generation"] = _to_native(generation_params)

    data["aggregate"] = _to_native({
        "n_particles": aggregate.current_size,
        "length_unit": aggregate.length_unit,
        "mass_unit": aggregate.mass_unit,
        "density": aggregate.density,
        "positions": aggregate.positions,
        "radii": aggregate.radii,
        "masses": aggregate.masses,
    })

    if analysis_results is not None:
        data["analysis"] = _to_native(analysis_results)

    with open(output_path, "w") as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_io/test_data.py -v`
Expected: All 5 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/pyFracAggregate/io/data.py tests/test_io/test_data.py
git commit -m "feat: replace JSON export with YAML export (data + params + analysis)"
```

---

### Task 2: Update `vtk.py` — remove optional-import guards

**Files:**
- Modify: `src/pyFracAggregate/io/vtk.py`

- [ ] **Step 1: Update vtk.py**

Replace the two `try/except ImportError` blocks with a top-level import. The file should become:

```python
import numpy as np
import pyvista as pv
from pyFracAggregate.core.aggregate import Aggregate


def export_vtm(aggregate: Aggregate, path: str):
    """Export the aggregate as a VTM (MultiBlock) file.

    Each monomer is represented as a separate sphere block.
    """
    blocks = pv.MultiBlock()
    positions = aggregate.positions
    radii = aggregate.radii

    for i in range(len(positions)):
        sphere = pv.Sphere(radius=radii[i], center=positions[i])
        blocks.append(sphere)

    blocks.save(path)


def export_vtk(aggregate: Aggregate, path: str):
    """Export the aggregate as a VTK PolyData file (point cloud with attributes).

    In ParaView, use the 'Glyph' filter with 'Sphere' type to visualize.
    """
    positions = aggregate.positions
    radii = aggregate.radii
    masses = aggregate.masses

    point_cloud = pv.PolyData(positions)
    point_cloud["radius"] = radii
    point_cloud["mass"] = masses

    point_cloud.save(path)
```

- [ ] **Step 2: Run existing tests to verify nothing breaks**

Run: `pytest tests/ -v`
Expected: All existing tests PASS (no test currently imports vtk.py directly)

- [ ] **Step 3: Commit**

```bash
git add src/pyFracAggregate/io/vtk.py
git commit -m "refactor: remove optional-import guards from vtk.py (pyvista is now required)"
```

---

### Task 3: Delete `mesh.py` and update public API

**Files:**
- Delete: `src/pyFracAggregate/io/mesh.py`
- Modify: `src/pyFracAggregate/__init__.py`

- [ ] **Step 1: Delete mesh.py**

```bash
rm src/pyFracAggregate/io/mesh.py
```

- [ ] **Step 2: Update `__init__.py`**

In `src/pyFracAggregate/__init__.py`:

1. Replace line 11 (`from pyFracAggregate.io.mesh import export_glb, export_3mf`) with:
```python
from pyFracAggregate.io.data import export_yaml
```

2. Replace line 13 (`from pyFracAggregate.io.data import export_to_json`) with nothing (delete the line).

3. In `__all__`, replace:
```python
    "export_glb",
    "export_3mf",
    "export_vtm",
    "export_vtk",
    "export_to_json",
```
with:
```python
    "export_yaml",
    "export_vtm",
    "export_vtk",
```

- [ ] **Step 3: Run all tests**

Run: `pytest tests/ -v`
Expected: All PASS

- [ ] **Step 4: Commit**

```bash
git add src/pyFracAggregate/io/mesh.py src/pyFracAggregate/__init__.py
git commit -m "refactor: remove mesh.py (GLB/3MF), update public API exports"
```

---

### Task 4: Update dependencies and examples

**Files:**
- Modify: `pyproject.toml`
- Modify: `examples/generate_and_export.py`

- [ ] **Step 1: Update `pyproject.toml`**

Replace the dependencies section:

```toml
dependencies = [
    "numpy>=1.21.0",
    "scipy>=1.7.0",
    "mathutils>=3.0.0",
    "pyvista",
    "pyyaml",
]
```

Remove the `[project.optional-dependencies] science` entry entirely. Keep `plot` and `dev`:

```toml
[project.optional-dependencies]
plot = [
    "matplotlib>=3.5.0"
]
dev = [
    "pytest",
    "ruff",
    "mypy"
]
```

- [ ] **Step 2: Update `examples/generate_and_export.py`**

Replace with:

```python
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

    print("\nAll tasks completed. Check the 'output' directory.")

if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Commit**

```bash
git add pyproject.toml examples/generate_and_export.py
git commit -m "chore: update dependencies (remove trimesh/lxml, add pyvista/pyyaml) and example"
```

---

### Task 5: Update CLAUDE.md

**Files:**
- Modify: `CLAUDE.md`

- [ ] **Step 1: Update CLAUDE.md**

In the Build & Development Commands section, remove:
- `pip install -e ".[science]"` line

In the Architecture → IO section, replace:
```
### IO (`src/pyFracAggregate/io/`)
- **mesh.py**: `export_glb()` (glTF binary) and `export_3mf()` via trimesh
- **vtk.py**: `export_vtm()` and `export_vtk()` via pyvista (optional dependency)
- **data.py**: `export_to_json()` for structured data output
```
with:
```
### IO (`src/pyFracAggregate/io/`)
- **data.py**: `export_yaml()` — full aggregate snapshot (data, generation params, analysis results)
- **vtk.py**: `export_vtm()` (MultiBlock) and `export_vtk()` (point cloud) via pyvista
```

In the Top-level API section, replace `export_to_json`, `export_glb`, `export_3mf` references with `export_yaml`.

- [ ] **Step 2: Commit**

```bash
git add CLAUDE.md
git commit -m "docs: update CLAUDE.md for simplified IO layer"
```
