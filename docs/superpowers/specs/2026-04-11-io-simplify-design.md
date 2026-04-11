# IO Module Simplification Design

## Goal

Simplify export formats to YAML + VTK + VTM only. Remove trimesh/lxml dependency, promote pyvista to required.

## Export Formats

| Format | Function | File |
|--------|----------|------|
| YAML | `export_yaml()` | `io/data.py` (rewrite) |
| VTK point cloud | `export_vtk()` | `io/vtk.py` (keep) |
| VTM MultiBlock | `export_vtm()` | `io/vtk.py` (keep) |

Delete `io/mesh.py` (GLB/3MF).

## YAML Export Content

Signature: `export_yaml(aggregate, path, analysis_results=None, generation_params=None)`

Three sections:

```yaml
generation:
  method: pca
  n_particles: 100
  df: 1.8
  kf: 1.2
  placement: algebraic

aggregate:
  n_particles: 100
  length_unit: nm
  mass_unit: fg
  density: 1.0
  positions: [[x, y, z], ...]
  radii: [r1, r2, ...]
  masses: [m1, m2, ...]

analysis:
  Rg: 50.3
  center_of_mass: [x, y, z]
  fractal_dimension: 1.82
```

- `analysis_results` and `generation_params` are optional; omitted sections are not written
- NumPy arrays are converted to Python lists before serialization

## Dependency Changes

```diff
  dependencies = [
      "numpy>=1.21.0",
      "scipy>=1.7.0",
      "mathutils>=3.0.0",
-     "trimesh",
-     "lxml"
+     "pyvista",
+     "pyyaml",
  ]
- [project.optional-dependencies]
- science = ["pyvista"]
  plot = ["matplotlib>=3.5.0"]
```

- Remove `trimesh`, `lxml`
- `pyvista`: optional → required
- Add `pyyaml`
- Remove `[science]` extra

## Public API Changes

```diff
- export_glb
- export_3mf
- export_to_json
+ export_yaml
  export_vtk
  export_vtm
```

## Files to Update

- `src/pyFracAggregate/io/mesh.py` — delete
- `src/pyFracAggregate/io/data.py` — rewrite `export_to_json` → `export_yaml`
- `src/pyFracAggregate/io/vtk.py` — keep, remove try/except ImportError (pyvista is now required)
- `src/pyFracAggregate/__init__.py` — update public exports
- `src/pyFracAggregate/io/__init__.py` — update if it re-exports
- `pyproject.toml` — dependency changes
- `examples/generate_and_export.py` — remove GLB, use YAML
- `CLAUDE.md` — update architecture, install commands
- Tests — remove mesh tests, add YAML export test
