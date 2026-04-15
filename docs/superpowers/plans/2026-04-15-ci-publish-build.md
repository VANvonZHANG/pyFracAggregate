# CI/CD & Build Config Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add GitHub Actions CI (test matrix + pip cache), PyPI publishing via Trusted Publisher on GitHub Release, wheel/sdist build exclusions, and remove scipy fallback from math_utils.py.

**Architecture:** Two GitHub Actions workflows — `test.yml` (matrix Python 3.9-3.12, pip caching) and `publish.yml` (uv build + Trusted Publisher OIDC, triggered by GitHub Release). Build exclusions added to pyproject.toml to keep wheels lean. scipy fallback removed from math_utils.py since mathutils is a hard dependency.

**Tech Stack:** GitHub Actions, hatchling, uv, PyPI Trusted Publisher (OIDC)

---

## File Structure

| Action | File | Purpose |
|--------|------|---------|
| Create | `.github/workflows/test.yml` | CI test matrix |
| Create | `.github/workflows/publish.yml` | PyPI publishing on Release |
| Modify | `pyproject.toml` | Add wheel/sdist build exclusions |
| Modify | `src/pyFracAggregate/core/math_utils.py` | Remove scipy fallback, direct mathutils import |
| No change | `tests/test_core/test_math_utils.py` | Tests already pass with mathutils only |

---

### Task 1: Create test.yml workflow

**Files:**
- Create: `.github/workflows/test.yml`

- [ ] **Step 1: Create the workflow directory and file**

```yaml
name: Test
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: ["3.9", "3.10", "3.11", "3.12"]
    steps:
      - uses: actions/checkout@v4
        with:
          fetch-depth: 0
      - uses: actions/setup-python@v5
        with:
          python-version: ${{ matrix.python-version }}
          cache: 'pip'
      - run: pip install -e ".[dev]"
      - run: pytest
```

- [ ] **Step 2: Commit**

```bash
git add .github/workflows/test.yml
git commit -m "ci: add test workflow with matrix Python 3.9-3.12 and pip caching"
```

---

### Task 2: Create publish.yml workflow

**Files:**
- Create: `.github/workflows/publish.yml`

- [ ] **Step 1: Create the publish workflow file**

```yaml
name: Publish to PyPI
on:
  release:
    types: [published]
permissions:
  id-token: write
  contents: read
jobs:
  publish:
    runs-on: ubuntu-latest
    environment: pypi
    steps:
      - uses: actions/checkout@v4
        with:
          fetch-depth: 0
      - uses: astral-sh/setup-uv@v3
      - run: uv build
      - uses: pypa/gh-action-pypi-publish@release/v1
```

- [ ] **Step 2: Commit**

```bash
git add .github/workflows/publish.yml
git commit -m "ci: add publish workflow with uv build and Trusted Publisher OIDC"
```

---

### Task 3: Add build exclusions to pyproject.toml

**Files:**
- Modify: `pyproject.toml:37` (append after `[tool.pytest.ini_options]` section)

- [ ] **Step 1: Append build target sections to pyproject.toml**

Add the following after the existing `[tool.pytest.ini_options]` block:

```toml
[tool.hatch.build.targets.wheel]
packages = ["src/pyFracAggregate"]
exclude = [
    "tests",
    "docs",
    "examples",
    ".github",
]

[tool.hatch.build.targets.sdist]
exclude = [
    ".github",
]
```

Note: sdist keeps `docs` and `examples` (useful for source distribution), wheel excludes everything except runtime code.

- [ ] **Step 2: Verify build produces clean output**

Run: `pip install build && python -m build --wheel`
Check: `python -m zipfile -l dist/pyFracAggregate-0.1.0-py3-none-any.whl | grep -E 'tests|docs|examples|\.github'`
Expected: No matches (no test/doc/example/github files in wheel)

- [ ] **Step 3: Clean up build artifacts**

Run: `rm -rf dist/`

- [ ] **Step 4: Commit**

```bash
git add pyproject.toml
git commit -m "build: add wheel/sdist exclusion rules for lean packages"
```

---

### Task 4: Remove scipy fallback from math_utils.py

**Files:**
- Modify: `src/pyFracAggregate/core/math_utils.py:1-9` (replace import block)
- Modify: `src/pyFracAggregate/core/math_utils.py:25-35` (rotate_points fallback)
- Modify: `src/pyFracAggregate/core/math_utils.py:51-59` (rotate_points_quaternion fallback)
- No change: `tests/test_core/test_math_utils.py` (tests only use mathutils path)

**Important context:** `scipy` remains in `pyproject.toml` dependencies because `analysis/correlation.py` uses `scipy.spatial.cKDTree`. Only the fallback in math_utils.py is removed.

- [ ] **Step 1: Run existing tests to confirm baseline**

Run: `pytest tests/test_core/test_math_utils.py -v`
Expected: All tests PASS

- [ ] **Step 2: Replace the import block (lines 1-9)**

Replace:
```python
import numpy as np
from typing import Tuple, Optional

try:
    import mathutils
    HAS_MATHUTILS = True
except ImportError:
    HAS_MATHUTILS = False
    from scipy.spatial.transform import Rotation as R
```

With:
```python
import numpy as np
from typing import Tuple, Optional

import mathutils
```

- [ ] **Step 3: Remove fallback from rotate_points (lines 25-35)**

Replace:
```python
    if HAS_MATHUTILS:
        # Create an Euler rotation
        euler = mathutils.Euler(euler_angles, 'XYZ')
        # Convert to a 3x3 rotation matrix
        rot_matrix = np.array(euler.to_matrix())
        # Apply rotation
        return points @ rot_matrix.T
    else:
        # Fallback for Python 3.12 where mathutils might fail to install
        rot = R.from_euler('XYZ', euler_angles, degrees=False)
        return rot.apply(points)
```

With:
```python
    euler = mathutils.Euler(euler_angles, 'XYZ')
    rot_matrix = np.array(euler.to_matrix())
    return points @ rot_matrix.T
```

- [ ] **Step 4: Remove fallback from rotate_points_quaternion (lines 51-59)**

Replace:
```python
    if HAS_MATHUTILS:
        q = mathutils.Quaternion(quaternion)
        rot_matrix = np.array(q.to_matrix())
        return points @ rot_matrix.T
    else:
        # scipy Rotation uses (x, y, z, w) instead of mathutils' (w, x, y, z)
        w, x, y, z = quaternion
        rot = R.from_quat([x, y, z, w])
        return rot.apply(points)
```

With:
```python
    q = mathutils.Quaternion(quaternion)
    rot_matrix = np.array(q.to_matrix())
    return points @ rot_matrix.T
```

- [ ] **Step 5: Run tests to confirm nothing broke**

Run: `pytest tests/test_core/test_math_utils.py -v`
Expected: All tests PASS

- [ ] **Step 6: Commit**

```bash
git add src/pyFracAggregate/core/math_utils.py
git commit -m "refactor: remove scipy fallback from math_utils, mathutils is now hard dependency"
```

---

## Post-Implementation: PyPI Trusted Publisher Setup

These are manual steps performed outside this plan:

1. **Create GitHub repository** — push this repo to GitHub
2. **Configure Pending Publisher on PyPI** — PyPI → Account settings → Publishing → Add a pending publisher
   - Owner: `<your-github-username>`
   - Repository name: `pyFracAggregate`
   - Workflow filename: `publish.yml`
   - Environment name: `pypi`
   - Expected project name: `pyFracAggregate`
3. **First release** — Create a GitHub Release with tag `v0.1.0`, fill in Release Notes → PyPI auto-creates project and uploads
