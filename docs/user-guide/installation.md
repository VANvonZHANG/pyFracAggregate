# Installation

pyFracAggregate is a library (no CLI): after installation you use it from
Python with `import pyFracAggregate as pfa`.

## Requirements

**pyFracAggregate requires Python 3.13 or newer.**

```{note}
The Python ≥ 3.13 floor is not stylistic — it comes from the
[`mathutils`](https://docs.blender.org/api/current/mathutils.html) dependency
used for 3D vector/quaternion math. Its C extension targets the 3.13 ABI, and
on several platforms no usable wheel exists for older interpreters, so older
Pythons fail at install time with compilation errors. If you see such errors,
switch to Python 3.13+ rather than fighting the build. The core runtime
dependencies (mathutils, NumPy, SciPy, pyvista, PyYAML, imageio) are installed
automatically.
```

## Install from PyPI

```console
$ pip install pyFracAggregate
```

Optional extras:

| Extra | Installs | Needed for |
|---|---|---|
| `[plot]` | matplotlib | [plotting helpers](/api-reference/index.md#analysis) such as `plot_pair_correlation` |
| `[dev]` | pytest, ruff, mypy | running the test suite and linters |
| `[docs]` | Sphinx, MyST, RTD theme | building this documentation |

```console
$ pip install "pyFracAggregate[plot]"
```

## Install from source

For development (editable install with all tools):

```console
$ git clone https://github.com/vanvonzhang/pyFracAggregate.git
$ cd pyFracAggregate
$ pip install -e ".[dev,plot]"
```

### Building `mathutils` from source

Depending on your platform and Python build, pip may need to compile
`mathutils` from source (this also happens inside virtual environments where
no matching wheel is published). That requires a working C/C++ toolchain and
Eigen headers:

```console
$ sudo apt install build-essential libeigen3-dev   # Debian/Ubuntu
```

On macOS, `xcode-select --install` plus a Homebrew Eigen (`brew install
eigen`) covers the same need; on other systems install the equivalent
`gcc`/`clang` and `eigen3` packages.

## Verify the installation

A 16-particle aggregate generates in well under a second, so this is a
complete smoke test:

```python
>>> import pyFracAggregate as pfa
>>> agg = pfa.generate(n_particles=16, df=1.8, kf=1.9, method="pca")
>>> agg.current_size
16
```

If this prints `16`, generation, the placement layer, and the core data
structure are all working. Continue with
[Generating aggregates](generators.md).
