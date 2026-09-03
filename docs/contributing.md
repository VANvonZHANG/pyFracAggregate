# Contributing

Issues and pull requests are welcome at
[github.com/vanvonzhang/pyFracAggregate](https://github.com/vanvonzhang/pyFracAggregate).
This page covers the local setup and the checks a change is expected to pass.

## Development setup

Development needs **Python ≥ 3.13** (see the
[installation guide](/user-guide/installation.md) for why the floor exists and
how to satisfy the `mathutils` dependency on your platform). Clone the
repository and install in editable mode with all development extras:

```bash
pip install -e ".[dev,plot]"     # pytest, ruff, mypy + matplotlib
```

## Running the tests

```bash
pytest                                      # Run all tests
pytest tests/test_core/test_aggregate.py    # Run a single test file
pytest -k "test_pca"                        # Run tests matching a name pattern
```

The suite mirrors the source layout (`tests/test_core/`, `test_generators/`,
`test_analysis/`, `test_io/`), so new modules should get a matching test
file. Slow performance tests are marked `benchmark`; deselect them with:

```bash
pytest -m "not benchmark"
```

and run the benchmark set alone with `pytest -m benchmark`.

## Lint and type checks

```bash
ruff check src/     # Lint source code
mypy src/           # Static type checking
```

The codebase is typed throughout; type hints on public functions are part of
the API.

## Building the documentation

The docs are Sphinx + MyST Markdown living in `docs/` (see
[Architecture](/architecture/index.md) for the site layout):

```bash
pip install -e ".[docs]"                         # Sphinx, MyST, RTD theme, ...
sphinx-build -W -b html docs docs/_build/html    # Strict build
```

**The zero-warning gate is mandatory for docs changes**: `sphinx-build -W`
turns any warning (broken cross-reference, missing anchor, autodoc error)
into a build failure, and a PR that touches `docs/` must build clean. The
first build fetches intersphinx inventories (Python, NumPy, scipy, pyvista)
over the network; subsequent builds use the cached environment.

Preview the result by opening `docs/_build/html/index.html` in a browser.

## Pull request expectations

Before you open a PR:

- `pytest` passes (or `-m "not benchmark"` if you are deliberately skipping
  the slow set — say so in the PR).
- `ruff check src/` and `mypy src/` are clean on the files you touched.
- If you touched anything under `docs/` (including docstrings that feed the
  API reference), `sphinx-build -W -b html docs docs/_build/html` exits 0
  with zero warnings.
- New public API gets a docstring (Google style) and a matching entry in the
  [API reference](/api-reference/index.md) when it belongs in `__all__`.
