"""Sphinx configuration for the pyFracAggregate documentation site.

Follows the owner's sibling-project pattern (SegSpy, aerosol3d): Sphinx + MyST
Markdown content, Read the Docs theme, and a strict zero-warning gate
(``sphinx-build -W``). Content pages are MyST Markdown only; API reference pages
embed autodoc via ``{eval-rst}`` blocks.
"""

import tomllib
from pathlib import Path

# -- Project information ------------------------------------------------------

_repo_root = Path(__file__).resolve().parent.parent
with (_repo_root / "pyproject.toml").open("rb") as _pyproject:
    _metadata = tomllib.load(_pyproject)["project"]

project = "pyFracAggregate"
author = "Fan Zhang"
copyright = "2026, Fan Zhang"
version = _metadata["version"]
release = _metadata["version"]
license = "MIT"

# -- General configuration ----------------------------------------------------

source_suffix = {
    ".md": "markdown",
}
exclude_patterns = [
    "_build",
    # Internal SDD plans/specs live under docs/superpowers/ but are not site
    # content; they are git-ignored scratch (only the approved design spec is
    # tracked) and must not be parsed as documentation sources.
    "superpowers",
    "Thumbs.db",
    ".DS_Store",
]
language = "en"

extensions = [
    "myst_parser",
    "sphinxcontrib.mermaid",
    "sphinx_copybutton",
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "sphinx.ext.viewcode",
]

# MyST: automatic heading anchors (h1-h3) for cross-page links from other pages.
myst_heading_anchors = 3

# Napoleon: the codebase uses Google-style docstrings.
napoleon_google_docstring = True
napoleon_numpy_docstring = False

# Autodoc defaults relied on by the API reference pages (Task 2).
autodoc_default_options = {
    "members": True,
}

# -- Intersphinx ---------------------------------------------------------------

# Every inventory URL below was verified reachable before inclusion (probed
# with the Sphinx User-Agent). Note: the canonical scipy inventory
# (https://docs.scipy.org/doc/scipy/) is TLS-unreachable from some networks,
# so scipy uses its official GitHub Pages mirror instead.
intersphinx_mapping = {
    "python": ("https://docs.python.org/3/", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "scipy": ("https://scipy.github.io/devdocs/", None),
    "pyvista": ("https://docs.pyvista.org/", None),
}

# Inventory fetches can be slow on constrained networks; don't let a hung
# fetch stall the (CI) build forever.
intersphinx_timeout = 30

# -- Options for HTML output ---------------------------------------------------

html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]
