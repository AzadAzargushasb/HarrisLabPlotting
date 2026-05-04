"""Sphinx configuration for HarrisLabPlotting docs."""
import os
import sys
from datetime import datetime

# Make our custom directive importable.
sys.path.insert(0, os.path.abspath("_ext"))
# Make the package itself importable for autoapi / sphinx-click.
sys.path.insert(0, os.path.abspath(".."))

# -- Project information -----------------------------------------------------

project = "HarrisLabPlotting"
author = "Harris Lab"
copyright = f"{datetime.now().year}, {author}"
release = "1.0.0"
version = "1.0"

# -- General configuration ---------------------------------------------------

extensions = [
    "myst_nb",                    # also pulls in myst_parser
    "sphinx.ext.intersphinx",
    "sphinx.ext.viewcode",
    "sphinx.ext.napoleon",
    "sphinx_design",
    "sphinx_copybutton",
    "sphinx_togglebutton",
    "sphinx_click",
    "autoapi.extension",
    "interactive_plot",           # local custom directive
]

# Source files. myst-nb registers .md / .ipynb parsers itself when listed
# in `extensions`; we don't override here.

# What to ignore when collecting sources
exclude_patterns = [
    "_build",
    "_pkg_root",            # autoapi-only scaffold (see autoapi setup below)
    "Thumbs.db",
    ".DS_Store",
    "**/.ipynb_checkpoints",
]

master_doc = "index"
language = "en"

# -- MyST / myst-nb configuration -------------------------------------------

myst_enable_extensions = [
    "colon_fence",
    "deflist",
    "attrs_inline",
    "attrs_block",
    "tasklist",
    "linkify",
    "substitution",
    "fieldlist",
]
myst_heading_anchors = 3
myst_url_schemes = ("http", "https", "mailto", "ftp")

# Notebooks: render committed outputs, never re-execute on build.
nb_execution_mode = "off"
nb_merge_streams = True

# -- Autoapi (Python API reference) -----------------------------------------

autoapi_type = "python"
# autoapi infers the package name from the directory containing __init__.py.
# This package uses a flat layout where __init__.py lives at the repo root,
# so the parent dir's name ends up being whatever the checkout was cloned
# into ("latest" on RTD) rather than "HarrisLabPlotting". Build a tiny
# scaffold dir (docs/_pkg_root/HarrisLabPlotting) at conf-load time that
# contains symlinks to the package's .py files, so autoapi sees them under
# the correct package name regardless of the checkout's directory name.
import shutil  # noqa: E402

_HERE = os.path.abspath(os.path.dirname(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_HERE, ".."))
_PKG_SCAFFOLD = os.path.join(_HERE, "_pkg_root")
_PKG_DIR = os.path.join(_PKG_SCAFFOLD, "HarrisLabPlotting")

# Wipe and recreate the scaffold each build so renamed/removed source files
# never leave stale entries behind.
if os.path.exists(_PKG_SCAFFOLD):
    shutil.rmtree(_PKG_SCAFFOLD)
os.makedirs(_PKG_DIR, exist_ok=True)

# Top-level .py files belonging to the package.
_PACKAGE_FILES = [
    "__init__.py",
    "camera.py",
    "combine.py",
    "connectivity.py",
    "loaders.py",
    "mesh.py",
    "modularity.py",
    "roi_coordinates.py",
    "utils.py",
]
for _name in _PACKAGE_FILES:
    _src = os.path.join(_REPO_ROOT, _name)
    _dst = os.path.join(_PKG_DIR, _name)
    if os.path.exists(_src):
        # Use copies, not symlinks: sphinx-autoapi 3.4 on Read the Docs
        # refuses to read symlinked sources whose realpath escapes
        # autoapi_dirs, which silently empties the API reference. Copies
        # are resolved as ordinary files. The cost is 9 small files (~200KB)
        # at conf-load time.
        shutil.copy2(_src, _dst)

autoapi_dirs = [_PKG_SCAFFOLD]
autoapi_root = "reference/api"
autoapi_keep_files = False
autoapi_add_toctree_entry = False
autoapi_file_patterns = ["*.py"]
autoapi_options = [
    "members",
    "undoc-members",
    "show-inheritance",
    "show-module-summary",
    "imported-members",
]
autoapi_python_class_content = "both"
autoapi_member_order = "groupwise"
# With the scaffold dir above, autoapi only scans the 9 explicit package
# .py files — so we don't need broad ignore patterns to exclude cli/,
# examples/, test_files/, build/, etc. Keep only the narrow ones that
# still apply (the script-entrypoint and bytecode caches).
autoapi_ignore = [
    "*/__main__.py",
    "*/__pycache__/*",
]

# -- Intersphinx --------------------------------------------------------------

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "pandas": ("https://pandas.pydata.org/docs/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
    "plotly": ("https://plotly.com/python-api-reference/", None),
    "networkx": ("https://networkx.org/documentation/stable/", None),
    "nibabel": ("https://nipy.org/nibabel/", None),
}

# -- HTML output --------------------------------------------------------------

html_theme = "furo"
html_title = "HarrisLabPlotting"
html_short_title = "HarrisLabPlotting"
html_static_path = ["_static"]
html_css_files = ["custom.css"]

html_theme_options = {
    "sidebar_hide_name": False,
    "navigation_with_keys": True,
    "source_repository": "https://github.com/AzadAzargushasb/HarrisLabPlotting",
    "source_branch": "main",
    "source_directory": "docs/",
    "footer_icons": [
        {
            "name": "GitHub",
            "url": "https://github.com/AzadAzargushasb/HarrisLabPlotting",
            "html": "",
            "class": "fa-brands fa-github",
        },
    ],
}

html_baseurl = "https://harrislabplotting.readthedocs.io/"

# -- Misc --------------------------------------------------------------------

# Suppress warnings that aren't actionable (e.g. autoapi cross-refs to private types).
suppress_warnings = [
    "autoapi.python_import_resolution",
    "myst.header",
]
