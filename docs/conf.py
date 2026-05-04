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
# Absolute path so RTD's pattern matching can't be confused by ".." resolution.
_PACKAGE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
autoapi_dirs = [_PACKAGE_ROOT]
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
# Patterns are narrow on purpose: a broad `*/test_*` glob can swallow
# package files in some autoapi versions. Stick to specific subdirs.
autoapi_ignore = [
    "*/cli/*",
    "*/__main__.py",
    "*/tests/*",
    "*/docs/*",
    "*/examples/*",
    "*/test_files/*",
    "*/build/*",
    "*/_build/*",
    "*/__pycache__/*",
    "*/HarrisLabPlotting.egg-info/*",
    "*/.git/*",
    "*/_readthedocs/*",
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
