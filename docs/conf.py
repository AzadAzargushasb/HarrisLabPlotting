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
    "sphinx.ext.autodoc",         # imports the package and reads docstrings
    "sphinx.ext.napoleon",        # NumPy/Google docstring formats
    "sphinx_design",
    "sphinx_copybutton",
    "sphinx_togglebutton",
    "sphinx_click",
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

# -- autodoc (Python API reference) -----------------------------------------
# We use sphinx.ext.autodoc — which imports the installed package and reads
# docstrings via Python introspection — instead of sphinx-autoapi, which
# scans source files. autoapi's source-file scanner kept silently failing
# on RTD ("Unable to read file:" for every module) because of quirks in how
# it resolves package names for flat-layout packages whose __init__.py sits
# at the repo root. autodoc has none of those quirks: it just does
# `import HarrisLabPlotting.utils` and reflects on the live module.
#
# Each module's API reference page lives at docs/reference/api/<module>.md
# and uses `.. automodule:: HarrisLabPlotting.<module>` to pull in members.

autodoc_default_options = {
    "members": True,
    "undoc-members": True,
    "show-inheritance": True,
    "member-order": "groupwise",
}
autodoc_typehints = "description"
autodoc_class_signature = "separated"

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

suppress_warnings = [
    "myst.header",
]
