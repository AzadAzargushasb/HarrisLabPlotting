---
hide-toc: true
---

# HarrisLabPlotting

**Interactive 3D brain connectivity and modularity visualization for neuroscience.**

HarrisLabPlotting turns ROI-level connectivity matrices, p-value matrices, and
modularity results into publication-ready Plotly figures rendered on a real
brain mesh — both as interactive HTML you can rotate and zoom, and as static
PNG / SVG / PDF for figures.

```{image} _static/images/modularity_hero.png
:alt: Multi-view brain network with modularity coloring
:class: only-light
:width: 100%
```

```{image} _static/images/modularity_hero.png
:alt: Multi-view brain network with modularity coloring
:class: only-dark
:width: 100%
```

## Get started in five minutes

::::{grid} 1 2 2 3
:gutter: 3

:::{grid-item-card} 🚀 Install & quickstart
:link: getting_started/quickstart
:link-type: doc
Set up the package and plot your first brain network in 5 minutes.
:::

:::{grid-item-card} 📚 Tutorials
:link: tutorials/basic_plotting
:link-type: doc
Walkthroughs for connectivity, p-values, modularity, multi-view, and batch.
:::

:::{grid-item-card} 🎨 Live gallery
:link: gallery/interactive_plots
:link-type: doc
Browse every plot type — rotate the actual 3D figures right in the page.
:::

:::{grid-item-card} 🧠 Data preparation
:link: data_preparation/nifti_to_mesh
:link-type: doc
Convert NIfTI atlases to meshes and extract ROI coordinates.
:::

:::{grid-item-card} 🛠 CLI reference
:link: reference/cli
:link-type: doc
Every `hlplot` subcommand and flag, auto-generated from the Click app.
:::

:::{grid-item-card} 🐍 Python API
:link: reference/api/index
:link-type: doc
Auto-generated reference for all public functions and classes.
:::
::::

## Features

- **Plot from one command** — `hlplot plot --mesh brain.gii --coords rois.csv --matrix conn.npy`
- **Interactive 3D** — Plotly HTML with rotate, zoom, hover tooltips, and toggleable legends
- **9 preset camera views** plus arbitrary custom angles
- **5 mesh lighting presets** — flat, matte, smooth, glossy, mirror
- **Modularity-aware** — color nodes and edges by module, with Guimerà–Amaral 7-role classification
- **Statistical mode** — p-value matrices auto-transformed to −log₁₀(p) with significance thresholding
- **Publication exports** — multi-view stitched PNG strips, vector SVG/PDF, configurable DPI
- **Batch mode** — drive everything from a YAML config

## Install

```bash
conda env create -f environment.yml
conda activate harris_lab_plotting
pip install -e .
```

See [Installation](getting_started/installation.md) for pip-only and troubleshooting steps.

## Project links

- **Source:** [github.com/AzadAzargushasb/HarrisLabPlotting](https://github.com/AzadAzargushasb/HarrisLabPlotting)
- **Issues:** [github.com/AzadAzargushasb/HarrisLabPlotting/issues](https://github.com/AzadAzargushasb/HarrisLabPlotting/issues)
- **License:** MIT

```{toctree}
:hidden:
:caption: Getting started

getting_started/installation
getting_started/quickstart
getting_started/concepts
```

```{toctree}
:hidden:
:caption: Data preparation

data_preparation/nifti_to_mesh
data_preparation/roi_coordinates
data_preparation/connectivity_inputs
```

```{toctree}
:hidden:
:caption: Tutorials

tutorials/basic_plotting
tutorials/cli_walkthrough
tutorials/pvalue_plotting
tutorials/modularity
tutorials/node_role_deep_dive
tutorials/legends_and_multiview
tutorials/static_export
tutorials/batch_processing
```

```{toctree}
:hidden:
:caption: How-to recipes

how_to/customize_camera
how_to/label_specific_rois
how_to/color_nodes_by_module
how_to/threshold_a_matrix
how_to/combine_brainnet_files
```

```{toctree}
:hidden:
:caption: Gallery

gallery/interactive_plots
gallery/notebooks
```

```{toctree}
:hidden:
:caption: Reference

reference/cli
reference/api/index
```

```{toctree}
:hidden:
:caption: About

about/troubleshooting
about/citation
```
