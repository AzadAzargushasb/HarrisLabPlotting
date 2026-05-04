# Quickstart

This page gets you from a clean install to a rendered interactive 3D brain
network — about five minutes total.

## 1. Install

```bash
git clone https://github.com/AzadAzargushasb/HarrisLabPlotting.git
cd HarrisLabPlotting
conda env create -f environment.yml
conda activate harris_lab_plotting
pip install -e .
```

If you don't use conda, see [Installation](installation.md) for the
pip-only path.

## 2. Plot

The repo ships with sample data in `test_files/tutorial_files/`. From the
repo root:

```bash
cd test_files/tutorial_files

hlplot plot \
  --mesh brain_mesh.gii \
  --coords atlas_114_coordinates.csv \
  --matrix k5_state_0/connectivity_matrix.csv \
  --output my_first_plot.html \
  --title "My first brain network"
```

That writes `my_first_plot.html`. Open it in a browser. You can rotate,
zoom, hover for ROI names, and toggle traces from the legend.

## 3. The same plot in Python

```python
from HarrisLabPlotting import quick_brain_plot

fig, stats = quick_brain_plot(
    mesh_file="brain_mesh.gii",
    coords_file="atlas_114_coordinates.csv",
    matrix_file="k5_state_0/connectivity_matrix.csv",
    plot_title="My first brain network",
    save_path="my_first_plot.html",
)

print(f"Edges plotted: {stats['n_edges_plotted']}")
print(f"Density: {stats['density']:.3f}")
```

`fig` is a Plotly `Figure` — you can keep customizing it before saving.
`stats` is a dict with edge counts, density, and color-mapping summaries.

## 4. What you just made

```{interactive-plot}
:image: images/cli_tutorial/05_114roi_metrics.png
:html: plots/auto_size_and_width.html
:alt: 114-ROI brain connectivity network
:caption: 114-ROI network rendered with default settings — exactly what `my_first_plot.html` looks like.
:height: 540
```

## Where to go next

::::{grid} 1 2 2 2
:gutter: 2

:::{grid-item-card} 🧠 Mental model
:link: concepts
:link-type: doc
The three inputs every brain plot needs and how they fit together.
:::

:::{grid-item-card} 📚 Tutorials
:link: ../tutorials/basic_plotting
:link-type: doc
Color by module, threshold edges, export PNG, run in batch — by topic.
:::

:::{grid-item-card} 🛠 CLI walkthrough
:link: ../tutorials/cli_walkthrough
:link-type: doc
Every CLI flag, exercised end-to-end against the sample data.
:::

:::{grid-item-card} 🎨 Live gallery
:link: ../gallery/interactive_plots
:link-type: doc
Browse every plot type the package can produce.
:::
::::
