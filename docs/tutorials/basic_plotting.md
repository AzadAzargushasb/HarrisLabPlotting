# Basic plotting

This page gets you from "I have a brain mesh, ROI coordinates, and a
connectivity matrix" to a rendered interactive 3D plot in one command. For
the full feature tour with every CLI flag exercised, see the
[CLI walkthrough](cli_walkthrough.md).

## Three inputs

Every brain plot needs the same three pieces:

| Input | What it is | Common formats |
| --- | --- | --- |
| **Mesh** | The brain surface to draw on | `.gii`, `.obj`, `.mz3`, `.ply` |
| **ROI coordinates** | XYZ position of each ROI's center | CSV with `x, y, z` columns |
| **Connectivity matrix** | Pairwise edge weights between ROIs | `.npy`, `.csv`, `.mat`, `.edge` |

If you only have a NIfTI atlas, see [NIfTI → mesh](../data_preparation/nifti_to_mesh.md)
to make the surface and [ROI coordinates](../data_preparation/roi_coordinates.md)
to extract the centers.

## The CLI in one command

From `test_files/tutorial_files/`:

```bash
hlplot plot \
  --mesh brain_mesh.gii \
  --coords atlas_114_coordinates.csv \
  --matrix k5_state_0/connectivity_matrix.csv \
  --output basic.html \
  --title "Basic 114-ROI Network"
```

That writes `basic.html`. Open it in a browser — it's a fully interactive 3D
network you can rotate, zoom, hover, and toggle traces.

## The Python API in one command

```python
from HarrisLabPlotting import quick_brain_plot

fig, stats = quick_brain_plot(
    mesh_file="brain_mesh.gii",
    coords_file="atlas_114_coordinates.csv",
    matrix_file="k5_state_0/connectivity_matrix.csv",
    plot_title="Basic 114-ROI Network",
    save_path="basic.html",
)
```

`fig` is a Plotly `Figure`; `stats` is a dict with edge counts, density, and
color-mapping summaries.

## What you'll see

```{interactive-plot}
:image: images/cli_tutorial/05_114roi_metrics.png
:html: plots/auto_size_and_width.html
:alt: 114-ROI brain connectivity network
:caption: 114-ROI network rendered with default settings. Click "Interactive" to rotate.
:height: 540
```

Edges are red for positive weights and blue for negative; line width scales
with absolute weight; nodes sit at the ROI center coordinates and inherit a
single color by default.

## Where to go next

- Tune sizes, colors, thresholds, and labels: [CLI walkthrough §4–13](cli_walkthrough.md#4-basic-connectivity-plot-28-rois)
- Add per-node hover metrics: [CLI walkthrough §5](cli_walkthrough.md#5-114-roi-network-with-metrics)
- Color by module: [color nodes by module](../how_to/color_nodes_by_module.md)
- Statistical edge maps: [P-value plotting](pvalue_plotting.md)
- Modular structure: [Modularity tutorial](modularity.md)
