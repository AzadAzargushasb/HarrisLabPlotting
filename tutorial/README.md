# HarrisLabPlotting Tutorial

A comprehensive guide to brain connectivity visualization using the `hlplot` command-line tool and Python API.

---

## Table of Contents

1. [Installation](#1-installation)
2. [Quick Start](#2-quick-start)
3. [Required File Formats](#3-required-file-formats)
4. [Working with ROI Subsets](#4-working-with-roi-subsets)
5. [Basic Connectivity Plots](#5-basic-connectivity-plots)
6. [Modularity Visualization](#6-modularity-visualization)
7. [Node Customization](#7-node-customization)
8. [Edge Customization](#8-edge-customization)
9. [Static Image Export](#9-static-image-export)
10. [Camera Views](#10-camera-views)
11. [Node Metrics and Hover Data](#11-node-metrics-and-hover-data)
12. [Batch Processing](#12-batch-processing)
13. [Python API Examples](#13-python-api-examples)
14. [Complete Parameter Reference](#14-complete-parameter-reference)
15. [Troubleshooting](#15-troubleshooting)

---

## 1. Installation

### Using Conda (Recommended)

```bash
# Create environment from yml file
conda env create -f environment.yml
conda activate harris_lab_plotting

# Install the package
pip install -e .
```

### Using pip

```bash
pip install .
```

### Verify Installation

```bash
hlplot --version
hlplot --help
```

---

## 2. Quick Start

### Command Line

```bash
# Basic connectivity plot
hlplot plot \
  --mesh brain.gii \
  --coords rois.csv \
  --matrix connectivity.npy \
  --output my_plot.html

# Modularity visualization
hlplot modular \
  --mesh brain.gii \
  --coords rois.csv \
  --matrix connectivity.npy \
  --modules modules.csv \
  --q-score 0.45 \
  --output modularity.html
```

### Python

```python
from HarrisLabPlotting import (
    load_mesh_file,
    create_brain_connectivity_plot,
    create_brain_connectivity_plot_with_modularity
)
import pandas as pd

# Load data
vertices, faces = load_mesh_file("brain.gii")
roi_df = pd.read_csv("rois.csv")

# Create plot
fig, stats = create_brain_connectivity_plot(
    vertices=vertices,
    faces=faces,
    roi_coords_df=roi_df,
    connectivity_matrix="connectivity.npy",
    plot_title="My Brain Network"
)
```

---

## 3. Required File Formats

### Brain Mesh File

Supported formats: `.gii` (GIFTI), `.obj`, `.mz3`, `.ply`

The mesh defines the 3D brain surface on which nodes are displayed.

### ROI Coordinates File

CSV file with the following required columns:

| Column | Description |
|--------|-------------|
| `cog_x` | X coordinate (world space) |
| `cog_y` | Y coordinate (world space) |
| `cog_z` | Z coordinate (world space) |
| `roi_name` | Name/label for the ROI |

Optional columns: `roi_index`, voxel coordinates, etc.

**Example:**
```csv
roi_index,roi_name,cog_x,cog_y,cog_z
1,Acumbens_left,-15.36,66.02,-20.08
2,AID_left,-45.77,64.44,-7.54
3,AIP_left,-61.89,29.91,-21.02
```

### Connectivity Matrix

Supported formats: `.npy`, `.csv`, `.txt`, `.mat`, `.edge`

- Square matrix (N x N) where N = number of ROIs
- Values represent connection strength
- Positive values: positive connections
- Negative values: negative/anti-correlations
- Zero: no connection

**Important:** The matrix size must match the number of ROIs in your coordinates file!

### Module Assignments File

CSV or NPY file with integer assignments (1-indexed):

**Option 1: CSV with 'module' column**
```csv
roi_index,module
0,1
1,2
2,1
3,3
```

**Option 2: Single column (one per line)**
```
1
2
1
3
```

---

## 4. Working with ROI Subsets

Often your connectivity matrix has fewer nodes than your full atlas. Use `map_coordinate()` to extract matching ROIs.

### The Problem

```
Atlas coordinates: 170 ROIs
Connectivity matrix: 28 x 28

ERROR: Dimension mismatch!
```

### The Solution

```python
from HarrisLabPlotting import map_coordinate, load_node_file
import pandas as pd

# Load full atlas coordinates
full_roi_df = pd.read_csv("atlas_170_coordinates.csv")

# Load node file with subset of ROIs
node_df = load_node_file("my_subset.node")

# Map coordinates to match your subset
mapped_coords, unmapped = map_coordinate(full_roi_df, node_df)

# Now use mapped_coords with your 28x28 matrix
fig, stats = create_brain_connectivity_plot(
    vertices=vertices,
    faces=faces,
    roi_coords_df=mapped_coords,  # Use mapped coordinates!
    connectivity_matrix=connectivity_28x28,
    ...
)
```

### Node File Format (.node)

BrainNet Viewer format with columns:
```
x  y  z  size  color  roi_name
-57.5089  3.3815  12.8073  4  1  AUD_left
-47.7442  2.7895  25.3957  4  1  PtPD_left
```

---

## 5. Basic Connectivity Plots

### Minimal Example

```bash
hlplot plot -m brain.gii -c rois.csv -x connectivity.npy
```

### Customized Appearance

```bash
hlplot plot \
  -m brain.gii \
  -c rois.csv \
  -x connectivity.npy \
  --output custom_plot.html \
  --title "My Study Results" \
  --node-size 12 \
  --node-color steelblue \
  --mesh-opacity 0.2 \
  --camera superior
```

### With Edge Threshold

```bash
# Show only edges with absolute weight > 0.3
hlplot plot \
  -m brain.gii -c rois.csv -x connectivity.npy \
  --edge-threshold 0.3
```

### Python Equivalent

```python
fig, stats = create_brain_connectivity_plot(
    vertices=vertices,
    faces=faces,
    roi_coords_df=roi_df,
    connectivity_matrix="connectivity.npy",
    plot_title="My Study Results",
    node_size=12,
    node_color='steelblue',
    mesh_opacity=0.2,
    camera_view='superior',
    edge_threshold=0.3
)

print(f"Total edges: {stats['total_edges']}")
print(f"Positive: {stats['positive_edges']}")
print(f"Negative: {stats['negative_edges']}")
```

---

## 6. Modularity Visualization

### Basic Modularity Plot

```bash
hlplot modular \
  -m brain.gii \
  -c rois.csv \
  -x connectivity.npy \
  -d modules.csv
```

### With Q and Z Scores

Scores are appended to the title automatically:

```bash
hlplot modular \
  -m brain.gii -c rois.csv -x connectivity.npy -d modules.csv \
  --q-score 0.452 \
  --z-score 3.21 \
  --title "Network Modularity"
```

Result: "Network Modularity (Q=0.452, Z=3.21)"

### Edge Coloring Modes

**Module-colored edges** (edges inherit source node's module color):
```bash
hlplot modular ... --edge-color-mode module
```

**Sign-colored edges** (red=positive, blue=negative):
```bash
hlplot modular ... --edge-color-mode sign
```

### Python Example

```python
fig, stats = create_brain_connectivity_plot_with_modularity(
    vertices=vertices,
    faces=faces,
    roi_coords_df=roi_df,
    connectivity_matrix="connectivity.npy",
    module_assignments="modules.csv",
    plot_title="Brain Network Modularity",
    Q_score=0.452,
    Z_score=3.21,
    edge_color_mode='module',
    camera_view='oblique'
)

print(f"Number of modules: {stats['n_modules']}")
print(f"Module sizes: {stats['module_sizes']}")
```

---

## 7. Node Customization

### Node Size

**Fixed size (all nodes same):**
```bash
--node-size 15
```

**Vector of sizes (per-node):**
```bash
--node-size path/to/sizes.csv
```

**Python - from participation coefficient:**
```python
import numpy as np

# Scale participation coefficient to node sizes 5-20
pc_values = metrics_df['participation_coef'].values
node_sizes = 5 + (pc_values / pc_values.max()) * 15

fig, stats = create_brain_connectivity_plot(
    ...,
    node_size=node_sizes
)
```

### Node Color

**Single color:**
```bash
--node-color purple
--node-color "#FF5733"
--node-color "rgb(255,87,51)"
```

**From module assignments (auto-generates colors):**
```bash
--node-color modules.csv
```

**Python - module assignments:**
```python
# Integer array (1-indexed modules)
module_array = np.array([1, 2, 1, 3, 2, ...])

fig, stats = create_brain_connectivity_plot(
    ...,
    node_color=module_array
)

# Colors are auto-generated for each unique module
print(f"Color map: {stats['module_color_map']}")
```

### Node Border Color

```bash
--node-border-color darkgray
--node-border-color "#333333"
```

---

## 8. Edge Customization

### Edge Width - Scaled by Weight

Edges scale linearly between min and max based on absolute weight:

```bash
--edge-width-min 0.5 --edge-width-max 8
```

```python
edge_width=(0.5, 8.0)  # (min, max) tuple
```

### Edge Width - Fixed

All edges same width (no scaling):

```bash
--edge-width-fixed 2.0
```

```python
edge_width=2.0  # Single float
```

### Edge Threshold

Only show edges above threshold:

```bash
--edge-threshold 0.1  # Absolute value threshold
```

### Edge Colors

```bash
--pos-edge-color red
--neg-edge-color blue
```

---

## 9. Static Image Export

Export publication-quality images alongside the interactive HTML.

### PNG Export (300 DPI)

```bash
hlplot plot ... \
  --export-image figure.png \
  --image-dpi 300
```

### SVG Export (Vector - Infinitely Scalable)

```bash
hlplot plot ... \
  --export-image figure.svg
```

### PDF Export (Vector for Publications)

```bash
hlplot plot ... \
  --export-image figure.pdf
```

### Clean Export (No Title/Legend)

For figures where you'll add your own caption:

```bash
hlplot plot ... \
  --export-image clean_figure.png \
  --export-no-title \
  --export-no-legend
```

### Python Export

```python
fig, stats = create_brain_connectivity_plot(
    ...,
    export_image="figure.png",
    image_format="png",
    image_dpi=300,
    export_show_title=True,
    export_show_legend=True
)
```

**Note:** Exported images use fixed 1200x900 base dimensions. DPI scales this up (300 DPI = 4x scale, capped at ~288 for memory safety).

---

## 10. Camera Views

### Available Presets

| View | Description |
|------|-------------|
| `oblique` | Default angled view |
| `anterior` | Front view |
| `posterior` | Back view |
| `left` | Left side |
| `right` | Right side |
| `superior` | Top view (dorsal) |
| `inferior` | Bottom view (ventral) |
| `lateral-left` | Left lateral |
| `lateral-right` | Right lateral |

### CLI Usage

```bash
hlplot plot ... --camera superior
hlplot plot ... --camera lateral-left
```

### Generate Multiple Views

```bash
for view in anterior posterior left right superior inferior; do
  hlplot plot -m brain.gii -c rois.csv -x conn.npy \
    --camera $view \
    --export-image "figure_${view}.png"
done
```

### Camera Controls Dropdown

By default, visualizations include an interactive camera dropdown. Disable with:

```bash
--no-camera-controls
```

---

## 11. Node Metrics and Hover Data

Display additional metrics when hovering over nodes.

### Metrics File Format

CSV with one row per node:

```csv
roi_index,node_idx,roi_name,module,participation_coef,within_module_zscore,node_role
0,0,Acumbens_left,1,0.425,-0.579,2
1,1,AID_left,2,0.000,1.991,1
```

### CLI Usage

```bash
hlplot plot ... --node-metrics metrics.csv
```

### Python Usage

```python
metrics_df = pd.read_csv("metrics.csv")

fig, stats = create_brain_connectivity_plot(
    ...,
    node_metrics=metrics_df
)
```

Hovering over a node shows all metrics columns.

---

## 12. Batch Processing

Process multiple subjects from a single configuration file.

### Batch Configuration

```yaml
# batch_config.yaml
mesh_file: "data/brain.gii"
roi_coords_file: "data/rois.csv"
output_dir: "./outputs"
output_format: "html"

plot:
  mesh_opacity: 0.2
  node_size: 10

camera:
  view: anterior

batch:
  - name: "subject_01"
    matrix: "data/sub01_conn.npy"
    modules: "data/sub01_modules.csv"
    q_score: 0.45

  - name: "subject_02"
    matrix: "data/sub02_conn.npy"
    modules: "data/sub02_modules.csv"
    q_score: 0.48
```

### Run Batch

```bash
hlplot batch --config batch_config.yaml
```

### Dry Run

```bash
hlplot batch --config batch_config.yaml --dry-run
```

---

## 13. Python API Examples

### Example 1: Basic Network with Scaled Edge Widths

```python
from HarrisLabPlotting import (
    load_mesh_file,
    create_brain_connectivity_plot,
    load_connectivity_input
)
import pandas as pd

# Load data
vertices, faces = load_mesh_file("brain.gii")
roi_df = pd.read_csv("rois.csv")
conn_matrix = load_connectivity_input("connectivity.npy")

# Create visualization
fig, stats = create_brain_connectivity_plot(
    vertices=vertices,
    faces=faces,
    roi_coords_df=roi_df,
    connectivity_matrix=conn_matrix,
    plot_title="Brain Network",
    save_path="network.html",
    node_size=10,
    node_color='purple',
    edge_width=(1.0, 6.0),  # Scaled by weight
    mesh_opacity=0.15,
    camera_view='oblique'
)

print(f"Edges: {stats['total_edges']}")
```

### Example 2: Node Sizes from Participation Coefficient

```python
import numpy as np

# Load metrics with participation coefficient
metrics = pd.read_csv("metrics.csv")
pc_values = metrics['participation_coef'].values

# Scale PC to node sizes 5-20
node_sizes = 5 + (pc_values / pc_values.max()) * 15

fig, stats = create_brain_connectivity_plot(
    vertices=vertices,
    faces=faces,
    roi_coords_df=roi_df,
    connectivity_matrix=conn_matrix,
    plot_title="Node Size = Participation Coefficient",
    node_size=node_sizes,
    node_metrics=metrics,  # Show metrics on hover
    edge_width=(0.5, 4.0),
    camera_view='superior'
)
```

### Example 3: Modularity with All Features

```python
from HarrisLabPlotting import create_brain_connectivity_plot_with_modularity

fig, stats = create_brain_connectivity_plot_with_modularity(
    vertices=vertices,
    faces=faces,
    roi_coords_df=roi_df,
    connectivity_matrix="connectivity.csv",
    module_assignments="modules.csv",
    plot_title="Network Modularity Analysis",
    save_path="modularity.html",
    Q_score=0.452,
    Z_score=3.21,
    edge_color_mode='module',
    node_size=12,
    edge_width=(1, 5),
    mesh_opacity=0.15,
    camera_view='oblique',
    # Export static image too
    export_image="modularity.png",
    image_dpi=300
)

print(f"Modules: {stats['n_modules']}")
print(f"Sizes: {stats['module_sizes']}")
```

### Example 4: ROI Subset Mapping

```python
from HarrisLabPlotting import map_coordinate, load_node_file, load_edge_file

# Load full atlas
full_atlas = pd.read_csv("atlas_170_coordinates.csv")

# Load subset node file (28 ROIs)
node_df = load_node_file("subset.node")

# Map coordinates
mapped_coords, unmapped = map_coordinate(full_atlas, node_df)
print(f"Mapped {len(mapped_coords)} ROIs")

# Load corresponding edge matrix
edge_matrix = load_edge_file("subset.edge")

# Now dimensions match!
fig, stats = create_brain_connectivity_plot(
    vertices=vertices,
    faces=faces,
    roi_coords_df=mapped_coords,
    connectivity_matrix=edge_matrix,
    ...
)
```

---

## 14. Complete Parameter Reference

### `hlplot plot` Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--mesh`, `-m` | PATH | required | Brain mesh file (.gii, .obj, .mz3, .ply) |
| `--coords`, `-c` | PATH | required | ROI coordinates CSV |
| `--matrix`, `-x` | PATH | required | Connectivity matrix |
| `--output`, `-o` | PATH | brain_connectivity.html | Output HTML path |
| `--title`, `-t` | TEXT | "Brain Connectivity Network" | Plot title |
| `--node-size` | TEXT | "8" | Node size (number or file path) |
| `--node-color` | TEXT | "purple" | Node color (name, hex, or file) |
| `--node-border-color` | TEXT | "magenta" | Node border color |
| `--pos-edge-color` | TEXT | "red" | Positive edge color |
| `--neg-edge-color` | TEXT | "blue" | Negative edge color |
| `--edge-width-min` | FLOAT | 1.0 | Min edge width (scaled) |
| `--edge-width-max` | FLOAT | 5.0 | Max edge width (scaled) |
| `--edge-width-fixed` | FLOAT | None | Fixed edge width |
| `--edge-threshold` | FLOAT | 0.0 | Min absolute edge weight |
| `--mesh-color` | TEXT | "lightgray" | Brain mesh color |
| `--mesh-opacity` | FLOAT | 0.15 | Brain mesh opacity (0-1) |
| `--label-font-size` | INT | 8 | Label font size |
| `--fast-render` | FLAG | False | Enable fast rendering |
| `--camera` | CHOICE | "oblique" | Camera view preset |
| `--enable-camera-controls` | FLAG | True | Show camera dropdown |
| `--show-only-connected` | FLAG | True | Hide isolated nodes |
| `--hide-nodes-with-hidden-edges` | FLAG | True | Hide nodes with hidden edges |
| `--node-metrics` | PATH | None | Metrics CSV for hover |
| `--export-image` | PATH | None | Static image export path |
| `--image-format` | CHOICE | "png" | Image format |
| `--image-dpi` | INT | 300 | Export DPI |
| `--export-show-title` | FLAG | True | Title in export |
| `--export-show-legend` | FLAG | True | Legend in export |
| `--show` | FLAG | False | Open in browser |

### `hlplot modular` Additional Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--modules`, `-d` | PATH | required | Module assignments file |
| `--q-score` | FLOAT | None | Modularity Q score |
| `--z-score` | FLOAT | None | Z-rand score |
| `--edge-color-mode` | CHOICE | "module" | Edge coloring: 'module' or 'sign' |

---

## 15. Troubleshooting

### Dimension Mismatch Error

```
Matrix size (28) differs from ROI count (170)
```

**Solution:** Use `map_coordinate()` to extract matching ROIs:
```python
mapped_coords, _ = map_coordinate(full_roi_df, node_subset_df)
```

### Memory Error on Large Exports

```
Memory error during image export
```

**Solution:** Reduce DPI (max ~288 recommended):
```bash
--image-dpi 200
```

### Unicode/Encoding Errors on Windows

If you see encoding errors with special characters:
```bash
# Set console to UTF-8
chcp 65001
```

### Module Not Found

```
ModuleNotFoundError: No module named 'HarrisLabPlotting'
```

**Solution:** Install the package:
```bash
pip install -e .
```

### Edge File Not Loading

Make sure your .edge file has the correct format (BrainNet Viewer format):
```
i j weight
0 1 0.5
0 2 -0.3
1 2 0.8
```

---

## Example Data

The `test_files/` directory contains example data you can use:

- `test_files/brain_filled_2.gii` - Brain mesh
- `test_files/k5_data/state_0/connectivity_matrix.csv` - 114x114 connectivity
- `test_files/k5_data/state_0/module_assignments.csv` - Module assignments
- `test_files/k5_data/state_0/combined_metrics.csv` - Node metrics
- `test_files/node and edges/` - Example .node and .edge files

---

## Getting Help

```bash
# Main help
hlplot --help

# Command-specific help
hlplot plot --help
hlplot modular --help
hlplot batch --help
```

For issues: https://github.com/AzadAzargushasb/HarrisLabPlotting/issues

---

*HarrisLabPlotting v1.0.0 - Brain Connectivity Visualization Tools*
