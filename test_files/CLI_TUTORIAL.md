# HarrisLabPlotting CLI Tutorial

This tutorial demonstrates all features of the `hlplot` command-line interface. Each section corresponds to a test from the Jupyter notebook and shows the equivalent CLI command.

**All commands should be run from the `test_files` directory:**

```bash
cd C:\Users\Azad Azargushasb\Research\HarrisLabPlotting\test_files
```

**Output folder for all results:**

```
tutorial_cli_output/
```

---

## Table of Contents

1. [Prerequisites: Mapping ROI Coordinates](#1-prerequisites-mapping-roi-coordinates)
2. [Test 1: Basic Plot with Edge File](#2-test-1-basic-plot-with-edge-file)
3. [Test 2: Vector Node Sizes with Metrics Hover](#3-test-2-vector-node-sizes-with-metrics-hover)
4. [Test 3: Utility Commands](#4-test-3-utility-commands)
5. [Test 4: Fixed Edge Width](#5-test-4-fixed-edge-width)
6. [Test 5: Static Image Exports](#6-test-5-static-image-exports)
7. [Test 6: Clean Exports (No Title/Legend)](#7-test-6-clean-exports-no-titlelegend)
8. [Test 7: Node Visibility with Edge Toggling](#8-test-7-node-visibility-with-edge-toggling)
9. [Test 8: Node Colors from Module Assignments](#9-test-8-node-colors-from-module-assignments)
10. [Test 9: Modularity Visualization](#10-test-9-modularity-visualization)
11. [Complete Flag Reference](#11-complete-flag-reference)

---

## 1. Prerequisites: Mapping ROI Coordinates

Before creating visualizations, you often need to map your ROI coordinates to match your connectivity matrix dimensions. The notebook uses a 170-ROI atlas but the edge file only has 28 ROIs.

### Problem

```
Full atlas: 170 ROIs
Edge matrix: 28 x 28
ERROR: Dimension mismatch!
```

### Solution: Map coordinates using the node file

The `.node` file contains the 28 ROI names that match the edge file. We extract only those coordinates from the full atlas.

### Copy-Paste Command

```bash
hlplot coords map-subset \
  --coords "C:\Users\Azad Azargushasb\Research\roi_coordinates\atlas_170_coordinates\atlas_170_coordinates_comma.csv" \
  --subset "node and edges\total.node" \
  --output-dir "tutorial_cli_output" \
  --name "atlas_28_mapped"
```

### Output Files

- `tutorial_cli_output/atlas_28_mapped_comma.csv` - Use this for plotting
- `tutorial_cli_output/atlas_28_mapped_tab.csv` - Tab-delimited version
- `tutorial_cli_output/atlas_28_mapped.pkl` - Python pickle format

### Flag Explanations

| Flag | Description |
|------|-------------|
| `--coords`, `-c` | Path to the full ROI coordinates CSV file (170 ROIs) |
| `--subset`, `-s` | Path to the subset file. Supports `.node` (BrainNet Viewer), `.txt`, or `.csv` with ROI names |
| `--output-dir`, `-o` | Directory where output files will be saved |
| `--name`, `-n` | Base name for output files (without extension) |

---

## 2. Test 1: Basic Plot with Edge File

This test demonstrates:
- Loading connectivity from a `.edge` file (BrainNet Viewer format)
- Positive/negative edge coloring (red/blue)
- Scaled edge widths based on connection strength
- Hiding nodes when their edges are toggled off

### Copy-Paste Command

```bash
hlplot plot \
  --mesh "C:\Users\Azad Azargushasb\Research\brain_filled_3_smoothed.gii" \
  --coords "tutorial_cli_output\atlas_28_mapped_comma.csv" \
  --matrix "node and edges\total(1).edge" \
  --output "tutorial_cli_output\test1_edge_file_toggle.html" \
  --title "Test 1: Edge File + Pos/Neg Toggling" \
  --node-size 10 \
  --edge-width-min 1.0 \
  --edge-width-max 10.0 \
  --camera superior \
  --hide-nodes-with-hidden-edges \
  --label-font-size 20
```

### Expected Output

- Interactive HTML file with 28 nodes and 27 edges
- Positive edges shown in red, negative in blue
- Edge widths scale from 1.0 to 10.0 based on absolute weight
- Clicking legend items toggles edge visibility AND node visibility

### Flag Explanations

| Flag | Description |
|------|-------------|
| `--mesh`, `-m` | Path to brain mesh file (.gii, .obj, .mz3, .ply) |
| `--coords`, `-c` | Path to ROI coordinates CSV (must have cog_x, cog_y, cog_z, roi_name columns) |
| `--matrix`, `-x` | Path to connectivity matrix (.npy, .csv, .edge, .txt, .mat) |
| `--output`, `-o` | Output HTML file path |
| `--title`, `-t` | Plot title displayed at top |
| `--node-size` | Size of all nodes (single number) or path to CSV with per-node sizes |
| `--edge-width-min` | Minimum edge width when scaling by weight |
| `--edge-width-max` | Maximum edge width when scaling by weight |
| `--camera` | Camera view preset (oblique, anterior, posterior, left, right, superior, inferior) |
| `--hide-nodes-with-hidden-edges` | When enabled, nodes hide when their edge type is toggled off in legend |
| `--label-font-size` | Font size for ROI labels on hover |

---

## 3. Test 2: Vector Node Sizes with Metrics Hover

This test demonstrates:
- Using a 114-ROI network
- Node metrics displayed on hover
- Note: Vector node sizes from a CSV file are not directly supported in CLI (use Python API for this)

### Copy-Paste Command

```bash
hlplot plot \
  --mesh "C:\Users\Azad Azargushasb\Research\brain_filled_3_smoothed.gii" \
  --coords "G:\My Drive\research stim data cci\atlas_114_mapped_comma.csv" \
  --matrix "k5_data\state_0\connectivity_matrix.csv" \
  --output "tutorial_cli_output\test2_with_metrics.html" \
  --title "Test 2: 114-ROI Network with Metrics Hover" \
  --node-size 10 \
  --edge-width-min 0.5 \
  --edge-width-max 4.0 \
  --camera oblique \
  --node-metrics "k5_data\state_0\combined_metrics.csv" \
  --hide-nodes-with-hidden-edges
```

### Expected Output

- 114 nodes with 452 edges
- Hovering over nodes shows all metrics from the CSV
- Metrics include: module, participation_coef, within_module_zscore, etc.

### Flag Explanations

| Flag | Description |
|------|-------------|
| `--node-metrics` | Path to CSV file with node metrics. Each row corresponds to a node. All columns are displayed on hover. |

---

## 4. Test 3: Utility Commands

The CLI provides utility commands for inspecting and manipulating data files.

### 4a. View Matrix Information

```bash
hlplot utils info --matrix "node and edges\total(1).edge"
```

**Output shows:**
- Matrix shape and data type
- Non-zero values and density
- Min/max values
- Positive/negative edge counts
- Whether matrix is symmetric

### 4b. View Matrix Information (114x114)

```bash
hlplot utils info --matrix "k5_data\state_0\connectivity_matrix.csv"
```

### 4c. Validate Files for Compatibility

```bash
hlplot utils validate \
  --mesh "brain_filled_2.gii" \
  --coords "tutorial_cli_output\atlas_28_mapped_comma.csv" \
  --matrix "node and edges\total(1).edge"
```

**Output shows:**
- Whether each file can be loaded
- ROI counts and matrix dimensions
- Compatibility between files

### 4d. Load and Inspect Coordinates

```bash
hlplot coords load \
  --file "tutorial_cli_output\atlas_28_mapped_comma.csv" \
  --show-stats \
  --validate
```

### Flag Explanations

| Command | Flag | Description |
|---------|------|-------------|
| `utils info` | `--matrix`, `-m` | Connectivity matrix file to inspect |
| `utils validate` | `--mesh`, `-m` | Mesh file to validate |
| `utils validate` | `--coords`, `-c` | Coordinates file to validate |
| `utils validate` | `--matrix`, `-x` | Matrix file to validate |
| `utils validate` | `--modules`, `-d` | Module assignments file to validate |
| `coords load` | `--file`, `-f` | Coordinates file to load |
| `coords load` | `--show-stats` | Show coordinate statistics |
| `coords load` | `--validate` | Check coordinate format |
| `coords load` | `--show-head N` | Show first N rows |

---

## 5. Test 4: Fixed Edge Width

This test demonstrates using a fixed edge width instead of scaling by weight.

### Copy-Paste Command

```bash
hlplot plot \
  --mesh "C:\Users\Azad Azargushasb\Research\brain_filled_3_smoothed.gii" \
  --coords "G:\My Drive\research stim data cci\atlas_114_mapped_comma.csv" \
  --matrix "k5_data\state_0\connectivity_matrix.csv" \
  --output "tutorial_cli_output\test4_fixed_edge_width.html" \
  --title "Test 4: Fixed Edge Width" \
  --node-size 10 \
  --edge-width-fixed 2.0 \
  --camera anterior
```

### Expected Output

- All edges have the same width (2.0)
- No scaling based on connection strength

### Flag Explanations

| Flag | Description |
|------|-------------|
| `--edge-width-fixed` | Fixed width for all edges. When set, ignores `--edge-width-min` and `--edge-width-max` |

---

## 6. Test 5: Static Image Exports

Export publication-quality static images alongside interactive HTML.

### 5a. PNG Export (300 DPI - Publication Quality)

```bash
hlplot plot \
  --mesh "C:\Users\Azad Azargushasb\Research\brain_filled_3_smoothed.gii" \
  --coords "G:\My Drive\research stim data cci\atlas_114_mapped_comma.csv" \
  --matrix "k5_data\state_0\connectivity_matrix.csv" \
  --output "tutorial_cli_output\test5a_png_export.html" \
  --title "Test 5a: PNG Export (300 DPI)" \
  --node-size 10 \
  --edge-width-min 0.5 \
  --edge-width-max 4.0 \
  --camera superior \
  --node-metrics "k5_data\state_0\combined_metrics.csv" \
  --export-image "tutorial_cli_output\test5a_brain_network.png" \
  --image-dpi 300
```

### 5b. SVG Export (Vector - Infinitely Scalable)

```bash
hlplot plot \
  --mesh "C:\Users\Azad Azargushasb\Research\brain_filled_3_smoothed.gii" \
  --coords "G:\My Drive\research stim data cci\atlas_114_mapped_comma.csv" \
  --matrix "k5_data\state_0\connectivity_matrix.csv" \
  --output "tutorial_cli_output\test5b_svg_export.html" \
  --title "Test 5b: SVG Export (Vector)" \
  --node-size 10 \
  --edge-width-min 0.5 \
  --edge-width-max 4.0 \
  --camera oblique \
  --export-image "tutorial_cli_output\test5b_brain_network.svg"
```

### 5c. PDF Export (Vector for Publications)

```bash
hlplot plot \
  --mesh "C:\Users\Azad Azargushasb\Research\brain_filled_3_smoothed.gii" \
  --coords "G:\My Drive\research stim data cci\atlas_114_mapped_comma.csv" \
  --matrix "k5_data\state_0\connectivity_matrix.csv" \
  --output "tutorial_cli_output\test5c_pdf_export.html" \
  --title "Test 5c: PDF Export (Publication)" \
  --node-size 10 \
  --edge-width-min 0.5 \
  --edge-width-max 4.0 \
  --camera anterior \
  --export-image "tutorial_cli_output\test5c_brain_network.pdf"
```

### 5d. High DPI PNG Export

```bash
hlplot plot \
  --mesh "C:\Users\Azad Azargushasb\Research\brain_filled_3_smoothed.gii" \
  --coords "G:\My Drive\research stim data cci\atlas_114_mapped_comma.csv" \
  --matrix "k5_data\state_0\connectivity_matrix.csv" \
  --output "tutorial_cli_output\test5d_highres_export.html" \
  --title "Test 5d: High DPI PNG" \
  --node-size 10 \
  --edge-width-min 0.5 \
  --edge-width-max 4.0 \
  --camera left \
  --export-image "tutorial_cli_output\test5d_brain_network_large.png" \
  --image-dpi 288
```

### Flag Explanations

| Flag | Description |
|------|-------------|
| `--export-image` | Path to save static image. Supports `.png`, `.svg`, `.pdf` |
| `--image-dpi` | DPI for PNG export. Max ~288 for memory safety. Default: 300 |
| `--image-format` | Image format if path has no extension. Options: png, svg, pdf |

### Output Dimensions

- Base dimensions: 1200 x 900 pixels
- At 300 DPI: 4800 x 3600 pixels (capped at 4x scale)
- SVG and PDF are vector formats (infinitely scalable)

---

## 7. Test 6: Clean Exports (No Title/Legend)

For publication figures where you want to add your own caption.

### 6a. Clean PNG (No Title, No Legend)

```bash
hlplot plot \
  --mesh "C:\Users\Azad Azargushasb\Research\brain_filled_3_smoothed.gii" \
  --coords "tutorial_cli_output\atlas_28_mapped_comma.csv" \
  --matrix "node and edges\total(1).edge" \
  --output "tutorial_cli_output\test6a_no_title_no_legend.html" \
  --title "This title will NOT appear in export" \
  --edge-width-min 1.0 \
  --edge-width-max 6.0 \
  --camera superior \
  --export-image "tutorial_cli_output\test6a_clean_export.png" \
  --export-no-title \
  --export-no-legend \
  --image-dpi 150
```

### 6b. Title Only (No Legend)

```bash
hlplot plot \
  --mesh "C:\Users\Azad Azargushasb\Research\brain_filled_3_smoothed.gii" \
  --coords "tutorial_cli_output\atlas_28_mapped_comma.csv" \
  --matrix "node and edges\total(1).edge" \
  --output "tutorial_cli_output\test6b_title_only.html" \
  --title "Brain Connectivity Network" \
  --edge-width-min 1.0 \
  --edge-width-max 6.0 \
  --camera oblique \
  --export-image "tutorial_cli_output\test6b_title_only.png" \
  --export-no-legend \
  --image-dpi 150
```

### 6c. Clean SVG

```bash
hlplot plot \
  --mesh "C:\Users\Azad Azargushasb\Research\brain_filled_3_smoothed.gii" \
  --coords "tutorial_cli_output\atlas_28_mapped_comma.csv" \
  --matrix "node and edges\total(1).edge" \
  --output "tutorial_cli_output\test6c_clean_svg.html" \
  --title "This should not appear" \
  --edge-width-min 1.0 \
  --edge-width-max 6.0 \
  --camera anterior \
  --export-image "tutorial_cli_output\test6c_clean.svg" \
  --export-no-title \
  --export-no-legend
```

### 6d. Clean PDF (Ideal for Publications)

```bash
hlplot plot \
  --mesh "C:\Users\Azad Azargushasb\Research\brain_filled_3_smoothed.gii" \
  --coords "tutorial_cli_output\atlas_28_mapped_comma.csv" \
  --matrix "node and edges\total(1).edge" \
  --output "tutorial_cli_output\test6d_clean_pdf.html" \
  --title "This should not appear" \
  --edge-width-min 1.0 \
  --edge-width-max 6.0 \
  --camera left \
  --export-image "tutorial_cli_output\test6d_clean.pdf" \
  --export-no-title \
  --export-no-legend
```

### Flag Explanations

| Flag | Description |
|------|-------------|
| `--export-no-title` | Exclude title from exported image |
| `--export-no-legend` | Exclude legend from exported image |
| `--export-show-title` | Include title in export (default) |
| `--export-show-legend` | Include legend in export (default) |

---

## 8. Test 7: Node Visibility with Edge Toggling

This test creates a visualization where nodes can be toggled along with their edges.

### Copy-Paste Command

```bash
hlplot plot \
  --mesh "C:\Users\Azad Azargushasb\Research\brain_filled_3_smoothed.gii" \
  --coords "tutorial_cli_output\atlas_28_mapped_comma.csv" \
  --matrix "node and edges\total(1).edge" \
  --output "tutorial_cli_output\test7_node_visibility.html" \
  --title "Test 7: Node Visibility with Edge Toggle" \
  --node-size 12 \
  --edge-width-min 1.0 \
  --edge-width-max 8.0 \
  --camera superior \
  --hide-nodes-with-hidden-edges \
  --label-font-size 14
```

### Expected Behavior

1. Click "Positive Edges" in legend → Positive edges AND their nodes hide
2. Click "Negative Edges" in legend → Negative edges AND their nodes hide
3. Click BOTH → ALL nodes and edges hide (only brain surface remains)

### Flag Explanations

| Flag | Description |
|------|-------------|
| `--hide-nodes-with-hidden-edges` | Enable node-edge linking. Nodes hide when their edge type is toggled off. |
| `--keep-nodes-visible` | Opposite of above. Nodes stay visible even when edges are hidden. |

---

## 9. Test 8: Node Colors from Module Assignments

Color nodes by module/community assignment.

### 8a. Using Module Assignments CSV File

```bash
hlplot plot \
  --mesh "C:\Users\Azad Azargushasb\Research\brain_filled_3_smoothed.gii" \
  --coords "G:\My Drive\research stim data cci\atlas_114_mapped_comma.csv" \
  --matrix "k5_data\state_0\connectivity_matrix.csv" \
  --output "tutorial_cli_output\test8a_module_colors.html" \
  --title "Test 8a: Node Colors from Module Assignments" \
  --node-size 10 \
  --node-color "k5_data\state_0\module_assignments.csv" \
  --node-border-color darkgray \
  --edge-width-min 0.5 \
  --edge-width-max 4.0 \
  --camera oblique
```

### 8b. Same with Anterior View

```bash
hlplot plot \
  --mesh "C:\Users\Azad Azargushasb\Research\brain_filled_3_smoothed.gii" \
  --coords "G:\My Drive\research stim data cci\atlas_114_mapped_comma.csv" \
  --matrix "k5_data\state_0\connectivity_matrix.csv" \
  --output "tutorial_cli_output\test8b_module_colors_csv.html" \
  --title "Test 8b: Node Colors from CSV File" \
  --node-size 10 \
  --node-color "k5_data\state_0\module_assignments.csv" \
  --node-border-color darkgray \
  --edge-width-min 0.5 \
  --edge-width-max 4.0 \
  --camera anterior
```

### Expected Output

- Nodes colored by module assignment (6 unique colors for 6 modules)
- Colors are auto-generated and visually distinct
- Module 1: Red, Module 2: Green, Module 3: Blue, etc.

### Flag Explanations

| Flag | Description |
|------|-------------|
| `--node-color` | Accepts: color name (`purple`), hex code (`#FF5733`), or CSV path with module assignments |
| `--node-border-color` | Border/outline color for nodes. Default: magenta |

### Module Assignments CSV Format

```csv
roi_index,module
0,1
1,2
2,1
3,3
...
```

---

## 10. Test 9: Modularity Visualization

Use the dedicated `hlplot modular` command for modularity analysis.

### 9a. Basic Modularity with Q and Z Scores

```bash
hlplot modular \
  --mesh "C:\Users\Azad Azargushasb\Research\brain_filled_3_smoothed.gii" \
  --coords "G:\My Drive\research stim data cci\atlas_114_mapped_comma.csv" \
  --matrix "k5_data\state_0\connectivity_matrix.csv" \
  --modules "k5_data\state_0\module_assignments.csv" \
  --output "tutorial_cli_output\test9a_modularity_viz.html" \
  --title "Brain Network Modularity" \
  --q-score 0.452 \
  --z-score 3.21 \
  --node-size 10 \
  --edge-width-min 0.5 \
  --edge-width-max 4.0 \
  --camera oblique
```

### Expected Output

- Title shows: "Brain Network Modularity (Q=0.452, Z=3.21)"
- Nodes colored by module
- Legend shows module colors
- Edges colored by positive/negative sign (default)

### 9b. Module-Colored Edges

```bash
hlplot modular \
  --mesh "C:\Users\Azad Azargushasb\Research\brain_filled_3_smoothed.gii" \
  --coords "G:\My Drive\research stim data cci\atlas_114_mapped_comma.csv" \
  --matrix "k5_data\state_0\connectivity_matrix.csv" \
  --modules "k5_data\state_0\module_assignments.csv" \
  --output "tutorial_cli_output\test9c_module_edges.html" \
  --title "Modularity with Module-Colored Edges" \
  --q-score 0.452 \
  --z-score 3.21 \
  --edge-color-mode module \
  --node-size 10 \
  --edge-width-min 0.5 \
  --edge-width-max 4.0 \
  --camera oblique
```

### Expected Output

- Edges colored by the SOURCE node's module color
- Same color for node and its outgoing edges

### 9c. Sign-Colored Edges (Default)

```bash
hlplot modular \
  --mesh "C:\Users\Azad Azargushasb\Research\brain_filled_3_smoothed.gii" \
  --coords "G:\My Drive\research stim data cci\atlas_114_mapped_comma.csv" \
  --matrix "k5_data\state_0\connectivity_matrix.csv" \
  --modules "k5_data\state_0\module_assignments.csv" \
  --output "tutorial_cli_output\test9b_sign_edges.html" \
  --title "Modularity with Sign-Colored Edges" \
  --edge-color-mode sign \
  --node-size 10 \
  --camera anterior
```

### Flag Explanations

| Flag | Description |
|------|-------------|
| `--modules`, `-d` | Path to module assignments file (required for `hlplot modular`) |
| `--q-score` | Modularity Q score to display in title |
| `--z-score` | Z-rand score to display in title |
| `--edge-color-mode` | Edge coloring mode: `sign` (red/blue by positive/negative) or `module` (by source node's module) |

---

## 11. Complete Flag Reference

### `hlplot plot` - All Flags

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--mesh`, `-m` | PATH | required | Brain mesh file (.gii, .obj, .mz3, .ply) |
| `--coords`, `-c` | PATH | required | ROI coordinates CSV |
| `--matrix`, `-x` | PATH | required | Connectivity matrix |
| `--output`, `-o` | PATH | brain_connectivity.html | Output HTML path |
| `--title`, `-t` | TEXT | "Brain Connectivity Network" | Plot title |
| `--node-size` | TEXT | "8" | Node size (number or CSV file path) |
| `--node-color` | TEXT | "purple" | Node color (name, hex, or CSV with modules) |
| `--node-border-color` | TEXT | "magenta" | Node border color |
| `--pos-edge-color` | TEXT | "red" | Positive edge color |
| `--neg-edge-color` | TEXT | "blue" | Negative edge color |
| `--edge-width-min` | FLOAT | 1.0 | Min edge width (scaled mode) |
| `--edge-width-max` | FLOAT | 5.0 | Max edge width (scaled mode) |
| `--edge-width-fixed` | FLOAT | None | Fixed edge width (disables scaling) |
| `--edge-threshold` | FLOAT | 0.0 | Min absolute edge weight to display |
| `--mesh-color` | TEXT | "lightgray" | Brain mesh color |
| `--mesh-opacity` | FLOAT | 0.15 | Mesh transparency (0-1) |
| `--label-font-size` | INT | 8 | Label font size |
| `--fast-render` | FLAG | False | Enable fast rendering |
| `--camera` | CHOICE | "oblique" | Camera view preset |
| `--enable-camera-controls` | FLAG | True | Show camera dropdown |
| `--no-camera-controls` | FLAG | False | Hide camera dropdown |
| `--show-only-connected` | FLAG | True | Hide isolated nodes |
| `--show-all-nodes` | FLAG | False | Show all nodes including isolated |
| `--hide-nodes-with-hidden-edges` | FLAG | True | Hide nodes when edges hidden |
| `--keep-nodes-visible` | FLAG | False | Keep nodes visible always |
| `--node-metrics` | PATH | None | CSV with metrics for hover |
| `--export-image` | PATH | None | Static image export path |
| `--image-format` | CHOICE | "png" | Image format (png, svg, pdf) |
| `--image-dpi` | INT | 300 | PNG export DPI |
| `--export-show-title` | FLAG | True | Title in export |
| `--export-no-title` | FLAG | False | No title in export |
| `--export-show-legend` | FLAG | True | Legend in export |
| `--export-no-legend` | FLAG | False | No legend in export |
| `--show` | FLAG | False | Open in browser |

### `hlplot modular` - Additional Flags

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--modules`, `-d` | PATH | required | Module assignments file |
| `--q-score` | FLOAT | None | Modularity Q score for title |
| `--z-score` | FLOAT | None | Z-rand score for title |
| `--edge-color-mode` | CHOICE | "module" | Edge coloring: 'module' or 'sign' |

### Camera View Options

| View | Description |
|------|-------------|
| `oblique` | Default angled view |
| `anterior` | Front view |
| `posterior` | Back view |
| `left` | Left side |
| `right` | Right side |
| `superior` | Top (dorsal) |
| `inferior` | Bottom (ventral) |
| `lateral-left` | Left lateral |
| `lateral-right` | Right lateral |

---

## Running All Tests

To run all tests in sequence, save this as a batch file:

```bash
# Create output directory
mkdir tutorial_cli_output

# Prerequisites: Map coordinates
hlplot coords map-subset --coords "C:\Users\Azad Azargushasb\Research\roi_coordinates\atlas_170_coordinates\atlas_170_coordinates_comma.csv" --subset "node and edges\total.node" --output-dir "tutorial_cli_output" --name "atlas_28_mapped"

# Test 1: Basic plot with edge file
hlplot plot --mesh "C:\Users\Azad Azargushasb\Research\brain_filled_3_smoothed.gii" --coords "tutorial_cli_output\atlas_28_mapped_comma.csv" --matrix "node and edges\total(1).edge" --output "tutorial_cli_output\test1_edge_file_toggle.html" --title "Test 1: Edge File + Pos/Neg Toggling" --node-size 10 --edge-width-min 1.0 --edge-width-max 10.0 --camera superior --hide-nodes-with-hidden-edges

# Test 4: Fixed edge width
hlplot plot --mesh "C:\Users\Azad Azargushasb\Research\brain_filled_3_smoothed.gii" --coords "G:\My Drive\research stim data cci\atlas_114_mapped_comma.csv" --matrix "k5_data\state_0\connectivity_matrix.csv" --output "tutorial_cli_output\test4_fixed_edge_width.html" --title "Test 4: Fixed Edge Width" --node-size 10 --edge-width-fixed 2.0 --camera anterior

# Test 5a: PNG export
hlplot plot --mesh "C:\Users\Azad Azargushasb\Research\brain_filled_3_smoothed.gii" --coords "G:\My Drive\research stim data cci\atlas_114_mapped_comma.csv" --matrix "k5_data\state_0\connectivity_matrix.csv" --output "tutorial_cli_output\test5a_png_export.html" --title "Test 5a: PNG Export" --camera superior --export-image "tutorial_cli_output\test5a_brain_network.png" --image-dpi 300

# Test 6a: Clean export
hlplot plot --mesh "C:\Users\Azad Azargushasb\Research\brain_filled_3_smoothed.gii" --coords "tutorial_cli_output\atlas_28_mapped_comma.csv" --matrix "node and edges\total(1).edge" --output "tutorial_cli_output\test6a_clean.html" --camera superior --export-image "tutorial_cli_output\test6a_clean.png" --export-no-title --export-no-legend

# Test 8a: Module colors
hlplot plot --mesh "C:\Users\Azad Azargushasb\Research\brain_filled_3_smoothed.gii" --coords "G:\My Drive\research stim data cci\atlas_114_mapped_comma.csv" --matrix "k5_data\state_0\connectivity_matrix.csv" --output "tutorial_cli_output\test8a_module_colors.html" --title "Test 8a: Module Colors" --node-color "k5_data\state_0\module_assignments.csv" --camera oblique

# Test 9a: Modularity visualization
hlplot modular --mesh "C:\Users\Azad Azargushasb\Research\brain_filled_3_smoothed.gii" --coords "G:\My Drive\research stim data cci\atlas_114_mapped_comma.csv" --matrix "k5_data\state_0\connectivity_matrix.csv" --modules "k5_data\state_0\module_assignments.csv" --output "tutorial_cli_output\test9a_modularity.html" --title "Brain Network Modularity" --q-score 0.452 --z-score 3.21 --camera oblique
```

---

## Notes

1. **File Paths**: All paths are relative to the `test_files` directory unless specified as absolute paths.

2. **Windows Paths**: Use backslashes (`\`) or forward slashes (`/`) for paths on Windows.

3. **Spaces in Paths**: Wrap paths containing spaces in quotes.

4. **Vector Node Sizes**: The CLI currently supports fixed node sizes or sizes from a CSV file. For computed vector sizes (like scaling by participation coefficient), use the Python API.

5. **Module File Format**: The module assignments CSV should have at minimum a `module` column with integer values (1-indexed).

---

*Generated from the Jupyter notebook: `bcp update to include node size vector test, edge width scaled to matrix, utils, etc.ipynb`*
