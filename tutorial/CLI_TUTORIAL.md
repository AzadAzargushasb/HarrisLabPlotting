# HarrisLabPlotting CLI Tutorial

This tutorial demonstrates all features of the `hlplot` command-line interface. Each section corresponds to a test from the Jupyter notebook and shows the equivalent CLI command.

**All commands should be run from the `test_files/tutorial_files` directory:**

```bash
cd HarrisLabPlotting/test_files/tutorial_files
```

**All output goes to:**

```
output/
```

> **Running the code blocks below.** Blocks marked **`bash`** are shell
> commands — paste them into a terminal. Blocks marked **`python`** are
> Python: paste them into a Jupyter notebook cell or a `.py` file
> as-is, **or** run them from a terminal by wrapping the code in
> `python -c '...'`. A few snippets are already written as
> `python -c "..."` so they drop straight into a shell — to reuse those
> in Jupyter, remove the `python -c "..."` wrapper and run the inner
> lines directly.

---

## Table of Contents

1. [Tutorial Files Overview](#1-tutorial-files-overview)
2. [Generating ROI Coordinates from NIfTI](#2-generating-roi-coordinates-from-nifti)
3. [Mapping ROI Subsets](#3-mapping-roi-subsets)
4. [Basic Connectivity Plot (28 ROIs)](#4-basic-connectivity-plot-28-rois)
5. [114-ROI Network with Metrics](#5-114-roi-network-with-metrics)
6. [Utility Commands](#6-utility-commands)
7. [Fixed Edge Width](#7-fixed-edge-width)
8. [Static Image Exports](#8-static-image-exports)
9. [Clean Exports (No Title/Legend)](#9-clean-exports-no-titlelegend)
10. [Node Visibility with Edge Toggling](#10-node-visibility-with-edge-toggling)
11. [Node Colors from Modules](#11-node-colors-from-modules)
12. [Modularity Visualization](#12-modularity-visualization)
13. [Vector Node Sizes from CSV](#13-vector-node-sizes-from-csv)
14. [Selectively Labelling ROIs](#14-selectively-labelling-rois)
15. [Cross-species montage grid (`hlplot montage`)](#15-cross-species-montage-grid-hlplot-montage)
16. [Scaling edges and nodes by p-value significance](#16-scaling-edges-and-nodes-by-p-value-significance)
17. [Command Reference](#17-command-reference)

---

## 1. Tutorial Files Overview

The `tutorial_files/` folder contains all data needed for this tutorial:

```
tutorial_files/
├── brain_atlas_170.nii          # NIfTI volume with 170 ROI labels
├── brain_mesh.gii               # Brain surface mesh (GIFTI format)
├── atlas_170_labels.txt         # Label file: 170 ROI names (index\tname)
├── atlas_170_coordinates.csv    # Pre-generated 170 ROI coordinates
├── atlas_114_labels.txt         # Label file: 114 ROI names (subset)
├── atlas_114_coordinates.csv    # Pre-generated 114 ROI coordinates
├── k5_state_0/
│   ├── connectivity_matrix.csv  # 114x114 connectivity matrix
│   ├── module_assignments.csv   # Module assignments for 114 ROIs
│   └── combined_metrics.csv     # Node metrics (PC, Z-scores, etc.)
└── node_edge_28/
    ├── rois_28.node             # BrainNet Viewer node file (28 ROIs)
    └── connectivity_28.edge     # BrainNet Viewer edge file (28x28)
```

### Setup: Create Output Directory

```bash
mkdir -p output
```

---

## 2. Generating ROI Coordinates from NIfTI

**This is the FIRST step when starting with a new atlas.**

Use `hlplot coords generate` to extract center-of-gravity (COG) coordinates from a NIfTI volume file.

> **⚠️ Float-labeled atlases.** Some atlases store integer ROI labels as
> floats with tiny rounding error (e.g. `0.9999999997` for label 1). An exact
> `volume == label` match then finds **zero voxels** and every COG comes out
> `NaN`. `coords generate` rounds labels by default (`--round-labels`), so this
> normally "just works". To check an atlas or pre-clean it:
> `hlplot utils info --volume atlas.nii.gz` (reports whether labels are
> bit-exact integers) and `hlplot utils clean-labels --volume atlas.nii.gz --output atlas_int.nii.gz`. See
> [ALIGNMENT_CHECKS.md](ALIGNMENT_CHECKS.md) for the full set of pre-flight
> checks (float labels, wrong template space, midline-collapsed atlases).

### Copy-Paste Command

```bash
hlplot coords generate \
  --volume brain_atlas_170.nii \
  --labels atlas_170_labels.txt \
  --output-dir output \
  --name my_170_coordinates
```

### Expected Output

Creates three files in `output/`:

- `my_170_coordinates_comma.csv` - Comma-delimited (use for plotting)
- `my_170_coordinates_tab.csv` - Tab-delimited
- `my_170_coordinates.pkl` - Python pickle

### Flag Explanations

| Flag             | Short  | Required | Description                                                      |
| ---------------- | ------ | -------- | ---------------------------------------------------------------- |
| `--volume`     | `-v` | Yes      | NIfTI file containing integer ROI labels (1-N)                   |
| `--labels`     | `-l` | Yes      | Text file mapping label numbers to names. Format:`1\tROI_Name` |
| `--output-dir` | `-o` | Yes      | Directory where output files will be saved                       |
| `--name`       | `-n` | No       | Base name for output files. Default:`roi_coordinates`          |

### Label File Format

The label file must be tab-delimited with format `index\tname`:

```
1	Acumbens_left
2	AID_left
3	AIP_left
...
170	VTA_right
```

---

## 3. Mapping ROI Subsets

When your connectivity matrix has fewer ROIs than your full atlas, use `hlplot coords map-subset` to extract matching coordinates.

`map-subset` finds each region's display coordinates by **matching ROI names**: every name in your subset file (`.node`, `.txt`, or `.csv`) must exactly match a `roi_name` in the `--coords` atlas. Any name that isn't found in both files is reported as unmatched and dropped.

### Understanding `map` vs `map-subset`

| Command               | Purpose                                              |
| --------------------- | ---------------------------------------------------- |
| `coords map`        | Transform coordinates: rename columns, apply scaling |
| `coords map-subset` | Extract a subset of ROIs by matching names           |

### Example A: Map 170 → 28 ROIs (using .node file)

The 28-ROI node file **must** contain ROI names that **also exist in** the 170-ROI atlas — `map-subset` matches by name to look up each region's coordinates.

```bash
hlplot coords map-subset \
  --coords atlas_170_coordinates.csv \
  --subset node_edge_28/rois_28.node \
  --output-dir output \
  --name atlas_28_mapped
```

### Example B: Map 170 → 114 ROIs (using .txt label file)

The 114-ROI label file **must** list ROI names that **also exist in** the 170-ROI atlas (it's a subset of them, with some tracts removed); the names are matched the same way.

```bash
hlplot coords map-subset \
  --coords atlas_170_coordinates.csv \
  --subset atlas_114_labels.txt \
  --output-dir output \
  --name atlas_114_mapped
```

### Expected Output

```
Summary: Successfully mapped 28 out of 28 ROIs
All ROIs were successfully mapped!
```

Creates (in a subdirectory with the same name):

- `output/atlas_28_mapped/atlas_28_mapped_comma.csv`
- `output/atlas_28_mapped/atlas_28_mapped_tab.csv`
- `output/atlas_28_mapped/atlas_28_mapped.pkl`

### Flag Explanations

| Flag             | Short  | Required | Description                                               |
| ---------------- | ------ | -------- | --------------------------------------------------------- |
| `--coords`     | `-c` | Yes      | Full coordinates CSV file (source atlas)                  |
| `--subset`     | `-s` | Yes      | Subset definition. Supports:`.node`, `.txt`, `.csv` |
| `--output-dir` | `-o` | Yes      | Output directory                                          |
| `--name`       | `-n` | No       | Output file name. Default:`mapped_roi_coordinates`      |

### Supported Subset File Formats

| Format    | Description                                         |
| --------- | --------------------------------------------------- |
| `.node` | BrainNet Viewer format. Uses last column (ROI name) |
| `.txt`  | One ROI name per line, or`index\tname` format     |
| `.csv`  | CSV with`roi_name` column                         |

---

## 4. Basic Connectivity Plot (28 ROIs)

Create a visualization with the 28-ROI network.

### Prerequisites

First, map the 28 ROI coordinates:

```bash
hlplot coords map-subset \
  --coords atlas_170_coordinates.csv \
  --subset node_edge_28/rois_28.node \
  --output-dir output \
  --name atlas_28_mapped
```

### Copy-Paste Command

```bash
hlplot plot \
  --mesh brain_mesh.gii \
  --coords output/atlas_28_mapped/atlas_28_mapped_comma.csv \
  --matrix node_edge_28/connectivity_28.edge \
  --output output/test1_basic_28roi.html \
  --title "28-ROI Brain Connectivity Network" \
  --node-size 10 \
  --edge-width-min 1.0 \
  --edge-width-max 10.0 \
  --camera superior \
  --hide-nodes-with-hidden-edges
```

![Basic 28-ROI connectivity network, superior view](../docs/images/cli_tutorial/04_basic_28roi.png)
*Static snapshot of `output/test1_basic_28roi.html` — 28 nodes, sign-colored edges scaled by `|weight|`, viewed from above.*

### Expected Output

- 28 nodes, 27 edges
- Positive edges in red, negative in blue
- Edge widths scaled by connection strength
- Clicking legend toggles edges AND nodes

### Flag Explanations

| Flag                               | Description                                                         |
| ---------------------------------- | ------------------------------------------------------------------- |
| `--mesh`, `-m`                 | Brain mesh file (.gii, .obj, .mz3, .ply)                            |
| `--coords`, `-c`               | ROI coordinates CSV (columns: cog_x, cog_y, cog_z, roi_name)        |
| `--matrix`, `-x`               | Connectivity matrix (.npy, .csv, .edge, .txt, .mat)                 |
| `--output`, `-o`               | Output HTML file                                                    |
| `--title`, `-t`                | Plot title                                                          |
| `--node-size`                    | Node size (number or CSV file path)                                 |
| `--edge-width-min`               | Minimum edge width when scaling                                     |
| `--edge-width-max`               | Maximum edge width when scaling                                     |
| `--camera`                       | View: oblique, anterior, posterior, left, right, superior, inferior |
| `--hide-nodes-with-hidden-edges` | Hide nodes when their edges are hidden                              |

---

## 5. 114-ROI Network with Metrics

Create a visualization with node metrics displayed on hover.

### Copy-Paste Command

```bash
hlplot plot \
  --mesh brain_mesh.gii \
  --coords atlas_114_coordinates.csv \
  --matrix k5_state_0/connectivity_matrix.csv \
  --output output/test2_114roi_metrics.html \
  --title "114-ROI Network with Metrics Hover" \
  --node-size 10 \
  --edge-width-min 0.5 \
  --edge-width-max 4.0 \
  --camera oblique \
  --node-metrics k5_state_0/combined_metrics.csv \
  --hide-nodes-with-hidden-edges
```

![114-ROI network with hover metrics](../docs/images/cli_tutorial/05_114roi_metrics.png)
*Static snapshot of `output/test2_114roi_metrics.html` — full 114-node network in oblique view. In the live HTML each node's hover tooltip shows `module`, `participation_coef`, `within_module_zscore`, etc. from `combined_metrics.csv`.*

### Expected Output

- 114 nodes, 452 edges
- Hovering over nodes shows: module, participation_coef, within_module_zscore, etc.

### Flag Explanations

| Flag               | Description                                                          |
| ------------------ | -------------------------------------------------------------------- |
| `--node-metrics` | CSV with node metrics. All columns shown on hover. One row per node. |

---

## 6. Utility Commands

### 6a. Check NIfTI ROI Count

Verify how many ROIs are in a NIfTI file:

```bash
python -c "
import nibabel as nib
import numpy as np
img = nib.load('brain_atlas_170.nii')
labels = np.unique(img.get_fdata())
labels = labels[labels != 0]
print(f'ROI count: {len(labels)}')
print(f'Label range: {labels.min():.0f} to {labels.max():.0f}')
"
```

*Shell-ready (`python -c "..."`): to run it in a Jupyter notebook or a `.py` file instead, drop the `python -c "..."` wrapper and keep the inner lines.*

### 6b. View Matrix Information

```bash
hlplot utils info --matrix node_edge_28/connectivity_28.edge
```

**Output shows:**

- Shape, non-zero values, density
- Min/max values, positive/negative edge counts
- Symmetry check

### 6c. View 114-ROI Matrix Info

```bash
hlplot utils info --matrix k5_state_0/connectivity_matrix.csv
```

### 6d. Validate File Compatibility

```bash
hlplot utils validate \
  --mesh brain_mesh.gii \
  --coords output/atlas_28_mapped/atlas_28_mapped_comma.csv \
  --matrix node_edge_28/connectivity_28.edge
```

### 6e. Inspect Coordinates File

```bash
hlplot coords load \
  --file atlas_170_coordinates.csv \
  --show-stats \
  --validate
```

### 6f. Map a node/edge subset into a full ROI matrix

`hlplot utils convert-node-edge` embeds a small BrainNet Viewer
`.node` + `.edge` pair into the larger `N × N` matrix defined by a
coordinates CSV (`--coords`). Edge values are placed by matching ROI
names; every unmatched row/column stays zero. The output lines up
row-for-row with the coords CSV, so it can be passed directly to
`hlplot plot --matrix` alongside the same coords CSV.

The `--coords` reference can be **any atlas size** — 114, 170, or a
custom list you generated yourself — as long as:

| Constraint                                                                         | Failure mode                                                                              |
| ---------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------- |
| `len(coords) ≥ len(node)` (coords has ≥ as many ROIs as the node file)         | `Coords CSV has X ROIs but .node file has Y`                                            |
| `edge.shape == (len(node), len(node))`                                           | `Edge matrix has X rows but .node file has Y entries`                                   |
| Every`roi_name` in the `.node` file appears in `coords`' `roi_name` column | `The following ROI names from the node file were not found in the ROI reference: [...]` |

#### Copy-Paste Command

```bash
hlplot utils convert-node-edge \
  --node node_edge_28/rois_28.node \
  --edge node_edge_28/connectivity_28.edge \
  --coords atlas_170_coordinates.csv \
  --output output/connectivity_28_in_170.csv
```

#### Expected Output

```
[OK] Loaded 28 nodes
[OK] Loaded (28, 28) edge matrix
[OK] Loaded 170 reference ROIs
[OK] Mapped 28 nodes into (170, 170) matrix
[OK] Saved (170, 170) matrix to output/connectivity_28_in_170.csv
```

The resulting `(170, 170)` matrix has 54 non-zero entries (the original
28×28 connections, placed at the rows/cols matching `atlas_170_coordinates.csv`).
You can plot it directly:

```bash
hlplot plot \
  --mesh brain_mesh.gii \
  --coords atlas_170_coordinates.csv \
  --matrix output/connectivity_28_in_170.csv \
  --output output/test_28_in_170.html \
  --camera superior
```

#### Flag Explanations

| Flag         | Short  | Required | Description                                          |
| ------------ | ------ | -------- | ---------------------------------------------------- |
| `--node`   | `-n` | Yes      | BrainNet Viewer node file (`.node`, 8-column)      |
| `--edge`   | `-e` | Yes      | BrainNet Viewer edge file (`.edge`, square matrix) |
| `--coords` | `-c` | Yes      | Full ROI coordinates CSV with a`roi_name` column   |
| `--output` | `-o` | Yes      | Output matrix path (`.csv` or `.npy`)            |

#### Note: name-overlap is required

The bundled 28-ROI file contains four ROIs (`S1_left`, `GIDI_right`,
`V1B_right`, `V1M_right`) that aren't present in `atlas_114_coordinates.csv`,
so attempting `--coords atlas_114_coordinates.csv` here will abort with
the missing names printed. To embed into a smaller atlas, make sure
every ROI in your `.node` file appears in that atlas first.

---

## 7. Fixed Edge Width

All edges same width (no scaling by weight).

### Copy-Paste Command

```bash
hlplot plot \
  --mesh brain_mesh.gii \
  --coords atlas_114_coordinates.csv \
  --matrix k5_state_0/connectivity_matrix.csv \
  --output output/test4_fixed_width.html \
  --title "Fixed Edge Width (2.0)" \
  --node-size 10 \
  --edge-width-fixed 2.0 \
  --camera anterior
```

![Fixed edge width: every line is 2 px](../docs/images/cli_tutorial/07_fixed_width.png)
*Static snapshot — every edge renders at the same 2-px width regardless of `|weight|`. Useful when significance is encoded by color rather than width.*

### Flag Explanations

| Flag                   | Description                                                 |
| ---------------------- | ----------------------------------------------------------- |
| `--edge-width-fixed` | Fixed width for ALL edges. Ignores`--edge-width-min/max`. |

---

## 8. Static Image Exports

Export publication-quality images alongside interactive HTML.

### 8a. PNG Export (300 DPI)

```bash
hlplot plot \
  --mesh brain_mesh.gii \
  --coords atlas_114_coordinates.csv \
  --matrix k5_state_0/connectivity_matrix.csv \
  --output output/test5a_png.html \
  --title "PNG Export (300 DPI)" \
  --camera superior \
  --export-image output/test5a_brain_network.png \
  --image-dpi 300
```

![PNG export — superior view, 114 ROIs](../docs/images/cli_tutorial/08a_png.png)
*Static snapshot of the same plot exported as a PNG. Sharp at any zoom; `--image-dpi` controls effective resolution.*

### 8b. SVG Export (Vector)

```bash
hlplot plot \
  --mesh brain_mesh.gii \
  --coords atlas_114_coordinates.csv \
  --matrix k5_state_0/connectivity_matrix.csv \
  --output output/test5b_svg.html \
  --title "SVG Export (Vector)" \
  --camera oblique \
  --export-image output/test5b_brain_network.svg
```

![SVG export preview — oblique view](../docs/images/cli_tutorial/08b_svg.png)
*PNG preview of the SVG content. The actual SVG output is resolution-independent (vectors), great for publication figures and arbitrary zoom.*

### 8c. PDF Export (Publication)

```bash
hlplot plot \
  --mesh brain_mesh.gii \
  --coords atlas_114_coordinates.csv \
  --matrix k5_state_0/connectivity_matrix.csv \
  --output output/test5c_pdf.html \
  --title "PDF Export" \
  --camera anterior \
  --export-image output/test5c_brain_network.pdf
```

![PDF export preview — anterior view](../docs/images/cli_tutorial/08c_pdf.png)
*PNG preview of the PDF export. Same vector quality as SVG, embeds cleanly in LaTeX/Word.*

### Flag Explanations

| Flag               | Description                                                           |
| ------------------ | --------------------------------------------------------------------- |
| `--export-image` | Output path. Extension determines format (.png, .svg, .pdf)           |
| `--image-dpi`    | DPI for PNG. No hard cap (very high = very large image). Default: 300 |
| `--image-format` | Format if path has no extension                                       |

### 8d. Custom / Transparent Background

By default the background is white. `--background-color` sets any background
color — a named color, a hex code, or `transparent` for a transparent PNG. It
applies to **both** the saved interactive HTML and the static export.

```bash
# Transparent PNG (alpha channel) — drops onto any slide / poster background
hlplot plot \
  --mesh brain_mesh.gii \
  --coords output/atlas_28_mapped/atlas_28_mapped_comma.csv \
  --matrix node_edge_28/connectivity_28.edge \
  --output output/test_transparent.html \
  --camera oblique \
  --export-image output/test_transparent.png \
  --background-color transparent

# Custom solid color (named color or hex)
hlplot plot \
  --mesh brain_mesh.gii \
  --coords output/atlas_28_mapped/atlas_28_mapped_comma.csv \
  --matrix node_edge_28/connectivity_28.edge \
  --output output/test_dark.html \
  --camera oblique \
  --export-image output/test_dark.png \
  --background-color "#1e1e1e"
```

![Transparent-background export shown over a checkerboard](../docs/images/static_export/transparent_bg_demo.png)
*`--background-color transparent` produces a real RGBA PNG (here composited over
a checkerboard so the transparency is visible). Works for single views and
multi-view stitched strips, and the same flag exists on `hlplot modular`.*

| Flag                   | Description                                                                                                    |
| ---------------------- | -------------------------------------------------------------------------------------------------------------- |
| `--background-color` | Background color: a name, hex (`#1e1e1e`), or `transparent`. Applies to HTML + export. Default: `white`. |

---

### 8e. Export Canvas Size and Tight Crops

Single-image exports render on a **square 1200×1200 canvas by default**, which
gives even margins on both sides *and* keeps the 3D aspect stable as you change
`--image-dpi`.

```bash
# Default 1200x1200
hlplot plot ... --export-image brain.png

# Bigger square canvas
hlplot plot ... --export-image brain.png --export-size "1600,1600"

# Trim tight to the content instead of even margins (opt-in)
hlplot plot ... --export-image brain.png --export-autocrop
```

> **Keep width == height if you plan to change `--image-dpi`.** `--image-dpi` is a
> supersampling factor (`min(dpi/72, 8)`) applied to this canvas. On a **non-square**
> canvas kaleido renders the 3D scene with a *scale-dependent* aspect, so the
> brain's proportions shift between DPIs — a non-square `--export-size` therefore
> prints a warning. On the square default the aspect is identical at 150/300/600.

#### Flag Explanations

| Flag                                             | Description                                                                                                                                                                 |
| ------------------------------------------------ | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `--export-size`                                | Export canvas as`'width,height'`. Default `'1200,1200'`. Keep it square if you change `--image-dpi`.                                                                  |
| `--export-autocrop` / `--no-export-autocrop` | Trim the background border tight to the content. Default**off** (even margins). Pure crop — never warps the aspect. Raster only; multi-view already crops per panel. |

---

## 9. Clean Exports (No Title/Legend)

For publication figures where you add your own caption.

### 9a. Clean PNG

```bash
hlplot plot \
  --mesh brain_mesh.gii \
  --coords output/atlas_28_mapped/atlas_28_mapped_comma.csv \
  --matrix node_edge_28/connectivity_28.edge \
  --output output/test6a_clean.html \
  --title "This title will NOT appear" \
  --camera superior \
  --export-image output/test6a_clean.png \
  --export-no-title \
  --export-no-legend \
  --image-dpi 150
```

![Clean PNG: no title, no legend](../docs/images/cli_tutorial/09a_clean.png)
*Static snapshot — title and legend are stripped from the export so the figure drops straight into a paper with your own caption.*

### 9b. Title Only (No Legend)

```bash
hlplot plot \
  --mesh brain_mesh.gii \
  --coords output/atlas_28_mapped/atlas_28_mapped_comma.csv \
  --matrix node_edge_28/connectivity_28.edge \
  --output output/test6b_title_only.html \
  --title "Brain Connectivity Network" \
  --camera oblique \
  --export-image output/test6b_title_only.png \
  --export-no-legend
```

![Title kept, legend stripped](../docs/images/cli_tutorial/09b_title_only.png)
*Static snapshot — `--export-no-legend` removes the legend but keeps the title.*

### 9c. Clean PDF (Publication)

```bash
hlplot plot \
  --mesh brain_mesh.gii \
  --coords output/atlas_28_mapped/atlas_28_mapped_comma.csv \
  --matrix node_edge_28/connectivity_28.edge \
  --output output/test6c_clean.html \
  --camera left \
  --export-image output/test6c_clean.pdf \
  --export-no-title \
  --export-no-legend
```

![Clean PDF preview — left lateral view](../docs/images/cli_tutorial/09c_clean_pdf.png)
*PNG preview of the clean PDF export — same plot from the left lateral view, no title or legend, ready for a publication panel.*

### Flag Explanations

| Flag                   | Description                        |
| ---------------------- | ---------------------------------- |
| `--export-no-title`  | Exclude title from exported image  |
| `--export-no-legend` | Exclude legend from exported image |

---

## 10. Node Visibility with Edge Toggling

Nodes can be toggled along with their edges in the interactive legend.

### Copy-Paste Command

```bash
hlplot plot \
  --mesh brain_mesh.gii \
  --coords output/atlas_28_mapped/atlas_28_mapped_comma.csv \
  --matrix node_edge_28/connectivity_28.edge \
  --output output/test7_node_visibility.html \
  --title "Node Visibility with Edge Toggle" \
  --node-size 12 \
  --edge-width-min 1.0 \
  --edge-width-max 8.0 \
  --camera superior \
  --hide-nodes-with-hidden-edges
```

![Node visibility toggling](../docs/images/cli_tutorial/10_node_visibility.png)
*Static snapshot — initial state with all nodes and edges visible. In the live HTML, clicking a legend entry hides both the edges AND their connected nodes.*

### Interactive Behavior

1. Click "Positive Edges" in legend → Positive edges AND their nodes hide
2. Click "Negative Edges" in legend → Negative edges AND their nodes hide
3. Click BOTH → ALL nodes and edges hide (only brain surface remains)

---

## 11. Node Colors from Modules

Color nodes by module/community assignment.

### 11a. Using Module CSV File

```bash
hlplot plot \
  --mesh brain_mesh.gii \
  --coords atlas_114_coordinates.csv \
  --matrix k5_state_0/connectivity_matrix.csv \
  --output output/test8a_module_colors.html \
  --title "Node Colors from Modules" \
  --node-size 10 \
  --node-color k5_state_0/module_assignments.csv \
  --node-border-color darkgray \
  --camera oblique
```

![Module-colored nodes](../docs/images/cli_tutorial/11_module_colors.png)
*Static snapshot — passing a CSV of module assignments to `--node-color` auto-generates 6 distinct colors (one per module). Nodes belonging to the same module render in the same color.*

### Expected Output

- Nodes colored by module (6 distinct colors for 6 modules)
- Colors auto-generated: Module 1=Red, 2=Green, 3=Blue, etc.

### Flag Explanations

| Flag                    | Description                                                        |
| ----------------------- | ------------------------------------------------------------------ |
| `--node-color`        | Accepts: color name, hex code, or CSV path with module assignments |
| `--node-border-color` | Border color for nodes                                             |

### Module CSV Format

```csv
roi_index,module
0,1
1,2
2,1
3,3
...
```

---

## 12. Modularity Visualization

Use `hlplot modular` for dedicated modularity analysis.

### 12a. With Q and Z Scores

```bash
hlplot modular \
  --mesh brain_mesh.gii \
  --coords atlas_114_coordinates.csv \
  --matrix k5_state_0/connectivity_matrix.csv \
  --modules k5_state_0/module_assignments.csv \
  --output output/test9a_modularity.html \
  --title "Brain Network Modularity" \
  --q-score 0.452 \
  --z-score 3.21 \
  --node-size 10 \
  --camera oblique
```

![Modularity plot with Q and Z scores in the title](../docs/images/cli_tutorial/12a_q_z.png)
*Static snapshot — title automatically becomes `Brain Network Modularity (Q=0.452, Z=3.21)`. Default edge coloring is sign mode (red/blue).*

**Output title:** "Brain Network Modularity (Q=0.452, Z=3.21)"

> **About `--q-score` / `--z-score`:** these are display-only values
> printed in the title — `hlplot` does **not** compute them. You obtain
> them from your community-detection / modularity routine (for example
> **netneurotools**' modularity functions, which return the modularity
> quality **Q** and a partition-stability **z-rand** score) and pass the
> resulting scalars in. Because they describe the partition as a whole,
> they belong in the title rather than on any single node or edge.
>
> This is the only thing that sets §12a apart from the variants below:
> §12a adds Q/Z to the title (with the default sign-based edge
> coloring), while §12b and §12c instead demonstrate the two
> **edge-color modes** (`--edge-color-mode module` vs `sign`) and don't
> pass Q/Z.

### 12b. Module-Colored Edges

Edges colored by source node's module instead of positive/negative sign:

```bash
hlplot modular \
  --mesh brain_mesh.gii \
  --coords atlas_114_coordinates.csv \
  --matrix k5_state_0/connectivity_matrix.csv \
  --modules k5_state_0/module_assignments.csv \
  --output output/test9b_module_edges.html \
  --title "Module-Colored Edges" \
  --edge-color-mode module \
  --node-size 10 \
  --camera anterior
```

![Module-colored edges, anterior view](../docs/images/cli_tutorial/12b_module_edges.png)
*Static snapshot — edges inherit their source module's color, so the within-module structure pops visually.*

### 12c. Sign-Colored Edges (Default)

```bash
hlplot modular \
  --mesh brain_mesh.gii \
  --coords atlas_114_coordinates.csv \
  --matrix k5_state_0/connectivity_matrix.csv \
  --modules k5_state_0/module_assignments.csv \
  --output output/test9c_sign_edges.html \
  --title "Sign-Colored Edges" \
  --edge-color-mode sign \
  --node-size 10 \
  --camera oblique
```

![Sign-colored edges with module-colored nodes](../docs/images/cli_tutorial/12c_sign_edges.png)
*Static snapshot — same modular plot but edges keep the conventional pos/neg sign coloring. Nodes still grouped and colored by module.*

### Flag Explanations

| Flag                  | Description                                        |
| --------------------- | -------------------------------------------------- |
| `--modules`, `-d` | Module assignments file (required)                 |
| `--q-score`         | Modularity Q score for title                       |
| `--z-score`         | Z-rand score for title                             |
| `--edge-color-mode` | `sign` (red/blue) or `module` (by source node) |

---

## 13. Vector Node Sizes from CSV

Node sizes can be loaded from a CSV or NPY file, allowing different sizes for each node.

### Create a Node Sizes CSV

First, create a CSV with per-node sizes based on a metric (e.g., participation coefficient):

```bash
python -c "
import pandas as pd
import numpy as np

# Load metrics
metrics = pd.read_csv('k5_state_0/combined_metrics.csv')

# Scale participation coefficient to node sizes 5-20
pc = metrics['participation_coef'].values
sizes = 5 + (pc / pc.max()) * 15

# Save as CSV (single column)
pd.DataFrame({'size': sizes}).to_csv('output/node_sizes_by_pc.csv', index=False)
print(f'Created node_sizes_by_pc.csv with {len(sizes)} sizes')
print(f'Size range: {sizes.min():.1f} to {sizes.max():.1f}')
"
```

*Shell-ready (`python -c "..."`): to run it in a Jupyter notebook or a `.py` file instead, drop the `python -c "..."` wrapper and keep the inner lines.*

### Use Vector Sizes in Plot

```bash
hlplot plot \
  --mesh brain_mesh.gii \
  --coords atlas_114_coordinates.csv \
  --matrix k5_state_0/connectivity_matrix.csv \
  --output output/test_vector_sizes.html \
  --title "Node Size = Participation Coefficient" \
  --node-size output/node_sizes_by_pc.csv \
  --node-metrics k5_state_0/combined_metrics.csv \
  --edge-width-min 0.5 \
  --edge-width-max 4.0 \
  --camera superior
```

![Per-node sizes from CSV (PC-scaled)](../docs/images/cli_tutorial/13_vector_sizes.png)
*Static snapshot — node sizes come from `output/node_sizes_by_pc.csv`, a per-node CSV mapping participation coefficient to a 5-20 px range. Nodes with higher PC are visibly larger.*

### Node Size Input Options

| Input Type    | Example                   | Description                      |
| ------------- | ------------------------- | -------------------------------- |
| Single number | `--node-size 10`        | All nodes same size              |
| CSV file      | `--node-size sizes.csv` | One size per node (first column) |
| NPY file      | `--node-size sizes.npy` | NumPy array with sizes           |

---

## 14. Selectively Labelling ROIs

Dense networks quickly become unreadable when every ROI gets a text label.
The `--show-node-labels` flag lets you label only the regions you want
to call out — for example, only the network's hub nodes — and hide the
rest, while still letting readers hover over any unlabelled node in the
saved HTML to discover its name. Hover tooltips are completely
independent of this flag.

The flag accepts three forms:

| Form                                      | Effect                                                     |
| ----------------------------------------- | ---------------------------------------------------------- |
| `--show-node-labels true` *(default)* | Every ROI gets a persistent label.                         |
| `--show-node-labels false`              | No persistent labels; hover still reveals names.           |
| `--show-node-labels path/to/mask.csv`   | Per-node 0/1 mask —`1` shows the label, `0` hides it. |

The same flag exists on `hlplot modular`, with identical semantics.

### 14a. Default — every ROI labelled

```bash
hlplot plot \
  --mesh brain_mesh.gii \
  --coords output/atlas_28_test_comma.csv \
  --matrix node_edge_28/connectivity_28.edge \
  --output output/labels_default.html \
  --export-image output/labels_default.png \
  --image-dpi 150
```

![28-ROI network with every ROI labelled (default)](../docs/images/cli_tutorial/14a_labels_default.png)
*Default `--show-node-labels true`: 28 ROI labels render at once. Useful
for quick exploration, but with many nodes the labels overlap and crowd
the figure, making it hard to read at publication scale.*

### 14b. All labels off

```bash
hlplot plot \
  --mesh brain_mesh.gii \
  --coords output/atlas_28_test_comma.csv \
  --matrix node_edge_28/connectivity_28.edge \
  --show-node-labels false \
  --output output/labels_off.html \
  --export-image output/labels_off.png \
  --image-dpi 150
```

![28-ROI network with no persistent labels](../docs/images/cli_tutorial/14b_labels_off.png)
*With `--show-node-labels false` the brain renders cleanly with no text
clutter. The saved HTML still shows ROI names on hover.*

### 14c. Label only the hub nodes (mask CSV)

The repository ships an example mask that labels only the six
highest-degree ROIs in `connectivity_28.edge`:

```bash
hlplot plot \
  --mesh brain_mesh.gii \
  --coords output/atlas_28_test_comma.csv \
  --matrix node_edge_28/connectivity_28.edge \
  --show-node-labels node_edge_28/show_labels_hubs_28.csv \
  --output output/labels_hubs.html \
  --export-image output/labels_hubs.png \
  --image-dpi 150
```

![28-ROI network with only the six hub ROIs labelled](../docs/images/cli_tutorial/14c_labels_hubs.png)
*With a per-node mask CSV, only the six top-degree hubs (V1_left,
PtPD_left, SaA_right, SaM_right, Thalamus_A_right, SaA_left) are
labelled. Every other node still appears, edges are unaffected, and
hover tooltips still surface the unlabelled ROI names on demand.*

### CSV mask format

The mask is a single-column file with one row per ROI in the same
order as your coordinates and matrix. The header is optional; when
present, name it `show_label`:

```csv
show_label
0
1
0
0
0
0
0
1
... (28 rows total)
```

Values can be `0` / `1`, `True` / `False`, or boolean. The file may be
a `.csv`, `.txt` (tab- or comma-delimited, with or without header),
or a `.npy` array.

### Building your own mask

Any selection rule that produces a length-`N` 0/1 vector works. For the
canonical "label only the top-`k` hubs" recipe:

```python
import numpy as np
import pandas as pd

matrix = np.loadtxt("node_edge_28/connectivity_28.edge", delimiter="\t")
abs_mat = np.abs(matrix)
np.fill_diagonal(abs_mat, 0)
degree = (abs_mat > 0).sum(axis=0)

TOP = 6
hubs = np.argsort(degree)[::-1][:TOP]

mask = np.zeros(len(degree), dtype=int)
mask[hubs] = 1
pd.DataFrame({"show_label": mask}).to_csv(
    "node_edge_28/show_labels_hubs_28.csv", index=False
)
```

*Python snippet: paste it into a Jupyter notebook or a `.py` file as-is, or run it from a shell by wrapping it in `python -c '...'`.*

Other useful patterns:

- **One hemisphere only**: `mask = np.array(["_left" in name for name in roi_names], dtype=int)`.
- **One module only**: `mask = (modules == 1).astype(int)`.
- **Hand-picked ROIs**: edit the CSV by hand and write `1` next to the
  rows you want to call out.

### Python API

The same parameter is available directly on the Python plotting
functions and accepts `True`, `False`, a numpy array / list / pandas
Series of 0/1, or a CSV path.

**How to run these examples.** The snippet below is **Python**, not a
shell command — paste it into a Jupyter notebook cell, or save it to a
`.py` file and run `python my_script.py`. It assumes your working
directory is `test_files/tutorial_files` (the same directory the CLI
commands above run from) so the relative paths resolve. To run it
straight from a terminal without saving a file, hand it to the
interpreter with `python -c '...'`: `python -c` tells Python to execute
the string as code (the shell can't run Python on its own), and the
**single** outer quotes keep the snippet's double-quoted paths intact.

```bash
python -c '
import pandas as pd
from HarrisLabPlotting import load_mesh_file, create_brain_connectivity_plot
vertices, faces = load_mesh_file("brain_mesh.gii")
coords = pd.read_csv("output/atlas_28_test_comma.csv")
create_brain_connectivity_plot(
    vertices=vertices, faces=faces, roi_coords_df=coords,
    connectivity_matrix="node_edge_28/connectivity_28.edge",
    show_node_labels="node_edge_28/show_labels_hubs_28.csv",
    save_path="output/labels_hubs.html")
'
```

The full Jupyter / `.py` version, showing both label-mask forms:

```python
import numpy as np
import pandas as pd
from HarrisLabPlotting import (
    load_mesh_file,
    create_brain_connectivity_plot,
    create_brain_connectivity_plot_with_modularity,
)

# Run from test_files/tutorial_files so these relative paths resolve.
vertices, faces = load_mesh_file("brain_mesh.gii")
coords = pd.read_csv("output/atlas_28_test_comma.csv")

# Labels only on the hub ROIs (CSV mask path)
fig, _ = create_brain_connectivity_plot(
    vertices=vertices, faces=faces, roi_coords_df=coords,
    connectivity_matrix="node_edge_28/connectivity_28.edge",
    show_node_labels="node_edge_28/show_labels_hubs_28.csv",
    save_path="output/labels_hubs.html",
)
fig.show()  # in a Jupyter notebook this renders the figure inline

# Equivalent with an explicit boolean array
mask = np.zeros(28, dtype=bool)
mask[[1, 7, 16, 17, 18, 24]] = True  # the same six indices

fig, _ = create_brain_connectivity_plot_with_modularity(
    vertices=vertices, faces=faces, roi_coords_df=coords,
    connectivity_matrix="node_edge_28/connectivity_28.edge",
    module_assignments="node_edge_28/modules_28.csv",
    show_node_labels=mask,
    save_path="output/labels_hubs_modular.html",
)
fig.show()
```

> **Note:** `show_node_labels` only affects the *persistent* text
> label that draws next to each node marker. The hover tooltip in the
> saved HTML always shows the full ROI name, module, and any node
> metrics, regardless of mask.

---

## 15. Cross-species montage grid (`hlplot montage`)

`--multi-view` (section 8-style stitching) renders **one** mesh from several
cameras. When you need a grid whose **columns are different meshes** — e.g. a
human, a rat and a macaque brain — render each panel separately, then compose
them with `hlplot montage`. It is a general image-grid composer: give it a
row-major list of pre-rendered PNGs and a grid shape, and it auto-crops each
panel and adds column headers, per-cell labels, and an optional title.

```bash
# Step 1 — render each species+view panel (single view, clean, no legend).
# Repeat per species/view; here is one panel (human, left lateral):
hlplot modular \
  --mesh "parcellation and meshes/HCPMMP1_on_MNI152_ICBM2009a_nlin_hd_0.obj" \
  --coords new_atlas_demo/human/hcpmmp1_coords.csv \
  --matrix new_atlas_demo/human/hcpmmp1_modular_network.csv \
  --modules new_atlas_demo/human/hcpmmp1_modules.csv \
  --node-size 10 --edge-width-fixed 2 \
  --no-width-legend --export-no-legend --title "" \
  --multi-view "left" --multi-view-panel-size "500,500" \
  --multi-view-no-first-legend \
  --image-dpi 600 --zoom 1.3 \
  --output dummy.html --export-image human_left.png

# Step 2 — compose the six panels (row-major) into a 2x3 grid.
hlplot montage \
  --images "human_left.png,rat_superior.png,macaque_right.png,human_anterior.png,rat_inferior.png,macaque_posterior.png" \
  --grid "2,3" \
  --col-labels "Human,Rat,Macaque" \
  --panel-labels "Left,Superior,Right,Anterior,Inferior,Posterior" \
  --output species_grid.png
```

### Flag Explanations (`hlplot montage`)

| Flag                   | Description                                                                                                       |
| ---------------------- | ----------------------------------------------------------------------------------------------------------------- |
| `--images` / `-i`  | Comma-separated panel PNG paths, in**row-major** order (left-to-right, then top-to-bottom).                 |
| `--grid`             | Grid shape`'rows,cols'` (e.g. `'2,3'`). Omit for a single row. `rows*cols` must be ≥ the number of images. |
| `--col-labels`       | One header per column, drawn once along the top.                                                                  |
| `--row-labels`       | One label per row, drawn in a left gutter.                                                                        |
| `--panel-labels`     | One label per image, drawn below each panel.                                                                      |
| `--title`            | Combined title above the whole grid.                                                                              |
| `--background-color` | Named color, hex, or`transparent` (RGBA output).                                                                |
| `--no-autocrop`      | Keep each panel's original border instead of trimming it.                                                         |

The Python equivalent is `compose_image_grid(images, output, grid=(2,3), col_labels=[...], panel_labels=[...])`. See
[FIGURE_CREATION.md](FIGURE_CREATION.md) for the full human/rat/macaque example.

---

## 16. Scaling edges and nodes by p-value significance

With `--matrix-type pvalue`, edge width already scales with `-log10(p)` (thicker
= more significant). You can encode significance on the **nodes** too by deriving
a per-node significance vector and passing it as `--node-size`. The contrast
below is uniform (nothing scaled) vs. edges **and** nodes scaled by significance.

```bash
# (a) uniform baseline: fixed edge width + scalar node size (nothing encodes p)
hlplot plot \
  --mesh brain_mesh.gii \
  --coords output/atlas_28_test_comma.csv \
  --matrix node_edge_28/pvalues_28.csv \
  --matrix-type pvalue --pvalue-threshold 0.05 \
  --edge-width-fixed 2 --node-size 8 \
  --camera superior --image-dpi 600 \
  --output pval_uniform.html --export-image pval_uniform.png

# (b) scaled: edge width ~ -log10(p) AND node size ~ per-node significance.
# Derive the per-node size CSV first (sum of -log10(p) over surviving edges):
python -c "
import numpy as np, pandas as pd
P = np.loadtxt('node_edge_28/pvalues_28.csv', delimiter=',')
W = np.where((P>0)&(P<=0.05), -np.log10(np.clip(P,1e-300,1.0)), 0.0); np.fill_diagonal(W,0)
sig = W.sum(1); px = 6 + (sig-sig.min())/(sig.max()-sig.min())*(24-6)
pd.DataFrame({'size': px}).to_csv('node_sig_sizes.csv', index=False)"

hlplot plot \
  --mesh brain_mesh.gii \
  --coords output/atlas_28_test_comma.csv \
  --matrix node_edge_28/pvalues_28.csv \
  --matrix-type pvalue --pvalue-threshold 0.05 \
  --edge-width-min 1 --edge-width-max 9 \
  --node-size node_sig_sizes.csv \
  --camera superior --image-dpi 600 \
  --output pval_scaled.html --export-image pval_scaled.png
```

The uniform figure draws every edge/node identically; the scaled figure makes the
most significant connections thick and their nodes large. See the
[p-value plotting tutorial](PVALUE_PLOTTING_TUTORIAL.md) for the full flag set.

---

## 17. Command Reference

### Main Commands

```bash
hlplot --help              # Main help
hlplot plot --help         # Connectivity plot
hlplot modular --help      # Modularity visualization
hlplot montage --help      # Compose pre-rendered PNGs into a grid
hlplot batch --help        # Batch processing
hlplot coords --help       # Coordinate utilities
hlplot utils --help        # Matrix utilities
```

### Coordinate Commands

```bash
hlplot coords generate --help    # Extract coords from NIfTI
hlplot coords map-subset --help  # Map ROI subset
hlplot coords load --help        # Inspect coordinates
hlplot coords extract --help     # Simple extraction (no labels)
```

### Utility Commands

```bash
hlplot utils info --help       # Matrix information
hlplot utils validate --help   # Validate file compatibility
hlplot utils threshold --help  # Threshold matrix
hlplot utils convert --help    # Convert file formats
```

### Camera View Presets

| View              | Description         |
| ----------------- | ------------------- |
| `oblique`       | Default angled view |
| `anterior`      | Front view          |
| `posterior`     | Back view           |
| `left`          | Left side           |
| `right`         | Right side          |
| `superior`      | Top (dorsal)        |
| `inferior`      | Bottom (ventral)    |
| `lateral-left`  | Left lateral        |
| `lateral-right` | Right lateral       |

---

## Complete Pipeline Example

Run the full pipeline from NIfTI to visualization:

```bash
# 1. Create output directory
mkdir -p output

# 2. Generate 170 ROI coordinates from NIfTI (optional - already provided)
hlplot coords generate \
  --volume brain_atlas_170.nii \
  --labels atlas_170_labels.txt \
  --output-dir output \
  --name atlas_170_generated

# 3. Map to 28-ROI subset
hlplot coords map-subset \
  --coords atlas_170_coordinates.csv \
  --subset node_edge_28/rois_28.node \
  --output-dir output \
  --name atlas_28

# 4. Create basic visualization
hlplot plot \
  --mesh brain_mesh.gii \
  --coords output/atlas_28/atlas_28_comma.csv \
  --matrix node_edge_28/connectivity_28.edge \
  --output output/brain_28.html \
  --title "28-ROI Brain Network"

# 5. Map to 114-ROI subset
hlplot coords map-subset \
  --coords atlas_170_coordinates.csv \
  --subset atlas_114_labels.txt \
  --output-dir output \
  --name atlas_114

# 6. Create modularity visualization
hlplot modular \
  --mesh brain_mesh.gii \
  --coords output/atlas_114/atlas_114_comma.csv \
  --matrix k5_state_0/connectivity_matrix.csv \
  --modules k5_state_0/module_assignments.csv \
  --output output/modularity_114.html \
  --title "114-ROI Modularity" \
  --q-score 0.452 \
  --z-score 3.21

# 7. Export publication figure
hlplot plot \
  --mesh brain_mesh.gii \
  --coords output/atlas_114/atlas_114_comma.csv \
  --matrix k5_state_0/connectivity_matrix.csv \
  --output output/publication.html \
  --node-color k5_state_0/module_assignments.csv \
  --camera anterior \
  --export-image output/publication.pdf \
  --export-no-title \
  --export-no-legend
```

---

*Generated from the Jupyter notebook: `bcp update to include node size vector test, edge width scaled to matrix, utils, etc.ipynb`*
