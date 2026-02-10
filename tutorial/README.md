# HarrisLabPlotting Tutorial

A comprehensive guide to brain connectivity visualization using the `hlplot` command-line tool.

---

## Table of Contents

1. [Installation](#1-installation)
2. [Data Pipeline Overview](#2-data-pipeline-overview)
3. [Step 1: Creating a Brain Mesh from NIfTI](#3-step-1-creating-a-brain-mesh-from-nifti)
4. [Step 2: Generating ROI Coordinates](#4-step-2-generating-roi-coordinates)
5. [Step 3: Mapping ROI Subsets (Optional)](#5-step-3-mapping-roi-subsets-optional)
6. [Step 4: Creating Connectivity Plots](#6-step-4-creating-connectivity-plots)
7. [Modularity Visualization](#7-modularity-visualization)
8. [Node and Edge Customization](#8-node-and-edge-customization)
9. [Static Image Export](#9-static-image-export)
10. [Camera Views](#10-camera-views)
11. [Batch Processing](#11-batch-processing)
12. [Complete Parameter Reference](#12-complete-parameter-reference)
13. [Troubleshooting](#13-troubleshooting)

---

## 1. Installation

### Step 1: Clone the Repository

First, download the HarrisLabPlotting package from GitHub:

```bash
git clone https://github.com/AzadAzargushasb/HarrisLabPlotting.git
cd HarrisLabPlotting
```

### Step 2: Create the Conda Environment

The repository includes an `environment.yml` file with all required dependencies:

```bash
# Create the conda environment from the yml file
conda env create -f environment.yml

# Activate the environment
conda activate harris_lab_plotting
```

### Step 3: Install the Package

With the conda environment activated, install the package in development mode:

```bash
pip install -e .
```

### Step 4: Verify Installation

Test that the installation was successful:

```bash
# Check version
hlplot --version

# View available commands
hlplot --help
```

You should see the HarrisLabPlotting CLI help with available commands.

---

## 2. Data Pipeline Overview

### What You Need

To create brain connectivity visualizations, you ultimately need:

1. **A brain mesh file** (.gii, .obj, .ply, .mz3) - The 3D brain surface
2. **ROI coordinates file** (.csv) - X, Y, Z positions for each brain region
3. **Connectivity matrix** (.npy, .csv, .edge) - Connection strengths between ROIs

### Starting Point: NIfTI Volume File

If you're starting from scratch, you typically have a **NIfTI volume file** (.nii or .nii.gz) containing your brain atlas/parcellation. From this single file, you can generate both:

- The brain **mesh** (using external tools)
- The ROI **coordinates** (using this package)

### Complete Pipeline

```
                      ┌─────────────────────────┐
                      │   NIfTI Volume File     │
                      │   (brain_atlas.nii.gz)  │
                      └───────────┬─────────────┘
                                  │
           ┌──────────────────────┼──────────────────────┐
           │                      │                      │
           ▼                      ▼                      │
    ┌─────────────┐      ┌─────────────────┐            │
    │ Create Mesh │      │ Generate ROI    │            │
    │ (Surfice or │      │ Coordinates     │            │
    │ nii2mesh)   │      │ (hlplot coords  │            │
    └──────┬──────┘      │  generate)      │            │
           │             └────────┬────────┘            │
           │                      │                     │
           │                      ▼                     │
           │             ┌─────────────────┐            │
           │             │ Map to Subset   │            │
           │             │ (Optional)      │            │
           │             │ (hlplot coords  │            │
           │             │  map-subset)    │            │
           │             └────────┬────────┘            │
           │                      │                     │
           └──────────┬───────────┘                     │
                      │                                 │
                      ▼                                 │
              ┌───────────────┐     ┌───────────────┐   │
              │  Mesh File    │     │ Connectivity  │◄──┘
              │  (brain.gii)  │     │ Matrix        │ (your data)
              └───────┬───────┘     └───────┬───────┘
                      │                     │
                      └──────────┬──────────┘
                                 │
                                 ▼
                      ┌─────────────────────┐
                      │ hlplot plot/modular │
                      │ (Visualization)     │
                      └─────────────────────┘
```

---

## 3. Step 1: Creating a Brain Mesh from NIfTI

Before you can visualize brain connectivity, you need a 3D mesh of the brain surface.

**See the detailed guide:** [MESH_CREATION_GUIDE.md](MESH_CREATION_GUIDE.md)

### Quick Options

#### Option A: nii2mesh Web Tool (Easiest)

1. Go to https://rordenlab.github.io/nii2meshWeb/
2. Upload your NIfTI file
3. Click "Convert" and download the mesh

#### Option B: Surfice (Recommended for Quality)

Download from: https://www.nitrc.org/projects/surfice/

Surfice provides better control over mesh quality and smoothing.

#### Option C: nii2mesh CLI (For Automation)

```bash
# See https://github.com/neurolabusc/nii2mesh
nii2mesh -s 3 brain_atlas.nii.gz brain_mesh.gii
```

### Supported Mesh Formats

| Format | Extension | Notes |
|--------|-----------|-------|
| GIFTI  | .gii      | Recommended, standard neuroimaging format |
| Wavefront OBJ | .obj | Common 3D format |
| PLY    | .ply      | Polygon file format |
| MZ3    | .mz3      | Surfice native format |

---

## 4. Step 2: Generating ROI Coordinates

Once you have your NIfTI volume file, use `hlplot coords generate` to extract the center-of-gravity (COG) coordinates for each ROI.

### Requirements

1. **NIfTI volume file** - Your brain atlas with integer labels for each ROI
2. **Label file** - A text file mapping ROI numbers to names

### Label File Format

Create a tab-delimited text file with ROI numbers and names:

```
1	Acumbens_left
2	AID_left
3	AIP_left
4	AIV_left
5	Amygdala_left
...
```

### Generate Coordinates

```bash
hlplot coords generate \
  --volume brain_atlas.nii.gz \
  --labels roi_labels.txt \
  --output-dir ./roi_coordinates \
  --name atlas_170_coordinates
```

### Output Files

This creates three files in your output directory:

- `atlas_170_coordinates_comma.csv` - Comma-delimited (use this for plotting)
- `atlas_170_coordinates_tab.csv` - Tab-delimited
- `atlas_170_coordinates.pkl` - Python pickle format

### Example Output CSV

```csv
roi_index,roi_name,cog_x,cog_y,cog_z,cog_voxel_i,cog_voxel_j,cog_voxel_k
1,Acumbens_left,-15.36,66.02,-20.08,79.28,34.93,142.81
2,AID_left,-45.77,64.44,-7.54,103.61,44.97,141.56
3,AIP_left,-61.89,29.91,-21.02,116.52,34.19,113.93
...
```

---

## 5. Step 3: Mapping ROI Subsets (Optional)

Often your connectivity matrix has fewer ROIs than your full atlas. For example:

- Full atlas: 170 ROIs
- Connectivity matrix: 28 x 28

You need to extract the coordinates for only those 28 ROIs that match your connectivity data.

### Using a Node File (.node)

If you have a BrainNet Viewer `.node` file with your subset of ROIs:

```bash
hlplot coords map-subset \
  --coords atlas_170_coordinates_comma.csv \
  --subset my_28_rois.node \
  --output-dir ./mapped_coordinates \
  --name atlas_28_mapped
```

### Using a Text File

Create a text file with ROI names (one per line):

```
AUD_left
PtPD_left
RSD_left
RSGc_left
...
```

Then map:

```bash
hlplot coords map-subset \
  --coords atlas_170_coordinates_comma.csv \
  --subset roi_subset.txt \
  --output-dir ./mapped_coordinates \
  --name atlas_28_mapped
```

### Output

The command creates:

- `atlas_28_mapped_comma.csv` - Use this for plotting
- `atlas_28_mapped_tab.csv`
- `atlas_28_mapped.pkl`

It also reports any ROIs that couldn't be matched.

---

## 6. Step 4: Creating Connectivity Plots

Now you have all the required files:
- Brain mesh (`.gii`)
- ROI coordinates (`.csv`)
- Connectivity matrix (`.npy`, `.csv`, or `.edge`)

### Basic Plot

```bash
hlplot plot \
  --mesh brain_mesh.gii \
  --coords atlas_28_mapped_comma.csv \
  --matrix connectivity.npy \
  --output brain_network.html
```

### With Custom Title and Styling

```bash
hlplot plot \
  --mesh brain_mesh.gii \
  --coords atlas_28_mapped_comma.csv \
  --matrix connectivity.npy \
  --output brain_network.html \
  --title "My Brain Network Analysis" \
  --node-size 12 \
  --node-color steelblue \
  --mesh-opacity 0.2 \
  --camera superior
```

### Using Edge Files Directly

If you have BrainNet Viewer format files:

```bash
hlplot plot \
  --mesh brain_mesh.gii \
  --coords atlas_28_mapped_comma.csv \
  --matrix my_data.edge \
  --output brain_network.html
```

### Interactive Features

The output HTML file includes:
- **Rotate/Zoom**: Click and drag to rotate, scroll to zoom
- **Camera presets**: Dropdown to switch between standard views
- **Legend**: Click to toggle positive/negative edges
- **Hover**: Mouse over nodes to see ROI names and metrics

---

## 7. Modularity Visualization

For networks with module assignments (community structure):

### Basic Modularity Plot

```bash
hlplot modular \
  --mesh brain_mesh.gii \
  --coords roi_coordinates.csv \
  --matrix connectivity.npy \
  --modules module_assignments.csv \
  --output modularity.html
```

### With Q and Z Scores

```bash
hlplot modular \
  --mesh brain_mesh.gii \
  --coords roi_coordinates.csv \
  --matrix connectivity.npy \
  --modules module_assignments.csv \
  --q-score 0.452 \
  --z-score 3.21 \
  --title "Network Modularity" \
  --output modularity.html
```

The title will show: "Network Modularity (Q=0.452, Z=3.21)"

### Module File Format

CSV with ROI index and module assignment:

```csv
roi_index,module
0,1
1,2
2,1
3,3
...
```

Or single column (one per line, 1-indexed):
```
1
2
1
3
...
```

### Edge Coloring Modes

```bash
# Color edges by positive/negative sign (default)
hlplot modular ... --edge-color-mode sign

# Color edges by source node's module
hlplot modular ... --edge-color-mode module
```

---

## 8. Node and Edge Customization

### Node Size

```bash
# Fixed size for all nodes
--node-size 15

# Size from a CSV file (per-node sizes)
--node-size sizes.csv
```

### Node Color

```bash
# Single color for all nodes
--node-color purple
--node-color "#FF5733"

# Color by module assignments (auto-generates colors)
--node-color modules.csv
```

### Edge Width

```bash
# Scaled by connection weight (min to max)
--edge-width-min 0.5 --edge-width-max 8.0

# Fixed width (no scaling)
--edge-width-fixed 2.0
```

### Edge Threshold

Only show edges above a minimum weight:

```bash
--edge-threshold 0.3
```

### Edge Colors

```bash
--pos-edge-color red
--neg-edge-color blue
```

---

## 9. Static Image Export

Export publication-quality images alongside interactive HTML:

### PNG Export (Raster)

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
  --export-image figure.png \
  --export-no-title \
  --export-no-legend
```

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

### Usage

```bash
hlplot plot ... --camera superior
hlplot plot ... --camera lateral-left
```

### Generate Multiple Views

```bash
for view in anterior posterior left right superior inferior; do
  hlplot plot \
    --mesh brain.gii \
    --coords rois.csv \
    --matrix conn.npy \
    --camera $view \
    --export-image "figure_${view}.png" \
    --output "brain_${view}.html"
done
```

---

## 11. Batch Processing

Process multiple subjects from a configuration file:

### Configuration File (YAML)

```yaml
# batch_config.yaml
mesh_file: "data/brain.gii"
roi_coords_file: "data/rois.csv"
output_dir: "./outputs"
output_format: "html"

plot:
  mesh_opacity: 0.2
  node_size: 10
  camera_view: oblique

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

### Dry Run (Preview)

```bash
hlplot batch --config batch_config.yaml --dry-run
```

---

## 12. Complete Parameter Reference

### `hlplot coords generate`

| Parameter | Required | Description |
|-----------|----------|-------------|
| `--volume`, `-v` | Yes | NIfTI volume file path |
| `--labels`, `-l` | Yes | Label text file (tab-delimited) |
| `--output-dir`, `-o` | Yes | Output directory |
| `--name`, `-n` | No | Output file name (default: roi_coordinates) |

### `hlplot coords map-subset`

| Parameter | Required | Description |
|-----------|----------|-------------|
| `--coords`, `-c` | Yes | Full coordinates CSV file |
| `--subset`, `-s` | Yes | Subset file (.node, .txt, or .csv) |
| `--output-dir`, `-o` | Yes | Output directory |
| `--name`, `-n` | No | Output file name (default: mapped_roi_coordinates) |

### `hlplot plot`

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--mesh`, `-m` | PATH | required | Brain mesh file |
| `--coords`, `-c` | PATH | required | ROI coordinates CSV |
| `--matrix`, `-x` | PATH | required | Connectivity matrix |
| `--output`, `-o` | PATH | brain_connectivity.html | Output path |
| `--title`, `-t` | TEXT | "Brain Connectivity" | Plot title |
| `--node-size` | TEXT | "8" | Node size (number or file) |
| `--node-color` | TEXT | "purple" | Node color |
| `--edge-width-min` | FLOAT | 1.0 | Min edge width |
| `--edge-width-max` | FLOAT | 5.0 | Max edge width |
| `--edge-threshold` | FLOAT | 0.0 | Min edge weight |
| `--mesh-opacity` | FLOAT | 0.15 | Mesh transparency |
| `--camera` | CHOICE | "oblique" | Camera view |
| `--export-image` | PATH | None | Static image export |
| `--image-dpi` | INT | 300 | Export resolution |

### `hlplot modular`

Inherits all `plot` parameters plus:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--modules`, `-d` | PATH | required | Module assignments file |
| `--q-score` | FLOAT | None | Modularity Q score |
| `--z-score` | FLOAT | None | Z-rand score |
| `--edge-color-mode` | CHOICE | "module" | Edge coloring mode |

---

## 13. Troubleshooting

### Dimension Mismatch Error

```
Matrix size (28) differs from ROI count (170)
```

**Solution:** Use `hlplot coords map-subset` to extract matching ROIs:

```bash
hlplot coords map-subset \
  --coords full_atlas.csv \
  --subset your_28_rois.node \
  --output-dir ./mapped
```

### Module Not Found Error

```
ModuleNotFoundError: No module named 'HarrisLabPlotting'
```

**Solution:** Make sure you installed the package:

```bash
# Activate environment first
conda activate harris_lab_plotting

# Then install
pip install -e .
```

### Missing ROI Labels

```
Warning: ROI X (name) not found in volume
```

**Solution:** Check that your label file matches the labels in your NIfTI volume:

```bash
# View unique labels in your NIfTI file
python -c "import nibabel as nib; import numpy as np; img = nib.load('your_file.nii.gz'); print(np.unique(img.get_fdata()))"
```

### Mesh File Not Loading

**Solution:** Try converting to a different format using nii2mesh or Surfice.

---

## Getting Help

```bash
# Main help
hlplot --help

# Command-specific help
hlplot plot --help
hlplot modular --help
hlplot coords generate --help
hlplot coords map-subset --help
```

For issues: https://github.com/AzadAzargushasb/HarrisLabPlotting/issues

---

*HarrisLabPlotting v1.0.0 - Brain Connectivity Visualization Tools*
