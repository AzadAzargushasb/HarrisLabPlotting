# HarrisLabPlotting CLI Tutorial

A comprehensive guide to using the `hlplot` command-line tool for brain connectivity and modularity visualization.

---

## Table of Contents

1. [Installation](#1-installation)
2. [Quick Start](#2-quick-start)
3. [Basic Plotting](#3-basic-plotting)
4. [Modularity Visualization](#4-modularity-visualization)
5. [Configuration Files](#5-configuration-files)
6. [Batch Processing](#6-batch-processing)
7. [Utility Commands](#7-utility-commands)
8. [Camera Views](#8-camera-views)
9. [Output Formats](#9-output-formats)
10. [Troubleshooting](#10-troubleshooting)
11. [API Reference](#11-api-reference)

---

## 1. Installation

### Option A: Using Conda (Recommended)

```bash
# Create and activate the environment
conda env create -f environment.yml
conda activate harris_lab_plotting

# Install the package in development mode
pip install -e .
```

### Option B: Using pip

```bash
# Install directly from the repository
pip install .

# Or in development mode
pip install -e .
```

### Verify Installation

```bash
# Check that hlplot is available
hlplot --version

# View all available commands
hlplot --help
```

---

## 2. Quick Start

Create your first brain connectivity plot in three steps:

```bash
# Step 1: Navigate to your data directory
cd /path/to/your/data

# Step 2: Create a basic plot
hlplot plot \
  --mesh brain.gii \
  --coords rois.csv \
  --matrix connectivity.npy \
  --output my_first_plot.html

# Step 3: Open the HTML file in your browser
# The plot is interactive - you can rotate, zoom, and hover for details!
```

---

## 3. Basic Plotting

The `hlplot plot` command creates brain connectivity visualizations.

### Required Inputs

| File | Format | Description |
|------|--------|-------------|
| `--mesh` | .gii | Brain surface mesh file |
| `--coords` | .csv | ROI coordinates (x, y, z columns) |
| `--matrix` | .npy, .csv, .txt | Connectivity matrix |

### Common Options

```bash
hlplot plot \
  --mesh brain.gii \
  --coords rois.csv \
  --matrix connectivity.npy \
  --output my_plot.html \
  --title "My Study Results" \
  --node-size 15 \
  --node-color blue \
  --edge-threshold 0.2 \
  --opacity 0.4 \
  --camera lateral-left
```

### Node Customization

**Fixed size for all nodes:**
```bash
--node-size 10
```

**Size from vector file:**
```bash
--node-size /path/to/node_sizes.csv
```

**Fixed color:**
```bash
--node-color purple
--node-color "#FF5733"
--node-color "rgb(255,87,51)"
```

**Color from module assignments:**
```bash
--node-color /path/to/modules.csv
```

### Edge Customization

```bash
# Threshold edges by absolute weight
--edge-threshold 0.1

# Show only top N edges
--top-n 100

# Control edge width range
--edge-width-min 1 --edge-width-max 8
```

### Full Example

```bash
hlplot plot \
  --mesh data/fsaverage_lh.gii \
  --coords data/schaefer400_coords.csv \
  --matrix data/group_connectivity.npy \
  --output figures/connectivity.html \
  --title "Group Average Connectivity" \
  --node-size 8 \
  --node-color steelblue \
  --edge-threshold 0.15 \
  --edge-width-min 1 \
  --edge-width-max 5 \
  --opacity 0.3 \
  --camera anterior \
  --show
```

---

## 4. Modularity Visualization

The `hlplot modular` command creates visualizations with module-based coloring.

### Basic Usage

```bash
hlplot modular \
  --mesh brain.gii \
  --coords rois.csv \
  --matrix connectivity.npy \
  --modules module_assignments.csv \
  --output modularity_plot.html
```

### Module Assignments File

The module file should contain integer assignments (1-indexed):

```
# modules.csv - one value per line
1
1
2
2
3
1
...
```

### Adding Scores to Title

```bash
hlplot modular \
  --mesh brain.gii \
  --coords rois.csv \
  --matrix connectivity.npy \
  --modules modules.csv \
  --q-score 0.45 \
  --z-score 3.2 \
  --title "Subject 001 Modularity"
```

This produces a title like: "Subject 001 Modularity (Q=0.450, Z=3.200)"

### Edge Coloring Modes

**By module (default):** Edges inherit colors from connected nodes
```bash
--edge-color-mode module
```

**By sign:** Red for positive, blue for negative correlations
```bash
--edge-color-mode sign
```

### Full Example

```bash
hlplot modular \
  --mesh data/brain.gii \
  --coords data/rois.csv \
  --matrix data/connectivity.npy \
  --modules data/community_assignments.csv \
  --output figures/modularity.html \
  --title "Network Modularity Analysis" \
  --q-score 0.52 \
  --z-score 4.1 \
  --edge-color-mode module \
  --node-size 12 \
  --top-n 200 \
  --camera superior \
  --show
```

---

## 5. Configuration Files

For complex or repeated analyses, use YAML configuration files.

### Creating a Config File

```bash
# Generate an example configuration
hlplot config init --output my_config.yaml
```

### Example Configuration

```yaml
# my_config.yaml

# Input files
mesh_file: "data/brain.gii"
roi_coords_file: "data/rois.csv"
connectivity_matrix: "data/connectivity.npy"

# Output settings
output_dir: "./outputs"
output_format: "html"

# Plot settings
plot:
  title: "Brain Connectivity"
  node_size: 10
  node_color: "purple"
  edge_threshold: 0.1
  opacity: 0.3

# Camera settings
camera:
  view: "anterior"

# Modularity (optional)
modularity:
  enabled: true
  module_file: "data/modules.csv"
  edge_color_mode: "module"
```

### Using a Config File

```bash
# Use config file
hlplot plot --config my_config.yaml

# Override specific options
hlplot plot --config my_config.yaml --camera lateral-left --output custom.html
```

### Validating Config

```bash
# Check for errors
hlplot config validate my_config.yaml

# Show parsed configuration
hlplot config show my_config.yaml
```

---

## 6. Batch Processing

Process multiple subjects with a single command.

### Batch Configuration

```yaml
# batch_config.yaml

mesh_file: "data/brain.gii"
roi_coords_file: "data/rois.csv"
output_dir: "./outputs"
output_format: "html"

plot:
  opacity: 0.3
  node_size: 10

camera:
  view: "anterior"

modularity:
  edge_color_mode: "module"

batch:
  - name: "subject_01"
    matrix: "data/sub01_connectivity.npy"
    modules: "data/sub01_modules.csv"
    q_score: 0.45
    z_score: 3.2

  - name: "subject_02"
    matrix: "data/sub02_connectivity.npy"
    modules: "data/sub02_modules.csv"
    q_score: 0.48
    z_score: 3.5

  - name: "subject_03"
    matrix: "data/sub03_connectivity.npy"
    modules: "data/sub03_modules.csv"
    q_score: 0.42
    z_score: 2.9
```

### Running Batch Processing

```bash
# Process all subjects
hlplot batch --config batch_config.yaml

# Override output directory
hlplot batch --config batch_config.yaml --output-dir ./new_results/

# Dry run to check what will be processed
hlplot batch --config batch_config.yaml --dry-run
```

---

## 7. Utility Commands

### Matrix Information

```bash
# Display matrix statistics
hlplot utils info --matrix connectivity.npy
```

### Thresholding

```bash
# Keep top 100 edges
hlplot utils threshold --matrix conn.npy --output thresh.npy --top-n 100

# Keep top 10% of edges
hlplot utils threshold --matrix conn.npy --output thresh.npy --percentile 90

# Keep edges above absolute value
hlplot utils threshold --matrix conn.npy --output thresh.npy --absolute 0.5
```

### File Conversion

```bash
# Convert numpy to CSV
hlplot utils convert --input matrix.npy --output matrix.csv

# Convert CSV to numpy
hlplot utils convert --input matrix.csv --output matrix.npy
```

### Convert Node/Edge Files

```bash
# Convert BrainNet Viewer format to matrix
hlplot utils convert-node-edge \
  --node data.node \
  --edge data.edge \
  --output matrix.npy
```

### Validate Files

```bash
# Check file compatibility
hlplot utils validate \
  --mesh brain.gii \
  --coords rois.csv \
  --matrix connectivity.npy
```

### ROI Coordinates

```bash
# Load and inspect coordinates
hlplot coords load --file rois.csv --show-stats

# Validate coordinate file
hlplot coords load --file rois.csv --validate

# Extract coordinates from atlas
hlplot coords extract \
  --file atlas.nii.gz \
  --output extracted_coords.csv
```

---

## 8. Camera Views

Available preset camera views:

| View | Description |
|------|-------------|
| `anterior` | Front view (default) |
| `posterior` | Back view |
| `lateral-left` | Left side view |
| `lateral-right` | Right side view |
| `superior` | Top view (dorsal) |
| `inferior` | Bottom view (ventral) |
| `dorsal` | Same as superior |
| `ventral` | Same as inferior |

### Usage

```bash
# Use preset view
hlplot plot ... --camera lateral-left

# Multiple views (run separately)
for view in anterior posterior lateral-left lateral-right superior inferior; do
  hlplot plot ... --camera $view --output "plot_${view}.html"
done
```

---

## 9. Output Formats

### Interactive HTML (Default)

```bash
hlplot plot ... --format html --output my_plot.html
```

Features:
- Rotate, zoom, pan
- Hover tooltips
- Self-contained file
- Works in any browser

### Static Images

```bash
# PNG (for publications)
hlplot plot ... --format png --output my_plot.png --width 1600 --height 1200

# PDF (vector graphics)
hlplot plot ... --format pdf --output my_plot.pdf

# SVG (scalable vector)
hlplot plot ... --format svg --output my_plot.svg
```

### Image Size Options

```bash
--width 1200   # Image width in pixels
--height 900   # Image height in pixels
```

Recommended sizes:
- Screen: 1200 x 900
- Publication: 1600 x 1200
- High-resolution: 2400 x 1800

---

## 10. Troubleshooting

### Common Issues

**"File not found" errors:**
```bash
# Use absolute paths
hlplot plot --mesh /full/path/to/brain.gii ...

# Or check current directory
pwd
ls data/
```

**"Module not found" errors:**
```bash
# Make sure the package is installed
pip install -e .

# Verify installation
python -c "import HarrisLabPlotting; print('OK')"
```

**Empty or incorrect plots:**
```bash
# Validate your files first
hlplot utils validate --mesh brain.gii --coords rois.csv --matrix connectivity.npy

# Check matrix info
hlplot utils info --matrix connectivity.npy
```

**Static image export fails:**
```bash
# Install kaleido
pip install kaleido

# On some systems, you may need:
pip install kaleido==0.2.1
```

### Getting Help

```bash
# General help
hlplot --help

# Command-specific help
hlplot plot --help
hlplot modular --help
hlplot batch --help
hlplot utils --help
hlplot coords --help
hlplot config --help
```

---

## 11. API Reference

### Main Commands

| Command | Description |
|---------|-------------|
| `hlplot plot` | Create basic connectivity plot |
| `hlplot modular` | Create modularity visualization |
| `hlplot batch` | Process multiple subjects |
| `hlplot coords` | ROI coordinate utilities |
| `hlplot utils` | Data processing utilities |
| `hlplot config` | Configuration management |

### hlplot plot Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--mesh`, `-m` | PATH | required | Brain mesh file |
| `--coords`, `-c` | PATH | required | ROI coordinates |
| `--matrix`, `-x` | PATH | required | Connectivity matrix |
| `--output`, `-o` | PATH | auto | Output file path |
| `--title`, `-t` | TEXT | "Brain Connectivity" | Plot title |
| `--node-size` | FLOAT | 10.0 | Node size |
| `--node-color` | TEXT | "purple" | Node color |
| `--edge-threshold` | FLOAT | 0.0 | Edge threshold |
| `--edge-width-min` | FLOAT | 1.0 | Min edge width |
| `--edge-width-max` | FLOAT | 5.0 | Max edge width |
| `--opacity` | FLOAT | 0.3 | Mesh opacity |
| `--camera` | CHOICE | "anterior" | Camera view |
| `--top-n` | INT | None | Top N edges |
| `--format` | CHOICE | "html" | Output format |
| `--width` | INT | 1200 | Image width |
| `--height` | INT | 900 | Image height |
| `--show` | FLAG | False | Open in browser |

### hlplot modular Options

All options from `hlplot plot`, plus:

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--modules`, `-d` | PATH | required | Module assignments |
| `--q-score` | FLOAT | None | Q score for title |
| `--z-score` | FLOAT | None | Z score for title |
| `--edge-color-mode` | CHOICE | "module" | Edge coloring |

---

## Examples Gallery

### 1. Basic Network Plot

```bash
hlplot plot \
  --mesh brain.gii \
  --coords rois.csv \
  --matrix group_connectivity.npy \
  --title "Group Average Network" \
  --node-color steelblue \
  --top-n 150 \
  --camera anterior
```

### 2. Modularity with Custom Styling

```bash
hlplot modular \
  --mesh brain.gii \
  --coords rois.csv \
  --matrix subject_connectivity.npy \
  --modules community_labels.csv \
  --title "Community Structure" \
  --q-score 0.52 \
  --edge-color-mode module \
  --node-size 12 \
  --opacity 0.25 \
  --camera superior
```

### 3. Publication-Ready Figure

```bash
hlplot modular \
  --mesh brain.gii \
  --coords rois.csv \
  --matrix connectivity.npy \
  --modules modules.csv \
  --format png \
  --output figure_2a.png \
  --width 2400 \
  --height 1800 \
  --camera lateral-left
```

### 4. Multi-View Generation Script

```bash
#!/bin/bash
views=("anterior" "posterior" "lateral-left" "lateral-right" "superior" "inferior")

for view in "${views[@]}"; do
  hlplot modular \
    --mesh brain.gii \
    --coords rois.csv \
    --matrix connectivity.npy \
    --modules modules.csv \
    --format png \
    --output "figure_${view}.png" \
    --camera "$view"
done
```

---

## Support

For issues and feature requests, please visit:
https://github.com/harrislabcodebase/HarrisLabPlotting/issues

---

*HarrisLabPlotting v1.0.0 - Brain Connectivity Visualization Tools*
