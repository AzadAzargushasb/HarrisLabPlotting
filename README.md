# HarrisLabPlotting

A Python toolkit for creating interactive 3D brain connectivity visualizations and managing ROI (Region of Interest) coordinates from neuroimaging data.

## Overview

This repository contains two main modules:

1. **`brain_connectivity_vizuals.py`** - Create interactive 3D brain connectivity plots with Plotly
2. **`roi_coordinate_tools.py`** - Extract, map, and clean ROI coordinates from brain atlases

## Features

### Brain Connectivity Visualization (`brain_connectivity_vizuals.py`)

- Interactive 3D brain mesh rendering with transparency control
- Overlay ROI nodes with customizable colors and sizes
- Display connectivity edges (positive/negative) with threshold filtering
- Toggle between "All Nodes" and "Connected Only" views
- Filter edges by sign (positive, negative, or all)
- Modularity visualization with color-coded modules
- Export to interactive HTML files
- Graph statistics calculation (density, degree, hub nodes)

### ROI Coordinate Tools (`roi_coordinate_tools.py`)

- **`coordinate_function()`** - Extract center-of-gravity (COG) coordinates from NIfTI atlas volumes
- **`map_coordinate()`** - Map a subset of ROIs to their coordinates, with clear reporting of unmapped ROIs
- **`load_and_clean_coordinates()`** - Load CSV coordinates and remove rows with missing values
- **`load_matrix_replace_nan()`** - Load connectivity matrices (.mat or .txt) and replace NaN values with zeros

## Installation

### Option 1: Conda (Recommended)

```bash
# Clone the repository
git clone https://github.com/AzadAzargushasb/HarrisLabPlotting.git
cd HarrisLabPlotting

# Create conda environment
conda env create -f environment.yml

# Activate environment
conda activate harris_lab_plotting
```

### Option 2: pip

```bash
# Clone the repository
git clone https://github.com/AzadAzargushasb/HarrisLabPlotting.git
cd HarrisLabPlotting

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

### 1. Extract ROI Coordinates from Atlas

```python
import roi_coordinate_tools as rct

# Extract coordinates from a NIfTI atlas file
roi_coords = rct.coordinate_function(
    volume_file_location="atlas.nii.gz",
    roi_label_file="atlas_labels.txt",
    name_of_file="my_roi_coordinates"
)
```

### 2. Map Coordinates to Study-Specific ROIs

```python
# Map to a subset of ROIs (returns DataFrame and list of unmapped ROIs)
mapped_coords, unmapped_rois = rct.map_coordinate(
    original_coords_file=roi_coords,  # DataFrame or file path
    reduced_roi_file="study_roi_list.txt",
    name_of_file="study_specific_coordinates"
)

# Check which ROIs couldn't be mapped
if unmapped_rois:
    print(f"Could not map: {[r['name'] for r in unmapped_rois]}")
```

### 3. Clean Coordinates (Remove Missing Values)

```python
# Load and clean coordinates CSV
cleaned_coords = rct.load_and_clean_coordinates(
    csv_file_path="coordinates.csv",
    output_file_name="coordinates_cleaned",
    save_directory="output"
)
```

### 4. Load Connectivity Matrix (Handle NaNs)

```python
# Load matrix and replace NaNs with zeros
conn_matrix = rct.load_matrix_replace_nan(
    file_path="connectivity.mat",  # or .txt
    replacement_value=0
)
```

### 5. Create Brain Connectivity Visualization

```python
import brain_connectivity_vizuals as bcv
import pandas as pd

# Load mesh file (GIFTI format)
vertices, faces = bcv.load_mesh_file("brain_mesh.gii")

# Load cleaned ROI coordinates
roi_coords_df = pd.read_csv("coordinates_cleaned_comma.csv")

# Create interactive visualization
fig, stats = bcv.create_brain_connectivity_plot(
    vertices=vertices,
    faces=faces,
    roi_coords_df=roi_coords_df,
    connectivity_matrix=conn_matrix,
    plot_title="My Brain Network",
    save_path="brain_network.html",
    node_size=8,
    node_color='purple',
    pos_edge_color='red',
    neg_edge_color='blue',
    edge_threshold=0.1,
    mesh_opacity=0.3
)

# Print network statistics
print(f"Total nodes: {stats['total_nodes']}")
print(f"Total edges: {stats['total_edges']}")
print(f"Network density: {stats['network_density']:.4f}")
```

### 6. Create Modularity Visualization

```python
# Visualize network modules with color-coded nodes
fig, module_stats = bcv.create_modularity_visualization(
    vertices=vertices,
    faces=faces,
    roi_coords_df=roi_coords_df,
    connectivity_matrix=conn_matrix,
    module_assignments=module_labels,  # 1D array of module IDs
    plot_title="Network Modules",
    save_path="modularity.html",
    visualization_type="all"  # "all", "intra", "inter", or "nodes_only"
)
```

## File Formats

### Input Files

| File Type | Description | Format |
|-----------|-------------|--------|
| Atlas Volume | NIfTI file with integer ROI labels | `.nii`, `.nii.gz` |
| ROI Labels | Tab-delimited: `index\tname` | `.txt` |
| Brain Mesh | GIFTI surface mesh | `.gii` |
| Connectivity Matrix | Square matrix | `.mat`, `.txt` |
| Coordinates CSV | ROI coordinates | `.csv` (comma or tab) |

### Output Files

All coordinate functions output files in three formats:
- `.pkl` - Pandas pickle (preserves data types)
- `_comma.csv` - Comma-delimited CSV
- `_tab.csv` - Tab-delimited CSV

## Example Notebook

See `brain connectivity example.ipynb` for a complete workflow demonstrating:
- Extracting ROI coordinates from atlas
- Mapping to study-specific ROIs
- Cleaning coordinates
- Loading connectivity matrices
- Creating interactive visualizations

## API Reference

### `roi_coordinate_tools.py`

#### `coordinate_function(volume_file_location, roi_label_file, name_of_file=None, save_directory=".")`
Extract ROI center-of-gravity coordinates from a NIfTI volume.

#### `map_coordinate(original_coords_file, reduced_roi_file, save_directory=".", name_of_file=None)`
Map a subset of ROIs to coordinates. Returns `(DataFrame, unmapped_rois_list)`.

#### `load_and_clean_coordinates(csv_file_path, output_file_name=None, save_directory=".")`
Load coordinates CSV and remove rows with missing values.

#### `load_matrix_replace_nan(file_path, replacement_value=0)`
Load connectivity matrix and replace NaN values.

### `brain_connectivity_vizuals.py`

#### `load_mesh_file(mesh_path)`
Load GIFTI brain mesh. Returns `(vertices, faces)`.

#### `create_brain_connectivity_plot(...)`
Create interactive 3D brain connectivity visualization. Returns `(figure, graph_stats)`.

#### `create_modularity_visualization(...)`
Create modularity-colored brain visualization. Returns `(figure, module_stats)`.

#### `quick_brain_plot(vertices, faces, roi_coords_df, connectivity_matrix, title, save_name)`
Quick plotting with default parameters.

## Requirements

- Python >= 3.9
- numpy >= 1.21
- pandas >= 1.3
- scipy >= 1.7
- nibabel >= 4.0
- networkx >= 2.6
- plotly >= 5.0
- jupyter (for notebooks)
- kaleido (for static image export)

## License

MIT License

## Authors

Harris Lab - Brain Connectivity Analysis Tools

## Citation

If you use this software in your research, please cite:

```
Harris Lab Plotting Tools
https://github.com/AzadAzargushasb/HarrisLabPlotting
```
