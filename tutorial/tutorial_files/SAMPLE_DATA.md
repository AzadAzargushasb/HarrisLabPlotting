# Sample Data Files for Tutorial

This document lists sample data files that can be used to follow the tutorial.

---

## Location of Sample Files

The main sample files are located in the parent `test_files/` directory:

```
HarrisLabPlotting/
├── test_files/
│   ├── brain_filled_2.gii          # Brain mesh file
│   ├── node and edges/
│   │   ├── total.node              # 28-ROI node file
│   │   └── total(1).edge           # 28x28 edge matrix
│   └── k5_data/
│       └── state_0/
│           ├── connectivity_matrix.csv   # 114x114 connectivity
│           ├── module_assignments.csv    # Module assignments
│           └── combined_metrics.csv      # Node metrics
└── tutorial/
    └── tutorial_files/
        └── SAMPLE_DATA.md          # This file
```

---

## File Descriptions

### Brain Mesh

**File:** `test_files/brain_filled_2.gii`

A GIFTI format brain mesh suitable for visualization.

### 28-ROI Example (Subset)

These files demonstrate working with a subset of ROIs:

| File | Description |
|------|-------------|
| `test_files/node and edges/total.node` | BrainNet Viewer node file with 28 ROIs |
| `test_files/node and edges/total(1).edge` | 28x28 connectivity edge file |

### 114-ROI Example

These files demonstrate a larger network:

| File | Description |
|------|-------------|
| `test_files/k5_data/state_0/connectivity_matrix.csv` | 114x114 connectivity matrix |
| `test_files/k5_data/state_0/module_assignments.csv` | Module assignments for 114 ROIs |
| `test_files/k5_data/state_0/combined_metrics.csv` | Node metrics (PC, within-module Z-score) |

---

## Example Usage

### Using 28-ROI Data

```bash
# First, generate mapped coordinates from a full atlas
# (assuming you have atlas_170_coordinates_comma.csv)
hlplot coords map-subset \
  --coords ../test_files/roi_coordinates/atlas_170_coordinates_comma.csv \
  --subset ../test_files/node\ and\ edges/total.node \
  --output-dir ./mapped \
  --name atlas_28_mapped

# Create plot
hlplot plot \
  --mesh ../test_files/brain_filled_2.gii \
  --coords ./mapped/atlas_28_mapped_comma.csv \
  --matrix "../test_files/node and edges/total(1).edge" \
  --output brain_28.html
```

### Using 114-ROI Data

```bash
hlplot plot \
  --mesh ../test_files/brain_filled_2.gii \
  --coords path/to/atlas_114_coordinates.csv \
  --matrix ../test_files/k5_data/state_0/connectivity_matrix.csv \
  --output brain_114.html

# With modularity
hlplot modular \
  --mesh ../test_files/brain_filled_2.gii \
  --coords path/to/atlas_114_coordinates.csv \
  --matrix ../test_files/k5_data/state_0/connectivity_matrix.csv \
  --modules ../test_files/k5_data/state_0/module_assignments.csv \
  --output modularity_114.html
```

---

## External Files Mentioned in Tutorial

The following files are referenced in the Jupyter notebook examples but are located outside the package:

| File | Description | Path |
|------|-------------|------|
| 170-ROI NIfTI | Brain atlas volume | `C:\Users\...\Downloads\brain_filled.nii.gz` |
| 170-ROI coordinates | Full atlas coordinates | `C:\Users\...\Research\roi_coordinates\atlas_170_coordinates_comma.csv` |
| 114-ROI coordinates | Mapped coordinates | `G:\My Drive\research stim data cci\atlas_114_mapped_comma.csv` |
| Smoothed mesh | High-quality mesh | `C:\Users\...\Research\brain_filled_3_smoothed.gii` |

To replicate the full tutorial, you'll need access to these files or create your own using the pipeline described in the main README.

---

*See the main [README.md](../README.md) for complete tutorial instructions.*
