# Creating Brain Mesh Files from NIfTI Volumes

This guide explains how to create brain mesh files (.gii, .obj, .ply, .mz3) from NIfTI volume files (.nii, .nii.gz).

---

## Overview

HarrisLabPlotting requires a 3D brain mesh file for visualization. If you have a NIfTI volume file (e.g., a brain atlas or parcellation), you'll need to convert it to a mesh format first.

---

## Methods for Converting NIfTI to Mesh

### Method 1: Surfice (Recommended)

**Surfice** is a desktop application that provides high-quality mesh generation with smoothing options.

**Download:** https://www.nitrc.org/projects/surfice/

**Instructions:**
<!-- TODO: Add detailed Surfice instructions -->

1. Open Surfice
2. Load your NIfTI file
3. Export as mesh format (.gii recommended)

---

### Method 2: nii2mesh Web Tool (Easy & Simple)

**Website:** https://rordenlab.github.io/nii2meshWeb/

This web-based tool is the easiest option for quick conversions. No installation required!

**Instructions:**

1. Go to https://rordenlab.github.io/nii2meshWeb/
2. Click "Choose File" and select your NIfTI file
3. Adjust settings if needed (mesh quality, smoothing)
4. Click "Convert"
5. Download the generated mesh file

---

### Method 3: nii2mesh Command-Line Tool

**Repository:** https://github.com/neurolabusc/nii2mesh

For batch processing or automation, the command-line version of nii2mesh is recommended.

**Installation:**
```bash
# Clone the repository
git clone https://github.com/neurolabusc/nii2mesh.git
cd nii2mesh

# Build (requires CMake)
mkdir build && cd build
cmake ..
make
```

**Usage:**
```bash
# Basic conversion
nii2mesh input.nii.gz output.gii

# With smoothing
nii2mesh -s 3 input.nii.gz output_smoothed.gii
```

---

## Recommended Settings

- **Output format:** `.gii` (GIFTI) is recommended for best compatibility
- **Smoothing:** Apply 2-5 iterations for cleaner visualization
- **Decimation:** Can reduce file size if needed, but may lose detail

---

## Next Steps

After creating your mesh file, you can use it with HarrisLabPlotting:

```bash
hlplot plot --mesh your_brain.gii --coords rois.csv --matrix connectivity.npy
```

---

## Troubleshooting

### Mesh appears hollow or inverted
- Try different threshold values during conversion
- Check that your NIfTI file has proper orientation

### File is too large
- Apply decimation to reduce polygon count
- Use a lower resolution during conversion

### Coordinates don't match mesh
- Ensure both mesh and coordinates use the same coordinate space
- Check affine transformations in your NIfTI header

---

*This guide will be expanded with more detailed instructions.*
