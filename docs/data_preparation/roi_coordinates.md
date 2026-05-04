# ROI coordinates

Every brain plot needs an XYZ position for each ROI — the network nodes are
drawn at these coordinates. HarrisLabPlotting can extract these
automatically from a labeled NIfTI atlas using
[`scipy.ndimage.center_of_mass`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.center_of_mass.html)
combined with the NIfTI affine.

The full CLI walkthrough is at
[CLI walkthrough §2–3](../tutorials/cli_walkthrough.md#2-generating-roi-coordinates-from-nifti).

## Generate from a labeled NIfTI

You need:

- **Volume file** — a NIfTI (`.nii` / `.nii.gz`) where each voxel value is
  an ROI index (0 = background)
- **Label file** — a tab-delimited `.txt` with `index<TAB>name` per line

```bash
hlplot coords generate \
  --volume brain_atlas_170.nii \
  --labels atlas_170_labels.txt \
  --output-dir ./coords_out
```

This writes three files: a CSV (the canonical input for `hlplot plot`), a
MATLAB `.mat`, and a NumPy `.npy`.

## Map a subset of ROIs

If your connectivity analysis used a subset of the full atlas — say 114 out
of 170 — re-derive coordinates for just that subset:

```bash
hlplot coords map-subset \
  --coords atlas_170_coordinates.csv \
  --subset atlas_114_labels.txt \
  --output-dir ./coords_114
```

The output CSV preserves the order of `atlas_114_labels.txt`, which must
match the row/column order of your connectivity matrix.

## CSV schema

The coordinate CSV consumed by `hlplot plot` has these columns:

| Column | Type | Required | Notes |
| --- | --- | --- | --- |
| `roi_index` | int | yes | 1-based ROI index from the atlas |
| `roi_name` | str | yes | Human-readable name |
| `cog_x` | float | yes | Center-of-gravity X (in mesh space) |
| `cog_y` | float | yes | Center-of-gravity Y |
| `cog_z` | float | yes | Center-of-gravity Z |

Coordinates are in mesh space — i.e. the same coordinate system as your
brain mesh's vertices. If your mesh and atlas use different conventions
(e.g. RAS vs LAS), apply the transform before exporting.

## Python API

```python
from HarrisLabPlotting import coordinate_function

df = coordinate_function(
    volume_file="brain_atlas_170.nii",
    roi_label_file="atlas_170_labels.txt",
    name_of_file="atlas_170",
    save_directory="./coords_out",
)
```

`df` is a pandas DataFrame; the function also writes the CSV / MAT / NPY
artifacts to `save_directory`.
