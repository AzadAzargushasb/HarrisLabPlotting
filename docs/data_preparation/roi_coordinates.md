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

This writes three files: a comma-delimited CSV (the canonical input for
`hlplot plot`), a tab-delimited CSV, and a Python pickle (`.pkl`).

:::{admonition} ⚠️ Float-labeled atlases → all-NaN COGs
:class: warning

Some atlases store integer ROI labels as **floats** with tiny rounding error
(e.g. `0.9999999997` for label 1 instead of `1.0`). An exact `volume == label`
match then finds **zero voxels**, and every coordinate comes back `NaN`.

`coords generate` now rounds labels by default (`--round-labels`, on), so this
"just works". To **check** an atlas, or to pre-clean one for other tools:

```bash
# Is the atlas cleanly integer-labeled?
hlplot utils info --volume my_atlas.nii.gz

# Round float labels to a clean integer volume
hlplot utils clean-labels --volume my_atlas.nii.gz --output my_atlas_int.nii.gz
```

The bundled `HCPMMP1_on_MNI152_ICBM2009a_nlin_hd.nii` is a real example of a
float-labeled atlas. See
[Checking atlas/mesh alignment](../how_to/check_atlas_mesh_alignment.md) for the
full set of pre-flight checks.
:::

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
    volume_file_location="brain_atlas_170.nii",
    roi_label_file="atlas_170_labels.txt",
    name_of_file="atlas_170",
    save_directory="./coords_out",
    round_labels=True,   # round float labels before matching (default)
)
```

`df` is a pandas DataFrame; the function also writes the `_comma.csv`,
`_tab.csv`, and `.pkl` artifacts to `save_directory`.
