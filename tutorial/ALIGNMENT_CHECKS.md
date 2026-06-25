# HarrisLabPlotting — Checking Atlas / Mesh Alignment

> Standalone companion to the rendered docs page
> [`docs/how_to/check_atlas_mesh_alignment.md`](../docs/how_to/check_atlas_mesh_alignment.md).
> Each check has a **CLI** snippet and an equivalent **Python / notebook** snippet.

Before plotting a new atlas, confirm the pieces fit. Three quiet failures cause
almost every "my brain looks wrong" bug:

- **Float-labeled atlas** → an exact `volume == label` match finds zero voxels →
  every ROI coordinate is `NaN`.
- **Wrong template space** → atlas and mesh are in different variants → nodes
  float off the brain.
- **Bilateral / merged parcellation** → each label spans both hemispheres →
  every centre-of-gravity collapses onto the midline.

Runnable examples use the bundled tutorial files; run from
`test_files/tutorial_files`.

---

## One command: `hlplot utils check-alignment`

```bash
hlplot utils check-alignment \
  --coords atlas_114_coordinates.csv \
  --mesh brain_mesh.gii
```

```
                   COGs vs Mesh
  ROIs                     | 114
  NaN COGs                 | 0
  Inside mesh hull         | 114 / 114
  COG bbox within mesh     | True
  On midline (|x|<2mm)     | 0 (0%)
  Nearest-vertex dist (mm) | max 38.7, mean 20.7
  Verdict                  | PASS
Overall: PASS
```

Add `--volume atlas.nii.gz` to also check the atlas shares the mesh's space, and
`--matrix conn.csv` to check the matrix size matches the ROI count.

```python
from HarrisLabPlotting import check_coords_in_mesh

report = check_coords_in_mesh("atlas_114_coordinates.csv", "brain_mesh.gii")
print(report["verdict"], report["n_inside"], "/", report["n_rois"])
for msg in report["messages"]:
    print("-", msg)
```

---

## Check 1 — Is the atlas cleanly integer-labeled?

Some atlases store integer labels as floats (`0.9999999997` for label 1). The
exact match then finds **zero voxels** → all-NaN COGs. `coords generate` rounds by
default, but check/fix explicitly with:

```bash
hlplot utils info --volume my_atlas.nii.gz                                  # check
hlplot utils clean-labels --volume my_atlas.nii.gz --output my_atlas_int.nii.gz  # fix
```

```python
from HarrisLabPlotting import inspect_label_volume, clean_label_volume

info = inspect_label_volume("my_atlas.nii.gz")
print("bit-exact integers:", info["is_integer_labeled"],
      "| labels:", info["n_labels"])
if not info["is_integer_labeled"]:
    clean_label_volume("my_atlas.nii.gz", output_path="my_atlas_int.nii.gz")
```

`is_integer_labeled` is `True` only for **bit-exact** integers — a `0.9999999997`
value reports `False`, because that tiny gap is enough to break the exact match.

---

## Check 2 — Do the ROI COGs land inside the mesh?

`check_coords_in_mesh` tests every COG against the mesh's **convex hull** (fast
even for a 200k-vertex mesh).

```bash
hlplot utils check-alignment --coords my_coords.csv --mesh my_brain.obj
```

```python
from HarrisLabPlotting import check_coords_in_mesh

r = check_coords_in_mesh("my_coords.csv", "my_brain.obj")
print("inside:", r["n_inside"], "/", r["n_rois"], "->", r["verdict"])
```

A few interior COGs just outside the hull near deep sulci → **WARN**. A large
fraction outside, or a COG bbox not inside the mesh → **FAIL** (space mismatch).

---

## Check 3 — Are the atlas and mesh in the same space?

```bash
hlplot utils check-alignment \
  --coords my_coords.csv --mesh my_brain.obj --volume my_atlas.nii.gz
```

```python
from HarrisLabPlotting import compare_volume_mesh_space

s = compare_volume_mesh_space("my_atlas.nii.gz", "my_brain.obj")
print(f"overlap {s['bbox_overlap_fraction']:.0%}, "
      f"offset {s['centroid_offset_mm']:.1f} mm -> {s['verdict']}")
```

Low overlap + large centroid offset = different spaces. Use the matching-space
release of the atlas, or resample/warp it into the mesh's space.

---

## Check 4 — Do the matrix / module counts line up?

```bash
hlplot utils validate \
  --mesh my_brain.obj --coords my_coords.csv \
  --matrix my_conn.csv --modules my_modules.csv
```

---

## Common pitfalls

| Symptom | Likely cause | Fix |
|---|---|---|
| Every COG is `NaN` / "ROI not found" | Float-labeled atlas | `clean-labels`, or `coords generate --round-labels` (default) |
| All nodes pile onto the midline | Bilateral/merged parcellation | Use a lateralized atlas (separate L/R labels) |
| Nodes float off the brain | Atlas and mesh in different spaces | Match spaces / use native-space atlas |
| A few nodes just outside the surface | Deep/sulcal regions vs convex hull | Usually fine (WARN) |
| "ROI count mismatch" | Matrix/modules length ≠ coords rows | Re-derive coords in matrix order |
