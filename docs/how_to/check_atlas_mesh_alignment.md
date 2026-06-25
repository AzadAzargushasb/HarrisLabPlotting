# Checking atlas / mesh alignment

Before you plot a new atlas, it pays to confirm the pieces actually fit
together. Three quiet failure modes account for almost every "my brain looks
wrong" bug:

- **Float-labeled atlas** → an exact `volume == label` match finds zero voxels,
  so every ROI coordinate comes out `NaN`.
- **Wrong template space** → the atlas and the mesh are in different macaque /
  MNI variants, so the nodes float off the brain.
- **Bilateral / merged parcellation** → each label spans both hemispheres, so
  every centre-of-gravity collapses onto the midline.

This page shows the one-shot check plus each individual check, with a **CLI**
snippet and an equivalent **Python / notebook** snippet for each. The runnable
examples use the bundled 28/114-ROI tutorial files, so they work out of the box
from `test_files/tutorial_files`.

---

## One command: `hlplot utils check-alignment`

The fastest pre-flight: point it at your coordinates CSV and mesh (optionally the
source atlas volume and the connectivity matrix) and read the PASS / WARN / FAIL
report.

```bash
hlplot utils check-alignment \
  --coords atlas_114_coordinates.csv \
  --mesh brain_mesh.gii
```

```
                   COGs vs Mesh
╭──────────────────────────┬─────────────────────╮
│ ROIs                     │ 114                 │
│ NaN COGs                 │ 0                   │
│ Inside mesh hull         │ 114 / 114           │
│ COG bbox within mesh     │ True                │
│ On midline (|x|<2mm)     │ 0 (0%)              │
│ Nearest-vertex dist (mm) │ max 38.7, mean 20.7 │
│ Verdict                  │ PASS                │
╰──────────────────────────┴─────────────────────╯
Overall: PASS
```

Add `--volume atlas.nii.gz` to also check the atlas shares the mesh's space, and
`--matrix conn.csv` to check the matrix size matches the ROI count:

```bash
hlplot utils check-alignment \
  --coords my_coords.csv --mesh my_brain.obj \
  --volume my_atlas.nii.gz --matrix my_conn.csv
```

The same logic from Python:

```python
from HarrisLabPlotting import check_coords_in_mesh

report = check_coords_in_mesh("atlas_114_coordinates.csv", "brain_mesh.gii")
print(report["verdict"], report["n_inside"], "/", report["n_rois"])
for msg in report["messages"]:
    print("-", msg)
```

---

## Check 1 — Is the atlas cleanly integer-labeled?

Some atlases store integer ROI labels as floats with tiny rounding error
(`0.9999999997` instead of `1.0`). The exact label match then matches **zero
voxels** and produces all-NaN COGs. `coords generate` rounds by default, but it's
worth knowing how to detect and fix it.

```bash
# Inspect: is it integer-labeled?
hlplot utils info --volume my_atlas.nii.gz

# Fix: write a clean integer-labeled copy
hlplot utils clean-labels --volume my_atlas.nii.gz --output my_atlas_int.nii.gz
```

```python
from HarrisLabPlotting import inspect_label_volume, clean_label_volume

info = inspect_label_volume("my_atlas.nii.gz")
print("bit-exact integers:", info["is_integer_labeled"],
      "| max deviation:", info["max_label_deviation"],
      "| labels:", info["n_labels"])

if not info["is_integer_labeled"]:
    clean_label_volume("my_atlas.nii.gz", output_path="my_atlas_int.nii.gz")
```

`is_integer_labeled` is `True` only when the values are **bit-exact** integers —
which is exactly what an exact `==` match needs. A `0.9999999997` value
(deviation ~1e-8) reports `False`, because that tiny gap is enough to break the
match.

---

## Check 2 — Do the ROI COGs land inside the mesh?

`check_coords_in_mesh` tests every centre-of-gravity against the mesh's **convex
hull** (fast even for a 200k-vertex mesh) and reports how many fall inside, the
nearest-vertex distances, and a verdict.

```bash
hlplot utils check-alignment --coords my_coords.csv --mesh my_brain.obj
```

```python
from HarrisLabPlotting import check_coords_in_mesh

r = check_coords_in_mesh("my_coords.csv", "my_brain.obj")
print("inside:", r["n_inside"], "/", r["n_rois"])
print("coords bbox:", r["coords_bbox"])
print("mesh   bbox:", r["mesh_bbox"])
print("verdict:", r["verdict"])
```

A handful of genuinely-interior COGs can sit just outside the hull near deep
sulci — that's reported as a **WARN**, not a failure. A large fraction outside,
or a COG bounding box that isn't inside the mesh, is a **FAIL** and almost always
means a space mismatch (next check).

---

## Check 3 — Are the atlas and mesh in the same space?

If your nodes float off the brain entirely, the atlas volume and the mesh are
probably in different template spaces (e.g. an `NMT`-space macaque atlas paired
with a native-space mesh). Compare their world bounding boxes:

```bash
hlplot utils check-alignment \
  --coords my_coords.csv --mesh my_brain.obj \
  --volume my_atlas.nii.gz
```

```python
from HarrisLabPlotting import compare_volume_mesh_space

s = compare_volume_mesh_space("my_atlas.nii.gz", "my_brain.obj")
print("bbox overlap:", f"{s['bbox_overlap_fraction']:.0%}")
print("centroid offset (mm):", round(s["centroid_offset_mm"], 1))
print("same space:", s["same_space"], "->", s["verdict"])
```

A low overlap fraction and a large centroid offset mean the two are not in the
same space. Pick the matching-space release of the atlas, or resample/warp it
into the mesh's space before extracting coordinates.

---

## Check 4 — Do the matrix / module counts line up?

The classic off-by-one: a connectivity matrix or module-assignment vector whose
length doesn't match the number of ROIs.

```bash
hlplot utils validate \
  --mesh my_brain.obj --coords my_coords.csv \
  --matrix my_conn.csv --modules my_modules.csv
```

`check-alignment --matrix my_conn.csv` does the same coords-vs-matrix size check
inline with the alignment report.

---

## Common pitfalls checklist

| Symptom | Likely cause | Fix |
| --- | --- | --- |
| Every COG is `NaN` / "ROI not found" warnings | Float-labeled atlas (exact match fails) | `clean-labels`, or `coords generate --round-labels` (default on) |
| All nodes pile onto the midline | Bilateral/merged parcellation (one label per bilateral area) | Use a lateralized atlas (separate L/R labels), or split labels by hemisphere |
| Nodes float off the brain | Atlas and mesh in different spaces (NMT vs native, MNI variants) | Match spaces; use the native-space atlas or resample |
| A few nodes just outside the surface | Deep/sulcal regions vs convex hull | Usually fine (reported as WARN) |
| "ROI count mismatch" | Matrix / modules length ≠ coords rows | Re-derive coords for the exact subset, in matrix order |
```
