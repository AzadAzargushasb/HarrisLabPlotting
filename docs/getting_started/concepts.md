# Concepts

A two-minute mental model of how HarrisLabPlotting thinks about a brain
plot. Knowing this makes every other page in the docs make sense.

## The three inputs

Every connectivity figure is a function of exactly three things:

```
   mesh         ──┐
   ROI coords   ──┼──> hlplot ──> Plotly Figure ──> .html / .png / .svg / .pdf
   matrix       ──┘
```

| Input | What it represents | Practical form |
| --- | --- | --- |
| **Mesh** | The brain surface that the network is drawn *on top of* | A triangle mesh with `vertices` (Nx3 float) and `faces` (Mx3 int) |
| **ROI coordinates** | The XYZ positions where network nodes are placed | One row per ROI, in the same coordinate space as the mesh |
| **Connectivity matrix** | The pairwise edge weights between ROIs | An `N × N` numeric matrix in the same row order as the coordinates |

Everything else — colors, sizes, thresholds, camera angles, lighting —
is *visualization configuration* layered on top of these three inputs.

## Coordinate alignment

The single most common source of bad-looking plots is a mismatch between
the mesh's coordinate system and the ROI coordinates. They must be in the
**same space**.

If you generate the coordinates from a NIfTI atlas using
[`hlplot coords generate`](../data_preparation/roi_coordinates.md), the
output is automatically in the atlas's voxel-to-world (`affine`) space.
If your mesh was made from the same NIfTI volume, they line up by
construction. See [NIfTI → mesh](../data_preparation/nifti_to_mesh.md).

## What the figure contains

The Plotly figure produced by HarrisLabPlotting has three trace groups:

1. **The mesh** — a `Mesh3d` trace with configurable lighting and opacity
2. **The nodes** — one `Scatter3d` trace per visual style (a base trace,
   plus one per role/module if you've enabled role classification)
3. **The edges** — `Scatter3d` line segments, by default split into two
   traces (positive in red, negative in blue), or split by module when
   you use module edge coloring

Toggling traces in the legend works exactly as in any other Plotly figure.

## CLI vs. Python

The two surfaces are isomorphic: every CLI flag has a corresponding Python
keyword argument and vice versa. Use the CLI for one-shot figures and
batch processing; use Python when you want to keep tweaking a single
figure interactively (e.g. in a Jupyter notebook).

```bash
hlplot plot --mesh brain.gii --coords rois.csv --matrix conn.npy
```

```python
from HarrisLabPlotting import quick_brain_plot
fig, _ = quick_brain_plot("brain.gii", "rois.csv", "conn.npy")
```

## Output formats

| Format | Use case | How |
| --- | --- | --- |
| `.html` | Sharing, exploration, embed in docs | Default; `--output foo.html` |
| `.png` | Bitmap for slide decks, posters | `--output foo.png` (uses `kaleido`) |
| `.svg` | Vector for paper main figures | `--output foo.svg` |
| `.pdf` | Vector with embedded fonts for print | `--output foo.pdf` |
| Multi-view PNG strip | Multi-panel figure with several camera angles | `--multi-view-output foo.png` |

The format is selected by the output file's extension. See
[Static export](../tutorials/static_export.md) for DPI control and panel
options.

## What the package does *not* do

- It does **not** compute connectivity matrices, modularity, or graph
  metrics. Use `bctpy`, `networkx`, or `netneurotools` upstream and feed
  their outputs into `hlplot`.
- It does **not** convert NIfTI volumes to meshes. Use `Surfice`,
  `nii2mesh`, or `FreeSurfer` upstream — see
  [NIfTI → mesh](../data_preparation/nifti_to_mesh.md).
- It does **not** do statistical testing. Use `scipy.stats`,
  permutation, NBS, or your usual stack to produce a p-value matrix; then
  feed the matrix into `hlplot --pvalue-mode`. See
  [P-value plotting](../tutorials/pvalue_plotting.md).

The package's job is the visualization itself, not the analyses upstream.
That separation is intentional and keeps the package focused.
