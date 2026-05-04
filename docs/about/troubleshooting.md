# Troubleshooting

Common issues and their fixes, grouped by where in the pipeline they
typically surface.

## Installation

### `kaleido` fails on Linux

Symptom: `ModuleNotFoundError: No module named 'kaleido'` after install,
or `OSError: libgbm.so.1: cannot open shared object file` when calling
`fig.write_image(...)`.

Fix:

```bash
sudo apt-get install -y libgbm1
pip install --upgrade kaleido
```

The `libgbm` dependency is a quirk of the headless Chromium that
`kaleido` ships. Most managed compute environments (Slurm, Kubernetes)
already have it; minimal Docker images often don't.

### conda vs. pip — which should I use?

Use **conda** if:

- You need `nibabel` and the rest of the neuroimaging stack to coexist
  cleanly with PyTorch, JAX, or other heavy native libraries.
- You're on Windows and want pre-built wheels for everything.

Use **pip** if:

- You're already in a working scientific-Python environment.
- You prefer a minimal install (just the runtime deps, ~10 packages).

The two flows recommended in this project:

```bash
# conda (recommended)
conda env create -f environment.yml
conda activate harris_lab_plotting
pip install -e .

# pip-only
python -m venv .venv && source .venv/bin/activate
pip install -e .
```

### Old NumPy crash

Symptom: `AttributeError: module 'numpy' has no attribute 'bool'` (or
`'int'`, `'float'`).

Cause: a transitive dep is pinned to a NumPy < 1.20 idiom that has been
removed in modern NumPy. Pin NumPy:

```bash
pip install "numpy>=1.21,<2.1"
```

We've never hit this on a fresh install but it shows up when retrofitting
into an old environment.

## Plotting

### "Empty figure" / no nodes visible

The most common cause is **edge threshold too high** combined with
`--show-only-connected-nodes`. Lower or remove the threshold first:

```bash
hlplot plot ... --edge-threshold 0
```

Second most common cause: the matrix you loaded was full of NaNs. Run:

```bash
hlplot utils info --matrix your_matrix.csv
```

If `non_zero` is `0`, the loader replaced all entries with zero.

### Mesh and ROIs don't line up

Symptom: nodes float in space outside the brain surface, or sit at the
origin instead of inside the volume.

Cause: the coordinate CSV is in a different coordinate system from the
mesh. Common culprits:

- Mesh in MNI space, coordinates in voxel space (need affine multiply)
- LAS vs. RAS axis convention (flip X)
- Different origin (mesh centered, coordinates not)

Diagnostic: open the HTML, click "show coords on hover" — if the
coordinates printed in the hover are in voxel ranges (e.g. 0–256) but the
mesh extends from −80 to +80, you need to apply the NIfTI affine before
exporting the coordinates. The `coordinate_function` in
[`HarrisLabPlotting`](../reference/api/HarrisLabPlotting/index) handles this for you when
you generate from NIfTI directly.

### Mesh looks chunky / faceted

Default lighting is `flat`, which preserves the polygons. Switch:

```bash
hlplot plot ... --mesh-style smooth
```

Other presets: `matte`, `glossy`, `mirror`. See the
[CLI walkthrough](../tutorials/cli_walkthrough.md) for visual comparisons.

### Mesh too dense → slow rendering

For meshes with > 200K vertices, downsample before plotting:

- Re-run `nii2mesh` with smoothing turned up (see
  [NIfTI → mesh](../data_preparation/nifti_to_mesh.md))
- Or use Surfice's "decimate" function before exporting

For *interactive* (HTML) plots, vertex counts > 100K start to feel
sluggish in the browser even on a fast machine.

## Static export (PNG/SVG/PDF)

### Empty PNG / black PNG

Symptom: PNG file is created but is blank, all-black, or only the
background renders.

Fix: pre-cache the kaleido executable:

```bash
python -c "import plotly.io as pio; pio.kaleido.scope.default_format = 'png'"
```

If that fails too, your kaleido install is broken — reinstall it.

### "My HTML output is huge"

Plotly inlines the entire `plotly.min.js` bundle (~3.5 MB) into every
exported HTML by default. For sharing single files this is fine. For
embedding in docs or websites, use the CDN reference instead:

```python
fig.write_html("plot.html", include_plotlyjs="cdn")
```

The HTML drops to ~200 KB plus your data. The site reads `plotly.js` from
the Plotly CDN at view time. (This is exactly what the docs site here does
for its 15 embedded interactive plots.)

### SVG fonts look wrong in Illustrator

Plotly's SVG export uses CSS-style font fallbacks; Illustrator doesn't
always honor them. If you need vector text in the exact font you specified,
export PDF instead — PDF embeds the font directly.

## Modularity

### Q-score reported is much lower than netneurotools

Cause: matrix sign convention. `hlplot modular` treats the absolute
value of edge weights for module *coloring* but the Q-statistic in your
title comes from whatever you pass in via `--q-score`. If you computed Q on
positive-only edges but pass the raw matrix, the *visual* will look
different from the score.

Fix: pre-threshold or sign-filter the matrix before passing it in:

```bash
hlplot utils threshold --matrix conn.csv --output pos_only.npy --absolute 0
```

### Node-role borders all the same color

Cause: your `--node-metrics` CSV doesn't have the expected column names.
The classifier looks for **`participation_coefficient`** and
**`within_module_zscore`** (or column-name variations like
`participation_coef`, `pc`, `z_score`). Open the CSV and confirm.

## CLI

### "Unknown command: hlplot"

You installed the package but the entry point isn't on your `$PATH`. With
`pip install -e .` from the repo root, `hlplot` should land in your
environment's `bin/`. Verify:

```bash
which hlplot
python -c "import HarrisLabPlotting; print(HarrisLabPlotting.__file__)"
```

If the second works but the first doesn't, your environment's bin
directory isn't on `$PATH` — common with `pip install --user`.

### CLI flag does nothing

Options are *position-sensitive* with respect to subcommands. `hlplot
--mesh foo plot` is **wrong**; the `--mesh` flag belongs to the `plot`
subcommand:

```bash
hlplot plot --mesh foo ...   # ✓
hlplot --mesh foo plot ...   # ✗
```

When in doubt, run `hlplot plot --help` (or `hlplot modular --help`,
etc.) to see flags scoped to that subcommand.

## Still stuck?

- Check open issues:
  [github.com/AzadAzargushasb/HarrisLabPlotting/issues](https://github.com/AzadAzargushasb/HarrisLabPlotting/issues)
- File a new issue with: the exact CLI command, the full traceback, the
  output of `pip list | grep -E "plotly|kaleido|nibabel|numpy"`, and your
  OS / Python version.
