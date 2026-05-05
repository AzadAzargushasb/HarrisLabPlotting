# Connectivity matrix inputs

`hlplot plot` and `hlplot modular` accept a connectivity matrix in any of
six formats. Auto-detection is by file extension.

## Supported formats

| Extension | Library | Notes |
| --- | --- | --- |
| `.npy` | NumPy | Binary; fastest; preserves dtype |
| `.csv` / `.tsv` / `.txt` | pandas | Header optional; delimiter auto-detected |
| `.mat` | scipy.io | Loads the first 2-D array variable |
| `.edge` | text | BrainNet Viewer format; whitespace-delimited square matrix |

Internally everything routes through
[`load_connectivity_input`](../reference/api/index), which returns a
square `numpy.ndarray`. NaNs are converted to zero so they don't trigger
`Plotly` errors during edge rendering.

## Shape requirements

- The matrix must be **square** (`N × N`) where `N` matches the number of
  ROIs in your coordinates CSV — and in the **same row order**.
- Self-loops on the diagonal are ignored.
- Asymmetric matrices are supported; the upper-triangular entries drive
  positive edges and lower-triangular drive separately if you pass a
  directed flag (default behavior treats the matrix as symmetric and
  averages or takes the upper triangle).

## Sign convention

By default:

- **Positive** weights render with `--pos-edge-color` (red).
- **Negative** weights render with `--neg-edge-color` (blue).

Edge width is proportional to **absolute** weight, so anti-correlations
appear as thick blue edges, not invisible ones.

## P-value matrices

If your matrix is in p-value space (entries in `[0, 1]`), pass
`--pvalue-mode` to apply a `-log10(p)` transform under the hood. A `0`
remains `0`; a `0.05` becomes `1.30`; a `0.001` becomes `3.0`. This makes
small p-values render as thick edges. See
[P-value plotting](../tutorials/pvalue_plotting.md) for the full story.

## BrainNet Viewer node/edge files

If you're coming from BrainNet Viewer and have separate `.node` (8-column)
and `.edge` (matrix) files, convert them in one shot:

```bash
hlplot utils convert-node-edge \
  --node rois_28.node \
  --edge connectivity_28.edge \
  --output combined.npy
```

To combine multiple condition-specific files into a single block-diagonal
matrix, see [Combine BrainNet files](../how_to/combine_brainnet_files.md).

## Validating before plotting

```bash
hlplot utils info --matrix connectivity_matrix.csv
hlplot utils validate --mesh brain_mesh.gii \
                      --coords atlas_114_coordinates.csv \
                      --matrix connectivity_matrix.csv
```

`info` prints shape, density, value range, NaN count. `validate` checks
that the row count of `coords` matches the matrix size.
