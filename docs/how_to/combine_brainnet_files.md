# Combine BrainNet Viewer files

If you've run analyses per condition and ended up with one `.node` /
`.edge` pair per condition, `hlplot combine` assembles them into a single
block-diagonal matrix suitable for one combined plot.

This is a faithful re-implementation of BrainNet Viewer's manual stacking
workflow — but as one shell command.

## Both files at once (recommended)

```bash
hlplot combine both ./condition_files/ \
  --edge-output total.edge \
  --node-output total.node
```

`./condition_files/` should contain pairs of `<name>.node` + `<name>.edge`
with matching base names. Output:

- `total.edge` — block-diagonal matrix with each condition's matrix on the
  diagonal, zeros off-diagonal.
- `total.node` — concatenated node list, with row offsets adjusted to match
  the new block layout.

The two output files are guaranteed to have aligned row/column indices.

## Just the edges, or just the nodes

```bash
hlplot combine edges ./condition_files/ --output total.edge
hlplot combine nodes ./condition_files/ --output total.node
```

## Sort modes

The order of conditions on the diagonal matters for interpretation. Pick
one:

| Flag | Behavior |
| --- | --- |
| (default) | Filesystem directory order |
| `--alphabetical` | Sort by base name |
| `--size` | Largest matrix first |

## In Python

```python
from HarrisLabPlotting import combine_node_edge_folder

edge_path, node_path = combine_node_edge_folder(
    folder="./condition_files/",
    edge_output="total.edge",
    node_output="total.node",
    sort_mode="alphabetical",
    precision=6,
)
```

## See also

- Convert the combined files to a single `.npy` for `hlplot plot`:
  `hlplot utils convert-node-edge --node total.node --edge total.edge --output combined.npy`
- The BrainNet Viewer file format itself:
  [Connectivity matrix inputs](../data_preparation/connectivity_inputs.md#brainnet-viewer-nodeedge-files)