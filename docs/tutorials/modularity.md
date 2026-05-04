# Modularity visualization

Once you've run a community-detection algorithm and have a vector of module
assignments — one integer per ROI — HarrisLabPlotting can color the network
by module and (optionally) classify each node into one of seven hub/non-hub
roles using participation coefficient and within-module Z-score.

For the full CLI tour of `hlplot modular` see
[CLI walkthrough §11–12](cli_walkthrough.md#11-node-colors-from-modules).
For the role-classification scheme see [Node-role deep dive](node_role_deep_dive.md).

## Inputs

In addition to the [three basic inputs](basic_plotting.md#three-inputs) you need:

- **Module assignments** — CSV or NPY of integer module IDs, one per ROI.
- *(Optional)* **Node metrics** — CSV with `participation_coefficient` and
  `within_module_zscore` columns to drive node sizing and role classification.

## Module-colored network

The `hlplot modular` subcommand wires this up directly:

```bash
hlplot modular \
  --mesh brain_mesh.gii \
  --coords atlas_114_coordinates.csv \
  --matrix k5_state_0/connectivity_matrix.csv \
  --modules k5_state_0/module_assignments.csv \
  --output modular.html \
  --edge-color-mode module
```

`--edge-color-mode module` colors each edge by the module of its endpoints
(intra-module edges get the module color; inter-module edges get a neutral
gray). `--edge-color-mode sign` falls back to the standard red/blue scheme.

```{interactive-plot}
:image: images/cli_tutorial/12b_module_edges.png
:html: plots/brain_modularity.html
:alt: Brain network with edges colored by module
:caption: Module-colored connectivity. Each color is one community; edges between communities are muted.
:height: 540
```

## Adding node metrics

If you have per-node graph metrics, pass them with `--node-metrics`. They
power three things at once: hover tooltips, dynamic node sizing
(`--node-size pc|zscore|both`), and the Guimerà–Amaral 7-role classification
that draws colored borders around each node.

```bash
hlplot modular \
  --mesh brain_mesh.gii \
  --coords atlas_114_coordinates.csv \
  --matrix k5_state_0/connectivity_matrix.csv \
  --modules k5_state_0/module_assignments.csv \
  --node-metrics k5_state_0/combined_metrics.csv \
  --node-size both \
  --output modular_with_roles.html
```

```{interactive-plot}
:image: images/cli_tutorial/12a_q_z.png
:caption: Module-colored network with Q-statistic and Z-score in the title and node size scaled by participation coefficient × within-module Z-score.
:height: 540
```

## Python API

The full pipeline is also one Python call:

```python
from HarrisLabPlotting import (
    load_mesh_file,
    load_and_clean_coordinates,
    load_connectivity_input,
    create_brain_connectivity_plot_with_modularity,
)
import numpy as np

vertices, faces = load_mesh_file("brain_mesh.gii")
coords = load_and_clean_coordinates("atlas_114_coordinates.csv")
matrix = load_connectivity_input("k5_state_0/connectivity_matrix.csv")
modules = np.loadtxt("k5_state_0/module_assignments.csv", dtype=int)

fig, stats = create_brain_connectivity_plot_with_modularity(
    vertices=vertices,
    faces=faces,
    roi_coords_df=coords,
    connectivity_matrix=matrix,
    module_assignments=modules,
    save_path="modular.html",
)
```

For the full enhanced pipeline — including PC/Z-score node roles and
intra/inter/significant-only edge filtering — use
[`create_enhanced_modularity_visualization`](../reference/api/HarrisLabPlotting/index).

## Where to go next

- The 7 hub/non-hub roles, with the threshold diagram and color legend:
  [Node-role deep dive](node_role_deep_dive.md)
- Color and threshold individual edges: [CLI walkthrough §10–11](cli_walkthrough.md#10-node-visibility-with-edge-toggling)
- Add module-aware p-value overlays: [P-value plotting §7](pvalue_plotting.md)
- Multi-view stitched export of a modular plot:
  [Legends & multi-view §4](legends_and_multiview.md)
