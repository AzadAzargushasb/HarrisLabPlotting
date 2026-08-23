# HarrisLabPlotting — Figure Creation with New Atlases

> Standalone CLI + notebook companion to the rendered docs page
> [`docs/tutorials/figure_creation.md`](../docs/tutorials/figure_creation.md) and the
> runnable notebook
> [`figure_creation_new_atlases.ipynb`](figure_creation_new_atlases.ipynb).

This walks through making figures on **two new atlases** — a human **HCP-MMP1**
(360 regions, MNI152) and a macaque **Brainnetome (MacBNA)** (304 regions) — by
extracting real ROI coordinates from the atlas, fabricating a small network, and
rendering.

> **What's real vs. synthetic.** The ROI **coordinates** and **names** are real
> (from the atlas volumes + their label tables). Every **connectivity matrix,
> module assignment, node size and metric** below is **synthetic** — fixed
> random seeds, for demonstration only. This is **not** the full figure set; it's
> a template to adapt. The atlas volumes/meshes are large, external data you
> supply under `test_files/tutorial_files/parcellation and meshes/`.

> **Where to download the atlases.**
> - **Human — HCP-MMP1 (360 regions, MNI152):** the lateralized volumetric
>   Glasser parcellation, [NeuroVault image 29489](https://neurovault.org/images/29489/)
>   (`MMP_in_MNI_corr.nii.gz`); region names are in the companion `roi_names.csv`.
> - **Macaque — Brainnetome (MacBNA, 304 regions):** the
>   [MacBNA dataset on Science Data Bank](https://www.scidb.cn/en/detail?dataSetId=f6eead10c2f84e9d91951cee2837048f)
>   (free account + license). Use the **native ex-vivo** label volume
>   `MacBNA__LR_304.nii.gz` (it aligns with the macaque mesh) — **not** the
>   `…_in_NMT2asym.nii.gz` variant (different, NMT space). Names come from
>   `Nomenclature_MBNA_304.xlsx`.

All commands run from `test_files/tutorial_files`. The generated data and the
scripts that build it live in `new_atlas_demo/`.

---

## 0. Prepare the data (LUT → coordinates)

The script `new_atlas_demo/generate_figure_data.py` builds both LUTs
(`index<TAB>name`, from each atlas's own label table), extracts coordinates, and
fabricates the synthetic networks deterministically:

```bash
cd test_files/tutorial_files/new_atlas_demo
python generate_figure_data.py
```

The coordinate step on its own is just `hlplot coords generate`:

```bash
hlplot coords generate \
  --volume "parcellation and meshes/HCPMMP1_on_MNI152_ICBM2009a_nlin_hd.nii/MMP_in_MNI_corr.nii.gz" \
  --labels new_atlas_demo/human/hcpmmp1_labels.txt \
  --output-dir new_atlas_demo/human --name hcpmmp1
```

> **Sanity-check a new atlas/mesh pair first.** See
> [`ALIGNMENT_CHECKS.md`](ALIGNMENT_CHECKS.md). One
> `hlplot utils check-alignment --coords … --mesh …` catches float-labeled
> atlases, wrong template spaces, and midline-collapsed bilateral atlases.

---

## 1. Human — modularity across six views (2×3 grid)

A clean 2×3 multi-view grid of a 50-edge / 5-module synthetic network: nodes
colored by module (default), no title, no edge-width key, module legend in the
first panel only.

### CLI

```bash
hlplot modular \
  --mesh "parcellation and meshes/HCPMMP1_on_MNI152_ICBM2009a_nlin_hd_0.obj" \
  --coords new_atlas_demo/human/hcpmmp1_coords.csv \
  --matrix new_atlas_demo/human/hcpmmp1_modular_network.csv \
  --modules new_atlas_demo/human/hcpmmp1_modules.csv \
  --multi-view "anterior,posterior,left,right,superior,oblique" \
  --multi-view-grid "2,3" \
  --multi-view-panel-size "700,700" \
  --show-node-labels true \
  --label-font-size 9 \
  --no-width-legend \
  --title "" \
  --zoom 1.3 --image-dpi 600 \
  --output new_atlas_demo/output/human_grid.html \
  --export-image new_atlas_demo/output/human_modularity_grid_2x3.png
```

### Python / notebook

```python
import pandas as pd
from HarrisLabPlotting import load_mesh_file, create_brain_connectivity_plot_with_modularity

vertices, faces = load_mesh_file(".../HCPMMP1_on_MNI152_ICBM2009a_nlin_hd_0.obj")
coords = pd.read_csv("new_atlas_demo/human/hcpmmp1_coords.csv")

fig, _ = create_brain_connectivity_plot_with_modularity(
    vertices=vertices, faces=faces, roi_coords_df=coords,
    connectivity_matrix="new_atlas_demo/human/hcpmmp1_modular_network.csv",
    module_assignments="new_atlas_demo/human/hcpmmp1_modules.csv",
    multi_view=["anterior", "posterior", "left", "right", "superior", "oblique"],
    multi_view_grid=(2, 3), multi_view_panel_size=(700, 700),
    show_node_labels=True, label_font_size=9, show_width_legend=False, plot_title="",
    zoom=1.3, image_dpi=600,
    save_path="human_grid.html",
    export_image="human_modularity_grid_2x3.png",
)
```

![Human HCP-MMP1 modularity, 2x3 multi-view grid](../docs/images/figure_creation/human_modularity_grid_2x3.png)
*Nodes are labeled with their HCP-MMP short names (`V1_L`, `MST_L`, …).*

---

## 2. Monkey — per-node sizes and legend keys

Reuse the bundled 28-node example topology (`node_edge_28/connectivity_28.edge`)
on 28 real MacBNA ROIs; node sizes are pre-scaled from a synthetic participation
coefficient.

### (a) Vector sizes + scaled edges → both keys appear automatically

```bash
hlplot plot \
  --mesh "parcellation and meshes/monkey_brain_mesh_MacBNA.obj" \
  --coords new_atlas_demo/monkey/coords_28.csv \
  --matrix node_edge_28/connectivity_28.edge \
  --node-size new_atlas_demo/monkey/sizes_from_pc.csv \
  --edge-width-min 1 --edge-width-max 8 \
  --node-size-scale 0.5 --zoom 1.5 --camera oblique \
  --output new_atlas_demo/output/monkey_a.html \
  --export-image new_atlas_demo/output/monkey_size_key.png
```

```python
from HarrisLabPlotting import load_mesh_file, create_brain_connectivity_plot
import pandas as pd

vertices, faces = load_mesh_file(".../monkey_brain_mesh_MacBNA.obj")
coords = pd.read_csv("new_atlas_demo/monkey/coords_28.csv")

create_brain_connectivity_plot(
    vertices=vertices, faces=faces, roi_coords_df=coords,
    connectivity_matrix="node_edge_28/connectivity_28.edge",
    node_size="new_atlas_demo/monkey/sizes_from_pc.csv",
    edge_width=(1.0, 8.0), node_size_scale=0.5, zoom=1.5, camera_view="oblique",
    save_path="monkey_a.html", export_image="monkey_size_key.png",
)
```

![Monkey vector sizes with auto keys](../docs/images/figure_creation/monkey_size_key.png)

### (b) Scalar size + fixed width → both keys auto-skipped

```bash
hlplot plot \
  --mesh "parcellation and meshes/monkey_brain_mesh_MacBNA.obj" \
  --coords new_atlas_demo/monkey/coords_28.csv \
  --matrix node_edge_28/connectivity_28.edge \
  --node-size 10 --edge-width-fixed 2 \
  --zoom 1.5 --camera oblique \
  --output new_atlas_demo/output/monkey_b.html \
  --export-image new_atlas_demo/output/monkey_no_keys.png
```

![Monkey scalar size + fixed width, no keys](../docs/images/figure_creation/monkey_no_keys.png)

### (c) Label the size key with the metric

```bash
hlplot plot \
  --mesh "parcellation and meshes/monkey_brain_mesh_MacBNA.obj" \
  --coords new_atlas_demo/monkey/coords_28.csv \
  --matrix node_edge_28/connectivity_28.edge \
  --node-size new_atlas_demo/monkey/sizes_from_pc.csv \
  --node-metrics new_atlas_demo/monkey/metrics.csv \
  --node-size-legend-metric participation_coef \
  --edge-width-min 1 --edge-width-max 8 \
  --node-size-scale 0.5 --zoom 1.5 --camera oblique \
  --output new_atlas_demo/output/monkey_c.html \
  --export-image new_atlas_demo/output/monkey_metric_key.png
```

![Monkey size key labeled by participation coefficient](../docs/images/figure_creation/monkey_metric_key.png)

---

## 3. Monkey — default vs. customized

| | Default | Customized |
|---|---|---|
| Node size | scalar `8` | vector, PC-scaled |
| Node color | default purple | by module (+ black border) |
| Edge width | fixed `2.0` | scaled `(1, 8)` |
| Legend keys | none | size key (PC) + width key |

```python
import numpy as np
modules = (np.arange(len(coords)) % 4) + 1

create_brain_connectivity_plot(            # default
    vertices=vertices, faces=faces, roi_coords_df=coords,
    connectivity_matrix="node_edge_28/connectivity_28.edge",
    node_size=8, edge_width=2.0, zoom=1.5, camera_view="oblique",
    save_path="monkey_default.html", export_image="monkey_default.png")

create_brain_connectivity_plot(            # customized
    vertices=vertices, faces=faces, roi_coords_df=coords,
    connectivity_matrix="node_edge_28/connectivity_28.edge",
    node_size="new_atlas_demo/monkey/sizes_from_pc.csv", node_size_scale=0.5,
    node_color=modules, node_border_color="black",
    node_metrics="new_atlas_demo/monkey/metrics.csv",
    node_size_legend_metric="participation_coef",
    edge_width=(1.0, 8.0), zoom=1.5, camera_view="oblique",
    save_path="monkey_custom.html", export_image="monkey_customized.png")
```

| Default | Customized |
|---|---|
| ![Default](../docs/images/figure_creation/monkey_default.png) | ![Customized](../docs/images/figure_creation/monkey_customized.png) |

---

## 4. Modularity visualization types (114-ROI k5 example)

The `viz_type` / `inter_edge_color` / `node_roles` knobs render the same *k=5*
community result (the bundled **rat** brain `brain_mesh.gii` + `k5_state_0/`,
114 ROIs, 6 modules; the region names are rodent) several ways. Each is saved as a 3-view multi-view PNG, a superior single, and a
local interactive HTML.

```bash
# vary --viz-type; add --inter-edge-color black or --node-roles per type
hlplot modular \
  --mesh brain_mesh.gii \
  --coords output/atlas_114_test/atlas_114_test_comma.csv \
  --matrix k5_state_0/connectivity_matrix.csv \
  --modules k5_state_0/module_assignments.csv \
  --node-metrics k5_state_0/combined_metrics.csv \
  --viz-type all \
  --multi-view "left,superior,posterior" \
  --output k5_all.html --export-image k5_all_multiview.png
```

| Type | Flags |
|---|---|
| All edges (default) | `--viz-type all` |
| All edges, inter black | `--viz-type all --inter-edge-color black` |
| Intra-module only | `--viz-type intra` |
| Inter-module only | `--viz-type inter` |
| Inter-module only, black | `--viz-type inter --inter-edge-color black` |
| Nodes only | `--viz-type nodes_only` |
| Nodal roles, no edges | `--viz-type nodes_only --node-roles --node-metrics k5_state_0/combined_metrics.csv` |
| Nodal roles, with edges | `--viz-type all --node-roles --node-metrics k5_state_0/combined_metrics.csv` |

| All edges | Intra-module | Nodal roles (superior) |
|---|---|---|
| ![all](../docs/images/figure_creation/k5/default_multiview.png) | ![intra](../docs/images/figure_creation/k5/intra_multiview.png) | ![roles](../docs/images/figure_creation/k5/nodal_roles_superior.png) |

### Nodal roles — with and without edges

`--node-roles` classifies each node by the Guimerà–Amaral cartographic two-cut
(needs `--node-metrics` with `participation_coef` + `within_module_zscore`) and
draws the role as a colored **border ring**; the node fill stays its module
color. It composes with any `--viz-type`, so you can show roles *without* edges
(`--viz-type nodes_only`) or *with* the full edge set (`--viz-type all`):

| Nodal roles, no edges | Nodal roles, with edges |
|---|---|
| ![roles no edges](../docs/images/figure_creation/k5/nodal_roles_multiview.png) | ![roles with edges](../docs/images/figure_creation/k5/nodal_roles_edges_multiview.png) |

```bash
# roles WITH edges (drop --viz-type all -> nodes_only for the no-edge version)
hlplot modular \
  --mesh brain_mesh.gii \
  --coords output/atlas_114_test/atlas_114_test_comma.csv \
  --matrix k5_state_0/connectivity_matrix.csv \
  --modules k5_state_0/module_assignments.csv \
  --node-metrics k5_state_0/combined_metrics.csv \
  --viz-type all --node-roles \
  --multi-view "left,superior,posterior" \
  --output k5_roles_edges.html --export-image k5_nodal_roles_edges_multiview.png
```

---

## 5. Cross-species comparison grid (human / rat / macaque)

`--multi-view` shows **one** mesh from several cameras. To put **different
meshes side by side** — a human, a rat and a macaque brain in one figure — render
each panel separately and stitch them with the new `hlplot montage` command (the
CLI face of `compose_image_grid`).

This is a 2×3 grid: columns are the three species, and the six cells are the six
canonical BrainNet views (row 1 = left / superior / right, row 2 = anterior /
inferior / posterior). Each species shows a minimal module-colored network on its
own mesh. Three versions are produced (labels off, full names, and short-form) —
shown below.

```python
import pandas as pd
from HarrisLabPlotting import (
    load_mesh_file, create_brain_connectivity_plot_with_modularity, compose_image_grid,
)

# Render each (species, view) panel to its own PNG (single-view, no legend), then
# compose. See new_atlas_demo/render_species_grid.py for the full loop; the key
# call for one panel:
create_brain_connectivity_plot_with_modularity(
    vertices=v, faces=f, roi_coords_df=coords,
    connectivity_matrix=matrix, module_assignments=modules,
    node_size=10, edge_width=2.0, show_node_labels=False,
    show_width_legend=False, plot_title="",
    multi_view=["left"],                 # one view -> tight autocropped panel
    multi_view_panel_size=(500, 500),
    multi_view_keep_first_legend=False,
    multi_view_panel_labels=[""],
    image_dpi=600, zoom=1.3,
    save_path="dummy.html", export_image="human_left.png",
)

# ... six panels later, compose them row-major into the grid:
compose_image_grid(
    ["human_left.png", "rat_superior.png", "macaque_right.png",
     "human_anterior.png", "rat_inferior.png", "macaque_posterior.png"],
    "species_grid.png",
    grid=(2, 3),
    col_labels=["Human", "Rat", "Macaque"],
    panel_labels=["Left", "Superior", "Right", "Anterior", "Inferior", "Posterior"],
)
```

```bash
# After rendering the six panel PNGs (one per species+view) with hlplot modular:
hlplot montage \
  --images "human_left.png,rat_superior.png,macaque_right.png,human_anterior.png,rat_inferior.png,macaque_posterior.png" \
  --grid "2,3" \
  --col-labels "Human,Rat,Macaque" \
  --panel-labels "Left,Superior,Right,Anterior,Inferior,Posterior" \
  --output species_grid.png
```

Three versions are produced — labels off, full `roi_name`, and **short-form** labels
(hemisphere suffix stripped: `V1_L`→`V1`, `AUD_left`→`AUD`, `IFG.cv_left`→`IFG.cv`),
which keeps 28–30 labels legible.

| ROI labels off | ROI labels on (full) |
|---|---|
| ![species grid, no labels](../docs/images/figure_creation/species/species_grid_nolabels.png) | ![species grid, labeled](../docs/images/figure_creation/species/species_grid_labeled.png) |

![species grid, short-form labels](../docs/images/figure_creation/species/species_grid_shortform.png)
*Short-form labels — `roi_name` minus the hemisphere suffix.*

*The rat mesh is the bundled `brain_mesh.gii`; the human and macaque meshes are the
HCP-MMP1 and MacBNA surfaces. `hlplot montage` auto-crops each panel and adds the
column headers + per-cell view labels. Camera zoom is set **per (species, view)**
(`ZOOM` in the script), since a view whose brain fills more of its panel otherwise
reads as too zoomed in beside the others.*

---

## 6. Scaling edges and nodes by p-value significance

A p-value network can encode significance two ways at once: **edge width** by
`-log10(p)` (built in via `--matrix-type pvalue`) and **node size** by a per-node
significance you derive yourself. Passing a `--sign-matrix` also colors each edge
by **direction — red = positive, blue = negative (opposite-direction) effect**.
The figure below contrasts a flat baseline (uniform width + scalar size) with the
fully scaled version. It uses `pvalues_28_spread.csv` (significance spread over ~5
orders of magnitude, so the widths span the full range) — see the
[p-value tutorial](PVALUE_PLOTTING_TUTORIAL.md) for how that file is built.

```python
import numpy as np, pandas as pd
from HarrisLabPlotting import load_mesh_file, create_brain_connectivity_plot

pvals = "node_edge_28/pvalues_28_spread.csv"
signs = "node_edge_28/pvalues_28_signs.csv"   # +1/-1 direction -> red / blue

# Per-node significance = sum of -log10(p) over each node's surviving edges.
P = np.loadtxt(pvals, delimiter=",")
thr = 0.05
W = np.where((P > 0) & (P <= thr), -np.log10(np.clip(P, 1e-300, 1.0)), 0.0)
np.fill_diagonal(W, 0.0)
sig = W.sum(axis=1)
node_px = 6 + (sig - sig.min()) / (sig.max() - sig.min()) * (24 - 6)   # -> [6, 24] px

common = dict(
    vertices=v, faces=f, roi_coords_df=coords,
    connectivity_matrix=pvals, matrix_type="pvalue", pvalue_threshold=thr,
    sign_matrix=signs,            # red positive / blue negative
    edge_width_scale=2, camera_view="superior", image_dpi=600)

# (a) uniform baseline: fixed width + scalar size (direction still shown by color)
create_brain_connectivity_plot(**common, edge_width=2.0, node_size=8,
    export_image="pval_uniform.png", save_path="pval_uniform.html")

# (b) scaled: edge width ~ -log10(p), node size ~ per-node significance
create_brain_connectivity_plot(**common,
    edge_width=(1.0, 9.0), node_size=node_px,
    node_metrics=pd.DataFrame({"roi_name": coords["roi_name"], "node_significance": sig}),
    node_size_legend_metric="node_significance",
    export_image="pval_scaled.png", save_path="pval_scaled.html")
```

Single superior view:

| Uniform (no scaling) | Significance-scaled edges + nodes |
|---|---|
| ![pval uniform](../docs/images/figure_creation/pvalue/pval_uniform.png) | ![pval scaled](../docs/images/figure_creation/pvalue/pval_scaled.png) |

Same figures as a 3-view multi-view strip (left / superior / posterior):

| Uniform | Scaled |
|---|---|
| ![pval uniform multiview](../docs/images/figure_creation/pvalue/pval_uniform_multiview.png) | ![pval scaled multiview](../docs/images/figure_creation/pvalue/pval_scaled_multiview.png) |

*Thicker edges = smaller p (see the p-value width key); bigger nodes = higher
summed significance (see the `node_significance` size key); **red edges are
positive, blue edges negative** (the legend splits into "Positive Edges" /
"Negative Edges"). The `PVALUE_THRESHOLD`, edge-width and size ranges are exposed
at the top of `render_pvalue_scaling.py` / the notebook cell so they are easy to
tweak.*

---

## 7. Directed networks

An **asymmetric** matrix — DCM, Granger causality, transition probabilities —
carries a different number in each direction, so `hlplot` draws arrowheads. It
detects this automatically and reports the verdict on every plot.

> **Which index is the source?** `hlplot` reads `M[i, j]` as **i → j** (row =
> source). SPM DCM stores the transpose and needs
> `--matrix-orientation col-to-row`; a row-stochastic transition matrix does
> not. Getting this wrong reverses every arrow and nothing warns you — see
> [DIRECTED_GRAPHS.md](DIRECTED_GRAPHS.md) §1.

```bash
hlplot plot \
  --mesh brain_mesh.gii \
  --coords output/atlas_28_test_comma.csv \
  --matrix node_edge_28/directed_28.csv \
  --edge-width-min 1 --edge-width-max 9 \
  --multi-view "left,superior,posterior" --zoom 1.3 \
  --output new_atlas_demo/output/directed.html \
  --export-image new_atlas_demo/output/directed_multiview.png
```

```python
from HarrisLabPlotting import load_mesh_file, create_brain_connectivity_plot
import pandas as pd

vertices, faces = load_mesh_file("brain_mesh.gii")
coords = pd.read_csv("output/atlas_28_test_comma.csv")

fig, stats = create_brain_connectivity_plot(
    vertices=vertices, faces=faces, roi_coords_df=coords,
    connectivity_matrix="node_edge_28/directed_28.csv",
    edge_width=(1.0, 9.0), zoom=1.3,
    multi_view=["left", "superior", "posterior"],
    save_path="directed.html", export_image="directed_multiview.png",
)
print(stats["symmetry"])
```

**Figure 8 — three views:**

![Directed network, three views](../docs/images/directed/02_multiview.png)

**and the same network from a single superior view:**

![Directed network, superior](../docs/images/directed/01_quickstart_superior.png)

*One-way edges are straight; reciprocal pairs bow to opposite sides so both
directions keep their own width and arrowhead. Arrowheads scale with their own
edge's width and are capped so a short edge never gets a head longer than
itself.*

---

## 8. Voxel maps on a glass brain

Statistical volumes render as a soft ray-cast cloud inside a translucent brain.
The mouse Fig 1 z-maps ship under `test_files/tutorial_files/mouse/`.

```bash
hlplot volume \
  --mesh mouse/bin_dilD_Parc_Atlas_0.obj \
  --volume mouse/Fig1_RM_Sham_pos_z_allen.nii.gz \
      --volume-cmap hot32 --volume-name Activation \
  --volume mouse/Fig1_RM_Sham_neg_z_allen.nii.gz \
      --volume-cmap ice28 --volume-name Deactivation \
  --volume-threshold 3.1 --volume-smooth-fwhm "0.54,0.11,0.11" \
  --volume-step 7 --zoom 1.25 \
  --multi-view "left,superior,posterior" \
  --no-html --export-image new_atlas_demo/output/voxels_multiview.png
```

```python
from HarrisLabPlotting import create_brain_volume_plot

create_brain_volume_plot(
    mesh="mouse/bin_dilD_Parc_Atlas_0.obj",
    volumes=[
        dict(path="mouse/Fig1_RM_Sham_pos_z_allen.nii.gz", name="Activation",
             cmap="hot32", threshold=3.1, smooth_fwhm="0.54,0.11,0.11", step=7),
        dict(path="mouse/Fig1_RM_Sham_neg_z_allen.nii.gz", name="Deactivation",
             cmap="ice28", threshold=3.1, smooth_fwhm="0.54,0.11,0.11", step=7),
    ],
    zoom=1.25, multi_view=["left", "superior", "posterior"],
    no_html=True, export_image="voxels_multiview.png",
)
```

**Figure 9 — three views:**

![Voxel maps, three views](../docs/images/voxel/02_multiview.png)

**and a single superior view:**

![Voxel maps, superior](../docs/images/voxel/01_quickstart_superior.png)

*`hot32` / `ice28` are matplotlib's `hot` truncated at 0.32 and a custom `ice`
at 0.28 — the same colormaps as the study's 2-D coronal montages. The
`--volume-smooth-fwhm` value is the ORIGINAL pre-warp voxel size, which is what
removes the stair-steps left by resampling 16 thick slices onto a 25 µm grid.
Full detail in [VOXEL_PLOTTING.md](VOXEL_PLOTTING.md).*

---

*Reproduce everything: `python new_atlas_demo/generate_figure_data.py`, then
`python new_atlas_demo/render_figures.py`,
`python new_atlas_demo/render_k5_viztypes.py`,
`python new_atlas_demo/render_species_grid.py`,
`python new_atlas_demo/generate_pvalue_spread.py`,
`python new_atlas_demo/render_pvalue_scaling.py`,
`python new_atlas_demo/generate_directed_demo.py`,
`python new_atlas_demo/render_directed.py`,
`python new_atlas_demo/render_voxels.py`, and
`python new_atlas_demo/render_combined.py`, or run
`tutorial/figure_creation_new_atlases.ipynb`. All figures render at 600 DPI.*
