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
  --show-node-labels false \
  --no-width-legend \
  --title "" \
  --zoom 1.3 --image-dpi 150 \
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
    show_node_labels=False, show_width_legend=False, plot_title="",
    zoom=1.3, image_dpi=150,
    save_path="human_grid.html",
    export_image="human_modularity_grid_2x3.png",
)
```

![Human HCP-MMP1 modularity, 2x3 multi-view grid](../docs/images/figure_creation/human_modularity_grid_2x3.png)

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

*Reproduce everything: `python new_atlas_demo/generate_figure_data.py` then
`python new_atlas_demo/render_figures.py`, or run
`tutorial/figure_creation_new_atlases.ipynb`.*
