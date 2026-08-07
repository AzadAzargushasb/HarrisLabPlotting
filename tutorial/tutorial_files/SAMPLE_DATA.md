# Sample data files for the tutorials

The reproducible fixtures the tutorials and notebooks use live under
**`test_files/tutorial_files/`** (relative to the repo root). Everything below is
committed to the repo, so every CLI/Python example runs without any external
downloads (the one exception — the large external atlas volumes/meshes for the
figure-creation tutorial — is noted at the bottom).

---

## Layout

```
test_files/tutorial_files/
├── brain_mesh.gii                 # bundled rat brain surface (GIFTI) used by the
│                                  #   28- and 114-ROI tutorials
├── node_edge_28/                  # 28-ROI example network
│   ├── connectivity_28.edge       #   28x28 weighted matrix (18 pos / 9 neg edges)
│   ├── rois_28.node               #   BrainNet Viewer node file (names + xyz)
│   ├── modules_28.csv             #   synthetic 4-module assignment
│   ├── pvalues_28.csv / .npy      #   28x28 p-value matrix (original)
│   ├── pvalues_28_spread.csv/.npy #   p-values spread over ~5 orders of magnitude
│   │                              #     (even edge-width demo; see below)
│   ├── pvalues_28_signs.csv/.npy  #   +1/-1/0 direction -> red / blue edges
│   ├── pvalues_28_tier_labels.csv #   significance-tier labels
│   ├── edge_groups.csv            #   per-edge categorical groups (color-matrix demo)
│   └── show_labels_hubs_28.csv    #   per-node label mask (label only hubs)
├── k5_state_0/                    # 114-ROI real community-detection result (k=5)
│   ├── connectivity_matrix.csv    #   114x114 connectivity
│   ├── module_assignments.csv     #   6 modules
│   └── combined_metrics.csv       #   participation_coef + within_module_zscore
│                                  #     (needed for nodal roles)
├── output/                        # pre-generated coordinates (mesh order)
│   ├── atlas_28_test_comma.csv    #   28-ROI COG coords
│   └── atlas_114_test_comma.csv   #   114-ROI COG coords
└── new_atlas_demo/                # figure-creation demo (see tutorial/FIGURE_CREATION.md)
    ├── generate_figure_data.py    #   builds the HCP-MMP1 / MacBNA LUTs + coords + nets
    ├── generate_pvalue_spread.py  #   builds pvalues_28_spread.csv (deterministic)
    ├── render_figures.py          #   human 2x3 grid + monkey legend demos
    ├── render_k5_viztypes.py      #   k5 modularity viz-types + nodal roles
    ├── render_species_grid.py     #   cross-species montage grid (3 label versions)
    ├── render_pvalue_scaling.py   #   p-value uniform vs significance-scaled (signed)
    ├── _figpaths.py               #   150-DPI committed / 600-DPI publication switch
    ├── human/  monkey/            #   per-atlas coords + synthetic networks
    └── parcellation and meshes/   #   EXTERNAL atlas volumes/meshes (not committed)
```

---

## Two p-value matrices

`node_edge_28/` ships **two** p-value matrices, both sharing the exact edge topology
of `connectivity_28.edge` (so `pvalues_28_signs.csv` applies to either):

- **`pvalues_28.csv`** — the original. Its noise term makes the surviving p-values
  bunch just under 0.05, so with width ∝ `-log10(p)` almost every edge renders at a
  similar thin width.
- **`pvalues_28_spread.csv`** — edges ranked by strength and assigned **log-spaced**
  p-values from `1e-6` to `0.045` (20 survive `p <= 0.05`). Edge widths then spread
  evenly across the whole range — the better choice for *demonstrating* width-encoded
  significance. Regenerate with `new_atlas_demo/generate_pvalue_spread.py`.

`pvalues_28_signs.csv` gives each edge a direction (`np.sign` of the connectivity):
positive edges render **red**, negative (opposite-direction) edges **blue**.

---

## Example usage

**28-ROI weighted network:**

```bash
cd test_files/tutorial_files
hlplot plot \
  --mesh brain_mesh.gii \
  --coords output/atlas_28_test_comma.csv \
  --matrix node_edge_28/connectivity_28.edge \
  --output brain_28.html
```

**114-ROI modularity:**

```bash
hlplot modular \
  --mesh brain_mesh.gii \
  --coords output/atlas_114_test_comma.csv \
  --matrix k5_state_0/connectivity_matrix.csv \
  --modules k5_state_0/module_assignments.csv \
  --node-metrics k5_state_0/combined_metrics.csv \
  --output modularity_114.html
```

**Signed p-value network (red = positive, blue = negative):**

```bash
hlplot plot \
  --mesh brain_mesh.gii \
  --coords output/atlas_28_test_comma.csv \
  --matrix node_edge_28/pvalues_28_spread.csv \
  --matrix-type pvalue --pvalue-threshold 0.05 \
  --sign-matrix node_edge_28/pvalues_28_signs.csv \
  --edge-width-min 1 --edge-width-max 9 \
  --output pval_signed.html
```

---

## External data (figure-creation tutorial only)

The cross-species figure-creation tutorial uses large external atlas volumes and
surface meshes that are **not committed** (they live under
`test_files/tutorial_files/parcellation and meshes/`). Download links and the
alignment pre-flight checks are in
[FIGURE_CREATION.md](../FIGURE_CREATION.md) and
[ALIGNMENT_CHECKS.md](../ALIGNMENT_CHECKS.md).

---

*See the main [README.md](../../README.md) for the full tutorial index.*
