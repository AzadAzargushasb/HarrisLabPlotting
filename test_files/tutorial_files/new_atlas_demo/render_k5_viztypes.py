"""
Render the k5 (114-ROI) modularity *visualization types* for the docs.

Uses the bundled RAT tutorial brain (brain_mesh.gii) + the real k5
community-detection result (connectivity, module assignments, node metrics). The
114 region names are rodent (Accumbens_left, RSGc_left, S1_left, ...). For each of
the 8 viz types it writes a 3-view multi-view PNG, a single superior-view PNG, and
interactive HTML.

  PNGs  -> docs/images/figure_creation/k5/        (committed)
  HTMLs -> new_atlas_demo/k5_outputs/             (gitignored, local only)

Run with the project env:
  /home/aazarg/.conda/envs/pre_env/bin/python render_k5_viztypes.py
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from HarrisLabPlotting import load_mesh_file, create_brain_connectivity_plot_with_modularity

HERE = Path(__file__).resolve().parent
TF = HERE.parent                                  # test_files/tutorial_files
REPO = HERE.parents[2]
from _figpaths import FIG_DPI, fig_root
IMGK = fig_root(REPO) / "k5"
IMGK.mkdir(parents=True, exist_ok=True)
HTMLK = HERE / "k5_outputs"
HTMLK.mkdir(parents=True, exist_ok=True)

MESH = TF / "brain_mesh.gii"
COORDS = TF / "output" / "atlas_114_test" / "atlas_114_test_comma.csv"
MATRIX = TF / "k5_state_0" / "connectivity_matrix.csv"
MODULES = TF / "k5_state_0" / "module_assignments.csv"
METRICS = TF / "k5_state_0" / "combined_metrics.csv"

MULTI_VIEW = ["left", "superior", "posterior"]

# The 7 visualization types. `kw` holds the type-specific knobs.
VIZ_TYPES = [
    dict(key="default", title="Modularity — all edges",
         kw=dict(viz_type="all")),
    dict(key="all_inter_black", title="Modularity — all edges, inter-module black",
         kw=dict(viz_type="all", edge_color_mode="module", inter_edge_color="black")),
    dict(key="intra", title="Modularity — intra-module edges only",
         kw=dict(viz_type="intra")),
    dict(key="inter", title="Modularity — inter-module edges only",
         kw=dict(viz_type="inter")),
    dict(key="inter_black", title="Modularity — inter-module edges only (black)",
         kw=dict(viz_type="inter", edge_color_mode="module", inter_edge_color="black")),
    dict(key="nodes_only", title="Modularity — nodes only",
         kw=dict(viz_type="nodes_only", show_width_legend=False)),
    dict(key="nodal_roles", title="Nodal roles (Guimera-Amaral), no edges",
         kw=dict(viz_type="nodes_only", node_roles=True, show_width_legend=False)),
    dict(key="nodal_roles_edges", title="Nodal roles (Guimera-Amaral) with edges",
         kw=dict(viz_type="all", node_roles=True, show_width_legend=False)),
]


def main():
    print("Loading mesh + coords ...")
    vertices, faces = load_mesh_file(str(MESH))
    coords = pd.read_csv(COORDS)

    base = dict(
        vertices=vertices, faces=faces, roi_coords_df=coords,
        connectivity_matrix=str(MATRIX),
        module_assignments=str(MODULES),
        node_metrics=str(METRICS),   # hover tooltips (+ required for nodal roles)
        node_size=10, image_dpi=FIG_DPI,
        show_node_labels=False,      # 114 nodes -> labels would be unreadable
    )

    for v in VIZ_TYPES:
        key, title, kw = v["key"], v["title"], v["kw"]
        # Multi-view (3 views) PNG + interactive HTML.
        create_brain_connectivity_plot_with_modularity(
            **base, **kw,
            plot_title=title,
            multi_view=MULTI_VIEW,
            multi_view_panel_size=(700, 700),
            save_path=str(HTMLK / f"{key}.html"),
            export_image=str(IMGK / f"{key}_multiview.png"),
        )
        # Single superior view PNG + interactive HTML.
        create_brain_connectivity_plot_with_modularity(
            **base, **kw,
            plot_title=title,
            camera_view="superior",
            save_path=str(HTMLK / f"{key}_superior.html"),
            export_image=str(IMGK / f"{key}_superior.png"),
        )
        print(f"  [{key}] wrote {key}_multiview.png + {key}_superior.png + HTML")

    print(f"\nPNGs -> {IMGK}\nHTMLs -> {HTMLK}")


if __name__ == "__main__":
    main()
