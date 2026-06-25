
"""
Render the Figure-Creation tutorial figures (PNGs) from the generated data.

Run AFTER generate_figure_data.py. Writes PNGs into docs/images/figure_creation/.
The notebook tutorial/figure_creation_new_atlases.ipynb contains the same calls
and is the canonical source; this script just (re)builds the static images.

  /home/aazarg/.conda/envs/pre_env/bin/python render_figures.py
"""
from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from HarrisLabPlotting import (
    load_mesh_file,
    create_brain_connectivity_plot,
    create_brain_connectivity_plot_with_modularity,
)

HERE = Path(__file__).resolve().parent
PARC = HERE.parent / "parcellation and meshes"
REPO = HERE.parents[2]                                  # repo root
IMG = REPO / "docs" / "images" / "figure_creation"
IMG.mkdir(parents=True, exist_ok=True)
HTML = Path(tempfile.mkdtemp(prefix="figcreate_html_"))  # throwaway HTML dummies

HUMAN = HERE / "human"
MONKEY = HERE / "monkey"
HUMAN_MESH = PARC / "HCPMMP1_on_MNI152_ICBM2009a_nlin_hd_0.obj"
MONKEY_MESH = PARC / "monkey_brain_mesh_MacBNA.obj"
EXAMPLE_EDGE = HERE.parent / "node_edge_28" / "connectivity_28.edge"


def human_modularity_grid(vertices, faces):
    """Figure 1 — clean 2x3 multi-view grid of the 50-edge / 5-module network."""
    coords = pd.read_csv(HUMAN / "hcpmmp1_coords.csv")
    create_brain_connectivity_plot_with_modularity(
        vertices=vertices, faces=faces, roi_coords_df=coords,
        connectivity_matrix=str(HUMAN / "hcpmmp1_modular_network.csv"),
        module_assignments=str(HUMAN / "hcpmmp1_modules.csv"),
        plot_title="",                   # clean: no combined title
        save_path=str(HTML / "human_grid.html"),
        multi_view=["anterior", "posterior", "left", "right", "superior", "oblique"],
        multi_view_grid=(2, 3),
        multi_view_panel_size=(700, 700),
        show_node_labels=False,          # clean: no per-node text
        show_width_legend=False,         # clean: drop the edge-width key
        export_image=str(IMG / "human_modularity_grid_2x3.png"),
        image_dpi=150,
        zoom=1.3,
    )
    print("  wrote human_modularity_grid_2x3.png")


def monkey_legend_demos(vertices, faces):
    """Figures 2a-c — the legend-tutorial §1-2 walkthrough on the monkey brain."""
    coords = pd.read_csv(MONKEY / "coords_28.csv")
    sizes = str(MONKEY / "sizes_from_pc.csv")
    metrics = str(MONKEY / "metrics.csv")
    common = dict(vertices=vertices, faces=faces, roi_coords_df=coords,
                  connectivity_matrix=str(EXAMPLE_EDGE), camera_view="oblique",
                  show_node_labels=False, image_dpi=150, zoom=1.5)

    # (a) vector node sizes -> auto size key; scaled edges -> auto width key
    create_brain_connectivity_plot(
        node_size=sizes, edge_width=(1.0, 8.0), node_size_scale=0.5,
        plot_title="Monkey — vector sizes + scaled edges (auto keys)",
        save_path=str(HTML / "monkey_a.html"),
        export_image=str(IMG / "monkey_size_key.png"), **common)
    print("  wrote monkey_size_key.png")

    # (b) scalar size + fixed width -> both keys auto-skipped
    create_brain_connectivity_plot(
        node_size=10, edge_width=2.0,
        plot_title="Monkey — scalar size + fixed width (keys auto-skipped)",
        save_path=str(HTML / "monkey_b.html"),
        export_image=str(IMG / "monkey_no_keys.png"), **common)
    print("  wrote monkey_no_keys.png")

    # (c) metric-labeled size key (participation coefficient)
    create_brain_connectivity_plot(
        node_size=sizes, node_metrics=metrics,
        node_size_legend_metric="participation_coef",
        edge_width=(1.0, 8.0), node_size_scale=0.5,
        plot_title="Monkey — size key labeled by participation coefficient",
        save_path=str(HTML / "monkey_c.html"),
        export_image=str(IMG / "monkey_metric_key.png"), **common)
    print("  wrote monkey_metric_key.png")


def monkey_default_vs_custom(vertices, faces):
    """Figure 3 — default vs customized render of the same monkey network."""
    coords = pd.read_csv(MONKEY / "coords_28.csv")
    n = len(coords)
    modules = (np.arange(n) % 4) + 1     # synthetic 4-module coloring
    common = dict(vertices=vertices, faces=faces, roi_coords_df=coords,
                  connectivity_matrix=str(EXAMPLE_EDGE), camera_view="oblique",
                  show_node_labels=False, image_dpi=150, zoom=1.5)

    # Default: scalar size, fixed width, default purple nodes, no keys.
    create_brain_connectivity_plot(
        node_size=8, edge_width=2.0,
        plot_title="Default render",
        save_path=str(HTML / "monkey_default.html"),
        export_image=str(IMG / "monkey_default.png"), **common)
    print("  wrote monkey_default.png")

    # Customized: PC-scaled vector sizes, scaled edges, metric-labeled key,
    # module node colors + border.
    create_brain_connectivity_plot(
        node_size=str(MONKEY / "sizes_from_pc.csv"), node_size_scale=0.5,
        node_color=modules, node_border_color="black",
        node_metrics=str(MONKEY / "metrics.csv"),
        node_size_legend_metric="participation_coef",
        edge_width=(1.0, 8.0),
        plot_title="Customized render",
        save_path=str(HTML / "monkey_custom.html"),
        export_image=str(IMG / "monkey_customized.png"), **common)
    print("  wrote monkey_customized.png")


def main():
    print("Loading meshes ...")
    hv, hf = load_mesh_file(str(HUMAN_MESH))
    mv, mf = load_mesh_file(str(MONKEY_MESH))

    print("Rendering human modularity figure ...")
    human_modularity_grid(hv, hf)

    print("Rendering monkey figures ...")
    monkey_legend_demos(mv, mf)
    monkey_default_vs_custom(mv, mf)

    print(f"\nAll figures written to {IMG}")


if __name__ == "__main__":
    main()
