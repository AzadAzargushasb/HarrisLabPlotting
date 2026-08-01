"""
Render the cross-species comparison grid for the docs.

A 2x3 grid whose COLUMNS are three different brain meshes (human / rat / macaque)
and whose six cells are six distinct orthographic views, filled in the canonical
BrainNet 6-view order:

    row 1 :  Left      Superior   Right
    row 2 :  Anterior  Inferior   Posterior

so Human shows Left + Anterior, Rat shows Superior + Inferior, Macaque shows
Right + Posterior. Each species column displays a minimal module-colored network
(its own small synthetic community structure).

Three versions are produced:
  nolabels  -- no ROI labels
  labeled   -- full roi_name (e.g. V1_L, AUD_left, IFG.cv_left)
  shortform -- roi_name minus the hemisphere suffix (V1, AUD, IFG.cv)

Each panel is rendered on its own mesh via the single-view multi-view path (so we
control the panel pixel size and get a tight autocrop), then all six panels are
composed with `compose_image_grid` -- the same helper behind `hlplot montage`.

  PNGs  -> docs/images/figure_creation/species/     (committed)
  Panel + HTML scratch -> a temp dir                (discarded)

Run with the project env:
  /home/aazarg/.conda/envs/pre_env/bin/python render_species_grid.py
"""
from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from HarrisLabPlotting import (
    load_mesh_file,
    create_brain_connectivity_plot_with_modularity,
    compose_image_grid,
    short_roi_name,
)

HERE = Path(__file__).resolve().parent
TF = HERE.parent                                   # test_files/tutorial_files
PARC = TF / "parcellation and meshes"
REPO = HERE.parents[2]
from _figpaths import FIG_DPI, fig_root
IMG = fig_root(REPO) / "species"
IMG.mkdir(parents=True, exist_ok=True)

EXAMPLE_EDGE = TF / "node_edge_28" / "connectivity_28.edge"

IMAGE_DPI = FIG_DPI       # 150 committed / 600 publication (see _figpaths.py)
PANEL = (500, 500)        # base panel px; final panel px = PANEL * 8
NODE_SIZE = 10
EDGE_WIDTH = 2.0          # fixed -> minimal look, no width key
LABEL_FONT = 9

# Per-(species, view) camera zoom. Tuned per cell: a view whose brain silhouette
# fills more of the panel needs a smaller zoom, otherwise it reads as "too zoomed
# in" next to the others (the rat inferior in particular).
ZOOM = {
    ("Human", "left"): 1.0,
    ("Human", "anterior"): 1.0,
    ("Rat", "superior"): 1.2,
    ("Rat", "inferior"): 1.0,
    ("Macaque", "right"): 1.2,
    ("Macaque", "posterior"): 1.2,
}

# Column = species (mesh + coords + network). `views` are that column's two
# cells: [row-1 view, row-2 view], per the canonical BrainNet order above.
SPECIES = [
    dict(
        name="Human",
        mesh=PARC / "HCPMMP1_on_MNI152_ICBM2009a_nlin_hd_0.obj",
        coords=HERE / "human" / "hcpmmp1_coords.csv",
        matrix=HERE / "human" / "hcpmmp1_modular_network.csv",
        modules=HERE / "human" / "hcpmmp1_modules.csv",
        views=["left", "anterior"],
    ),
    dict(
        name="Rat",
        mesh=TF / "brain_mesh.gii",
        coords=TF / "output" / "atlas_28_test_comma.csv",
        matrix=EXAMPLE_EDGE,
        modules=TF / "node_edge_28" / "modules_28.csv",
        views=["superior", "inferior"],
    ),
    dict(
        name="Macaque",
        mesh=PARC / "monkey_brain_mesh_MacBNA.obj",
        coords=HERE / "monkey" / "coords_28.csv",
        matrix=EXAMPLE_EDGE,
        modules=None,                       # synthesized below (4 modules)
        views=["right", "posterior"],
    ),
]

# Per-cell labels + column headers, in the composed (row-major) order.
PANEL_LABELS = ["Left", "Superior", "Right", "Anterior", "Inferior", "Posterior"]
COL_LABELS = [s["name"] for s in SPECIES]

# Versions: (tag, label mode)
VERSIONS = [("nolabels", "none"), ("labeled", "full"), ("shortform", "short")]

def _coords_for(spec, label_mode):
    """Load a species' coords, shortening roi_name when label_mode == 'short'."""
    c = pd.read_csv(spec["coords"])
    if label_mode == "short":
        c = c.copy()
        c["roi_name"] = c["roi_name"].map(short_roi_name)
    return c


def _render_panel(spec, view, modules, coords, label_mode, mesh, tmp):
    """Render one (species, view) panel to a tight autocropped PNG, return path."""
    out = tmp / f"{spec['name']}_{view}_{label_mode}.png"
    vertices, faces = mesh
    create_brain_connectivity_plot_with_modularity(
        vertices=vertices, faces=faces, roi_coords_df=coords,
        connectivity_matrix=str(spec["matrix"]),
        module_assignments=modules,
        node_size=NODE_SIZE, edge_width=EDGE_WIDTH,   # minimal, uniform
        show_node_labels=(label_mode != "none"), label_font_size=LABEL_FONT,
        show_width_legend=False,
        plot_title="",
        multi_view=[view],                       # single view -> panel_size control + autocrop
        multi_view_panel_size=PANEL,
        multi_view_keep_first_legend=False,      # clean: no legend on the panel
        multi_view_panel_labels=[""],            # compose adds the labels
        image_dpi=IMAGE_DPI, zoom=ZOOM[(spec["name"], view)],
        save_path=str(tmp / "dummy.html"),
        export_image=str(out),
    )
    return out


def main():
    tmp = Path(tempfile.mkdtemp(prefix="species_grid_"))
    print("Loading meshes ...")
    meshes = {s["name"]: load_mesh_file(str(s["mesh"])) for s in SPECIES}

    for tag, label_mode in VERSIONS:
        print(f"Rendering species panels ({tag}) ...")
        by_cell = {}
        for si, spec in enumerate(SPECIES):
            coords = _coords_for(spec, label_mode)
            modules = (spec["modules"] if spec["modules"] is not None
                       else (np.arange(len(coords)) % 4) + 1)
            if isinstance(modules, Path):
                modules = str(modules)
            for row, view in enumerate(spec["views"]):
                by_cell[(row, si)] = _render_panel(
                    spec, view, modules, coords, label_mode,
                    meshes[spec["name"]], tmp)
                print(f"  [{tag}] {spec['name']} / {view}")

        # Assemble row-major: row0 across species, then row1 across species.
        images = [by_cell[(row, si)] for row in (0, 1) for si in range(len(SPECIES))]
        out = IMG / f"species_grid_{tag}.png"
        compose_image_grid(
            images, out,
            grid=(2, 3),
            col_labels=COL_LABELS,
            panel_labels=PANEL_LABELS,
            background_color="white",
        )
        sz = out.stat().st_size / 1e6
        print(f"  wrote {out.name}  ({sz:.1f} MB)")

    print(f"\nPNGs -> {IMG}")


if __name__ == "__main__":
    main()
