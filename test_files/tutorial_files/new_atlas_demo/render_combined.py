"""
Render the combined voxel + network figures.

Shows the two features composed: a voxel cloud from the mouse Fig 1 z-maps with
a directed ROI network drawn on top, on the same Allen surface.

The NETWORK IS SYNTHETIC. Its ROI coordinates are real -- centres of gravity of
the 146-region atlas, in the same Allen world space -- but the connections are
fabricated with a fixed seed, purely to demonstrate the composition.

Outputs -> docs/images/combined/

Run with the project env:
  /home/aazarg/.conda/envs/pre_env/bin/python render_combined.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

from HarrisLabPlotting import (
    coordinate_function, create_brain_connectivity_plot, load_mesh_file,
)

HERE = Path(__file__).resolve().parent
TF = HERE.parent
REPO = HERE.parents[2]
sys.path.insert(0, str(HERE))
from _figpaths import FIG_DPI, fig_root  # noqa: E402

IMG = Path(str(fig_root(REPO)).replace("figure_creation", "combined"))
IMG.mkdir(parents=True, exist_ok=True)
TMP = Path("/tmp/hlplot_combined")
TMP.mkdir(exist_ok=True)

MOUSE = TF / "mouse"
MESH = MOUSE / "bin_dilD_Parc_Atlas_0.obj"
ATLAS = MOUSE / "ROI_Selected_Atlas_fill_dilD.nii.gz"
POS = MOUSE / "Fig1_RM_Sham_pos_z_allen.nii.gz"
NEG = MOUSE / "Fig1_RM_Sham_neg_z_allen.nii.gz"

N_NODES = 18          # a readable subset of the 146 regions
N_EDGES = 22
SEED = 7
STEP = 7
THR = 3.1
ANISO = "0.54,0.11,0.11"


def roi_coords():
    """Real ROI centres of gravity from the 146-label Allen atlas.

    Cached next to the script so the figure does not re-scan a 5.9 MB label
    volume on every run.
    """
    cache = HERE / "mouse" / "allen_roi_coords.csv"
    if cache.exists():
        return pd.read_csv(cache)
    cache.parent.mkdir(parents=True, exist_ok=True)

    import nibabel as nib
    from scipy.ndimage import center_of_mass
    img = nib.load(str(ATLAS))
    data = np.rint(np.asarray(img.dataobj)).astype(int)
    labels = [int(v) for v in np.unique(data) if v != 0]
    rows = []
    for lab in labels:
        com = center_of_mass(data == lab)
        world = nib.affines.apply_affine(img.affine, np.asarray(com))
        rows.append(dict(roi_name=f"ROI_{lab:03d}", cog_x=world[0],
                         cog_y=world[1], cog_z=world[2]))
    df = pd.DataFrame(rows)
    df.to_csv(cache, index=False)
    print(f"  cached {len(df)} ROI coordinates -> {cache}")
    return df


def synthetic_network(coords):
    """A small directed network with fixed seed. Synthetic, for display only."""
    rng = np.random.default_rng(SEED)
    n = len(coords)
    M = np.zeros((n, n))
    xyz = coords[["cog_x", "cog_y", "cog_z"]].to_numpy(float)
    d = np.linalg.norm(xyz[:, None, :] - xyz[None, :, :], axis=2)
    cand = [(i, j) for i in range(n) for j in range(n)
            if i != j and d[i, j] > 1.0]
    picks = rng.choice(len(cand), size=min(N_EDGES, len(cand)), replace=False)
    for k in picks:
        i, j = cand[k]
        M[i, j] = round(float(rng.uniform(0.3, 1.0)), 3)
        if rng.random() < 0.35:                       # some reciprocal pairs
            M[j, i] = round(M[i, j] * float(rng.uniform(0.3, 0.7)), 3)
    return M


def main() -> None:
    if not MESH.exists() or not ATLAS.exists():
        print(f"mouse data not found under {MOUSE}; skipping", file=sys.stderr)
        return
    print(f"combined figures -> {IMG}  ({FIG_DPI} DPI)")

    v, f = load_mesh_file(str(MESH))
    v, f = np.asarray(v, float), np.asarray(f, int)

    allen = roi_coords()
    rng = np.random.default_rng(SEED)
    keep = np.sort(rng.choice(len(allen), size=min(N_NODES, len(allen)),
                              replace=False))
    coords = allen.iloc[keep].reset_index(drop=True)
    M = synthetic_network(coords)

    overlays = [
        dict(path=str(POS), name="Activation", cmap="hot32", threshold=THR,
             smooth_fwhm=ANISO, step=STEP),
        dict(path=str(NEG), name="Deactivation", cmap="ice28", threshold=THR,
             smooth_fwhm=ANISO, step=STEP),
    ]

    common = dict(
        vertices=v, faces=f, roi_coords_df=coords, connectivity_matrix=M,
        edge_width=(2.0, 9.0), node_size=13, node_color="#2b2b3a",
        node_border_color="white", show_node_labels=False,
        mesh_opacity=0.05, plot_title="", no_html=True, image_dpi=FIG_DPI,
        zoom=1.25, save_path=str(TMP / "d.html"),
    )

    # single superior view
    create_brain_connectivity_plot(
        **common, volume_overlays=overlays, camera_view="superior",
        export_size=(900, 900),
        export_image=str(IMG / "01_combined_superior.png"))
    print("  01_combined_superior.png")

    # three views
    create_brain_connectivity_plot(
        **common, volume_overlays=overlays,
        multi_view=["left", "superior", "posterior"],
        multi_view_panel_size=(700, 700),
        export_image=str(IMG / "02_combined_multiview.png"))
    print("  02_combined_multiview.png")

    # network alone, for the side-by-side in the tutorial
    create_brain_connectivity_plot(
        **common, camera_view="superior", export_size=(900, 900),
        export_image=str(IMG / "03_network_only.png"))
    print("  03_network_only.png")

    print("done")


if __name__ == "__main__":
    main()
