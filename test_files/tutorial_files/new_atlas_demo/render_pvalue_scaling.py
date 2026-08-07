"""
Render the p-value edge/node significance-scaling demo for the docs.

Two renders of the SAME 28-ROI p-value network on the rat tutorial brain
(`brain_mesh.gii`), to show how significance can be encoded by edge width AND
node size:

  (a) uniform   -- fixed edge width + scalar node size. Significance is NOT
                   encoded; every edge/node looks the same. Baseline.
  (b) scaled    -- edge width scales with -log10(p) (thicker = more significant),
                   and node size scales with a per-node significance derived from
                   the p-value matrix (bigger = a node with stronger / more
                   significant incident edges).

The per-node significance and the size/width ranges are exposed as variables at
the top so they are easy to tweak (mirrored in the notebook).

  PNGs -> docs/images/figure_creation/pvalue/     (committed)

Run with the project env:
  /home/aazarg/.conda/envs/pre_env/bin/python render_pvalue_scaling.py
"""
from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from HarrisLabPlotting import (
    load_mesh_file,
    create_brain_connectivity_plot,
    short_roi_name,
)

HERE = Path(__file__).resolve().parent
TF = HERE.parent                                   # test_files/tutorial_files
REPO = HERE.parents[2]
from _figpaths import FIG_DPI, fig_root
IMG = fig_root(REPO) / "pvalue"
IMG.mkdir(parents=True, exist_ok=True)
HTML = Path(tempfile.mkdtemp(prefix="pvalue_scaling_"))

MESH = TF / "brain_mesh.gii"
COORDS = TF / "output" / "atlas_28_test_comma.csv"
PVALUES = TF / "node_edge_28" / "pvalues_28_spread.csv"   # log-spread significance
# Direction of each effect (+1/-1/0). Same edge topology as the p-value matrix,
# so positive edges render red and negative (opposite-direction) edges render
# blue. Of the 20 significant edges, 4 are negative.
SIGN_MATRIX = TF / "node_edge_28" / "pvalues_28_signs.csv"

# ----- Tweakable parameters (the notebook exposes the same knobs) -----------
PVALUE_THRESHOLD = 0.05      # edges with p > threshold are dropped
EDGE_WIDTH_MIN = 1.0         # thinnest significant edge (scaled render)
EDGE_WIDTH_MAX = 9.0         # thickest (most significant) edge
SIZE_MIN = 6.0               # smallest node (least significant)
SIZE_MAX = 24.0              # largest node (most significant)
NODE_SIZE_SCALE = 1.0        # uniform multiplier on the derived node sizes
EDGE_WIDTH_SCALE = 2         # uniform multiplier on every edge width
SHOW_NODE_LABELS = True      # show ROI names
SHORT_LABELS = True          # shorten the hemisphere suffix to cut label overlap
KEEP_HEMISPHERE = True       # ...but keep it as _L/_R: superior/inferior show BOTH
                             # hemispheres, so dropping it entirely would give the
                             # left and right node of a pair the SAME label.
LABEL_FONT_SIZE = 7          # small enough for 28 labels
IMAGE_DPI = FIG_DPI       # 150 committed / 600 publication (see _figpaths.py)
CAMERA = "superior"          # camera for the single-view exports
MULTI_VIEW = ["left", "superior", "posterior"]   # panels for the multi-view strip
MULTI_VIEW_PANEL = (700, 700)                    # base px per panel


def node_significance(pvalue_csv, threshold):
    """Per-node significance = sum of -log10(p) over that node's surviving edges.

    A node with many strong (small-p) connections gets a large value; an isolated
    or weakly-connected node gets a small one. Returns the raw per-node vector.
    """
    P = np.loadtxt(pvalue_csv, delimiter=",")        # 28x28, no header
    valid = (P > 0) & (P <= threshold)
    W = np.where(valid, -np.log10(np.clip(P, 1e-300, 1.0)), 0.0)
    np.fill_diagonal(W, 0.0)
    return W.sum(axis=1)


def sig_to_pixels(sig, size_min, size_max, scale):
    """Min-max map a per-node significance vector into [size_min, size_max] px."""
    lo, hi = float(sig.min()), float(sig.max())
    if hi > lo:
        px = size_min + (sig - lo) / (hi - lo) * (size_max - size_min)
    else:
        px = np.full_like(sig, size_min)
    return px * float(scale)


def main():
    print("Loading mesh + coords ...")
    vertices, faces = load_mesh_file(str(MESH))
    coords = pd.read_csv(COORDS)
    if SHORT_LABELS:
        coords = coords.copy()
        coords["roi_name"] = coords["roi_name"].map(
            lambda n: short_roi_name(n, keep_hemisphere=KEEP_HEMISPHERE))

    common = dict(
        vertices=vertices, faces=faces, roi_coords_df=coords,
        connectivity_matrix=str(PVALUES),
        matrix_type="pvalue", pvalue_threshold=PVALUE_THRESHOLD,
        sign_matrix=str(SIGN_MATRIX),   # -> red = positive, blue = negative direction
        image_dpi=IMAGE_DPI,
        edge_width_scale=EDGE_WIDTH_SCALE,
        show_node_labels=SHOW_NODE_LABELS,
        label_font_size=LABEL_FONT_SIZE,
    )

    sig = node_significance(PVALUES, PVALUE_THRESHOLD)
    node_px = sig_to_pixels(sig, SIZE_MIN, SIZE_MAX, NODE_SIZE_SCALE)
    metrics = pd.DataFrame({
        "roi_name": coords["roi_name"],
        "node_significance": sig,             # -> label the size key with this
    })

    # The two variants: (a) nothing encodes significance, (b) edges AND nodes do.
    variants = {
        "pval_uniform": dict(
            edge_width=2.0,      # fixed  -> width key auto-skipped
            node_size=8,         # scalar -> size key auto-skipped
            plot_title="p-values, uniform (no significance scaling)",
        ),
        "pval_scaled": dict(
            edge_width=(EDGE_WIDTH_MIN, EDGE_WIDTH_MAX),  # scaled by -log10(p)
            node_size=node_px,                            # per-node significance
            node_metrics=metrics,
            node_size_legend_metric="node_significance",  # key shows significance
            plot_title="p-values, significance-scaled edges + nodes",
        ),
    }

    written = []
    for key, kw in variants.items():
        # Single view.
        create_brain_connectivity_plot(
            **common, **kw, camera_view=CAMERA,
            save_path=str(HTML / f"{key}.html"),
            export_image=str(IMG / f"{key}.png"),
        )
        written.append(f"{key}.png")
        # Multi-view strip. The keys stay on the first panel
        # (multi_view_keep_first_legend defaults to True).
        create_brain_connectivity_plot(
            **common, **kw,
            multi_view=MULTI_VIEW,
            multi_view_panel_size=MULTI_VIEW_PANEL,
            save_path=str(HTML / f"{key}_mv.html"),
            export_image=str(IMG / f"{key}_multiview.png"),
        )
        written.append(f"{key}_multiview.png")
        print(f"  wrote {key}.png + {key}_multiview.png")

    for name in written:
        mb = (IMG / name).stat().st_size / 1e6
        print(f"    {name}: {mb:.1f} MB")
    print(f"\nPNGs -> {IMG}")


if __name__ == "__main__":
    main()
