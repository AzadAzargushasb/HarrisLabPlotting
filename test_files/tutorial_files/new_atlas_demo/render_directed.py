"""
Render the directed-graph figures for the docs.

Everything runs on the shipped 28-ROI fixtures plus `directed_28.csv`
(built by generate_directed_demo.py), so it reproduces from a clean clone.

Outputs -> docs/images/directed/   (committed at 150 DPI; a 600 DPI copy goes
to the git-ignored publication/ tree via HLP_FIG_DPI / HLP_FIG_PUB)

Run with the project env:
  /home/aazarg/.conda/envs/pre_env/bin/python render_directed.py
"""
from __future__ import annotations

import io
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

from HarrisLabPlotting import (
    create_brain_connectivity_plot,
    create_brain_connectivity_plot_with_modularity,
    load_mesh_file,
    short_roi_name,
)

HERE = Path(__file__).resolve().parent
TF = HERE.parent
REPO = HERE.parents[2]
sys.path.insert(0, str(HERE))
from _figpaths import FIG_DPI, fig_root  # noqa: E402

IMG = Path(str(fig_root(REPO)).replace("figure_creation", "directed"))
IMG.mkdir(parents=True, exist_ok=True)
TMP = Path("/tmp/hlplot_directed_render")
TMP.mkdir(exist_ok=True)

MESH = TF / "brain_mesh.gii"
COORDS = TF / "output" / "atlas_28_test_comma.csv"
MATRIX = TF / "node_edge_28" / "directed_28.csv"
SYMMETRIC = TF / "node_edge_28" / "connectivity_28.edge"
MODULES = TF / "node_edge_28" / "modules_28.csv"

EDGE_WIDTH = (1.0, 9.0)
ZOOM = 1.3
PANEL = (700, 700)


def _font(size):
    for cand in ("DejaVuSans-Bold.ttf", "DejaVuSans.ttf", "Arial.ttf"):
        try:
            return ImageFont.truetype(cand, int(size))
        except (OSError, IOError):
            continue
    return ImageFont.load_default()


def base(**kw):
    """Common arguments; kw overrides."""
    v, f = load_mesh_file(str(MESH))
    coords = pd.read_csv(COORDS)
    args = dict(
        vertices=np.asarray(v, float), faces=np.asarray(f, int),
        roi_coords_df=coords, connectivity_matrix=str(MATRIX),
        edge_width=EDGE_WIDTH, zoom=ZOOM, image_dpi=FIG_DPI,
        show_node_labels=False, plot_title="", no_html=True,
        save_path=str(TMP / "dummy.html"),
    )
    args.update(kw)
    return args


def contact_sheet(panels, labels, out_name, title, cols=None):
    """A labelled grid -- these are the parameter-reference sheets, which are
    the one place a baked-in title earns its keep."""
    cols = cols or len(panels)
    rows = (len(panels) + cols - 1) // cols
    ims = [Image.open(p).convert("RGB") for p in panels]
    cw = max(i.size[0] for i in ims)
    ch = max(i.size[1] for i in ims)
    pad = 14
    W = cols * cw + pad * (cols + 1)
    probe = Image.new("RGB", (10, 10))
    pd_ = ImageDraw.Draw(probe)

    def fit(text, max_w, start, floor=14):
        """Largest font size at which `text` fits in `max_w`."""
        sz = int(start)
        while sz > floor and pd_.textbbox((0, 0), text, font=_font(sz))[2] > max_w:
            sz -= 1
        return sz

    t_sz = fit(title, W - 2 * pad, max(30, cw // 14))
    l_sz = min(fit(lab, cw - 8, max(20, cw // 20)) for lab in labels)
    th, lh = int(t_sz * 2.0), int(l_sz * 2.0)
    H = th + rows * (ch + lh)
    sheet = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(sheet)
    tw = d.textbbox((0, 0), title, font=_font(t_sz))[2]
    d.text(((W - tw) // 2, int(t_sz * 0.45)), title, fill="black",
           font=_font(t_sz))
    for k, (im, lab) in enumerate(zip(ims, labels)):
        r, c = divmod(k, cols)
        x = pad + c * (cw + pad)
        y = th + r * (ch + lh)
        sheet.paste(im, (x + (cw - im.size[0]) // 2, y))
        lw = d.textbbox((0, 0), lab, font=_font(l_sz))[2]
        d.text((x + (cw - lw) // 2, y + ch + int(l_sz * 0.4)), lab,
               fill=(40, 40, 40), font=_font(l_sz))
    sheet.save(IMG / out_name)
    print(f"  {out_name}")


def render(tag, **kw):
    out = TMP / f"{tag}.png"
    create_brain_connectivity_plot(**base(export_image=str(out), **kw))
    return out


def main() -> None:
    print(f"directed figures -> {IMG}  ({FIG_DPI} DPI)")

    # --- fig 1: quick start, single superior view ---------------------
    create_brain_connectivity_plot(**base(
        camera_view="superior", show_node_labels=True, label_font_size=9,
        export_image=str(IMG / "01_quickstart_superior.png")))
    print("  01_quickstart_superior.png")

    # --- fig 2: the same network as a 3-panel multi-view --------------
    create_brain_connectivity_plot(**base(
        multi_view=["left", "superior", "posterior"],
        multi_view_panel_size=PANEL, show_node_labels=True, label_font_size=9,
        export_image=str(IMG / "02_multiview.png")))
    print("  02_multiview.png")

    # --- fig 3: orientation -- the same file read both ways -----------
    a = render("orient_row", camera_view="superior",
               matrix_orientation="row-to-col")
    b = render("orient_col", camera_view="superior",
               matrix_orientation="col-to-row")
    contact_sheet(
        [a, b],
        ["--matrix-orientation row-to-col  (default: row = source)",
         "--matrix-orientation col-to-row  (DCM: column = source)"],
        "03_orientation.png",
        "The SAME file, read both ways -- every arrow reverses")

    # --- fig 4: arrow size -------------------------------------------
    sizes = [1.0, 1.08, 1.4, 2.0]
    contact_sheet([render(f"k{k}", camera_view="superior", arrow_size=k)
                   for k in sizes],
                  [f"--arrow-size {k}" for k in sizes],
                  "ref_arrow_size.png",
                  "--arrow-size : cone radius as a multiple of its own line's "
                  "half-width", cols=2)

    # --- fig 5: slenderness ------------------------------------------
    slen = [0.12, 0.18, 0.25, 0.35]
    contact_sheet([render(f"s{s}", camera_view="superior",
                          arrow_slenderness=s) for s in slen],
                  [f"--arrow-slenderness {s}" for s in slen],
                  "ref_arrow_slenderness.png",
                  "--arrow-slenderness : radius / length. Lower = a longer, "
                  "thinner dart", cols=2)

    # --- fig 6: the short-edge cap -----------------------------------
    caps = [0.15, 0.30, 0.60, 1.00]
    contact_sheet([render(f"c{c}", camera_view="superior",
                          arrow_max_edge_frac=c) for c in caps],
                  [f"--arrow-max-edge-frac {c}" for c in caps],
                  "ref_arrow_cap.png",
                  "--arrow-max-edge-frac : cap on head length as a fraction of "
                  "the edge (watch the shortest pairs)", cols=2)

    # --- fig 7: the arc bow floor ------------------------------------
    bows = [0.0, 0.015, 0.03, 0.06]
    contact_sheet([render(f"b{b}", camera_view="superior", arc_bow_floor=b)
                   for b in bows],
                  [f"--arc-bow-floor {b}" for b in bows],
                  "ref_arc_bow_floor.png",
                  "--arc-bow-floor : minimum separation of a reciprocal pair, "
                  "as a fraction of the bbox diagonal", cols=2)

    # --- fig 8: view mode --------------------------------------------
    contact_sheet(
        [render("vm_camera", camera_view="superior", arrow_view_mode="camera"),
         render("vm_fixed", camera_view="superior", arrow_view_mode="fixed")],
        ["--arrow-view-mode camera  (arcs bow toward the view plane)",
         "--arrow-view-mode fixed   (anatomically fixed, same from every angle)"],
        "ref_arrow_view_mode.png",
        "--arrow-view-mode : how reciprocal pairs bow apart")

    # --- fig 9: symmetric vs directed, side by side -------------------
    v, f = load_mesh_file(str(MESH))
    coords = pd.read_csv(COORDS)
    sym_png = TMP / "symmetric.png"
    create_brain_connectivity_plot(
        vertices=np.asarray(v, float), faces=np.asarray(f, int),
        roi_coords_df=coords, connectivity_matrix=str(SYMMETRIC),
        edge_width=EDGE_WIDTH, zoom=ZOOM, camera_view="superior",
        show_node_labels=False, plot_title="", no_html=True,
        image_dpi=FIG_DPI, save_path=str(TMP / "d.html"),
        export_image=str(sym_png))
    contact_sheet(
        [sym_png, render("dir_only", camera_view="superior")],
        ["symmetric matrix -> undirected (unchanged behaviour)",
         "asymmetric matrix -> arrows, automatically"],
        "04_symmetric_vs_directed.png",
        "Arrows appear only when the matrix is actually asymmetric")

    # --- fig 10: modular, directed ------------------------------------
    if MODULES.exists():
        create_brain_connectivity_plot_with_modularity(
            vertices=np.asarray(v, float), faces=np.asarray(f, int),
            roi_coords_df=coords, connectivity_matrix=str(MATRIX),
            module_assignments=str(MODULES), edge_width=EDGE_WIDTH,
            zoom=ZOOM, multi_view=["left", "superior", "posterior"],
            multi_view_panel_size=PANEL, show_node_labels=False,
            plot_title="", no_html=True, image_dpi=FIG_DPI,
            save_path=str(TMP / "m.html"),
            export_image=str(IMG / "05_modular_directed.png"))
        print("  05_modular_directed.png")

    print("done")


if __name__ == "__main__":
    main()
