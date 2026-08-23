"""
Render the voxel-plotting figures for the docs.

Uses the mouse Fig 1 data shipped under test_files/tutorial_files/mouse/ --
Allen 25 um z-maps and the Allen surface, which are already in the same world
space, so nothing here is registered.

Outputs -> docs/images/voxel/   (committed at 150 DPI; a 600 DPI copy goes to
the git-ignored publication/ tree via HLP_FIG_DPI / HLP_FIG_PUB)

The sweeps are deliberately run at a coarse --volume-step: with a 0.54 mm
smoothing kernel a step of 7 is visually identical to full resolution and about
40x cheaper. The performance appendix in the tutorial shows the numbers.

Run with the project env:
  /home/aazarg/.conda/envs/pre_env/bin/python render_voxels.py
"""
from __future__ import annotations

import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from HarrisLabPlotting import create_brain_volume_plot

HERE = Path(__file__).resolve().parent
TF = HERE.parent
REPO = HERE.parents[2]
sys.path.insert(0, str(HERE))
from _figpaths import FIG_DPI, fig_root  # noqa: E402

IMG = Path(str(fig_root(REPO)).replace("figure_creation", "voxel"))
IMG.mkdir(parents=True, exist_ok=True)
TMP = Path("/tmp/hlplot_voxel_render")
TMP.mkdir(exist_ok=True)

MOUSE = TF / "mouse"
MESH = MOUSE / "bin_dilD_Parc_Atlas_0.obj"
POS = MOUSE / "Fig1_RM_Sham_pos_z_allen.nii.gz"
NEG = MOUSE / "Fig1_RM_Sham_neg_z_allen.nii.gz"
SIGNED = MOUSE / "Fig1_RM_Sham_combined_signed_z_allen.nii.gz"
POS10 = MOUSE / "Fig1_RM_Sham_pos_z_top10_allen.nii.gz"

THR = 3.1
ANISO = "0.54,0.11,0.11"     # the ORIGINAL MDT voxel size
STEP = 7
ZOOM = 1.25
PANEL = (700, 700)
VIEWS = ["left", "superior", "posterior"]


def _font(size):
    for cand in ("DejaVuSans-Bold.ttf", "DejaVuSans.ttf", "Arial.ttf"):
        try:
            return ImageFont.truetype(cand, int(size))
        except (OSError, IOError):
            continue
    return ImageFont.load_default()


def sheet(panels, labels, out_name, title, cols=None, bg="white"):
    cols = cols or len(panels)
    rows = (len(panels) + cols - 1) // cols
    ims = [Image.open(p).convert("RGB") for p in panels]
    cw = max(i.size[0] for i in ims)
    ch = max(i.size[1] for i in ims)
    pad = 14
    W = cols * cw + pad * (cols + 1)
    probe = ImageDraw.Draw(Image.new("RGB", (10, 10)))

    def fit(text, max_w, start, floor=14):
        sz = int(start)
        while sz > floor and probe.textbbox((0, 0), text, font=_font(sz))[2] > max_w:
            sz -= 1
        return sz

    t_sz = fit(title, W - 2 * pad, max(30, cw // 14))
    l_sz = min(fit(l, cw - 8, max(20, cw // 20)) for l in labels)
    th, lh = int(t_sz * 2.0), int(l_sz * 2.0)
    fg = "black" if bg == "white" else "#e8e8ee"
    out = Image.new("RGB", (W, th + rows * (ch + lh)), bg)
    d = ImageDraw.Draw(out)
    tw = d.textbbox((0, 0), title, font=_font(t_sz))[2]
    d.text(((W - tw) // 2, int(t_sz * 0.45)), title, fill=fg, font=_font(t_sz))
    for k, (im, lab) in enumerate(zip(ims, labels)):
        r, c = divmod(k, cols)
        x, y = pad + c * (cw + pad), th + r * (ch + lh)
        out.paste(im, (x + (cw - im.size[0]) // 2, y))
        lw = d.textbbox((0, 0), lab, font=_font(l_sz))[2]
        d.text((x + (cw - lw) // 2, y + ch + int(l_sz * 0.4)), lab, fill=fg,
               font=_font(l_sz))
    out.save(IMG / out_name)
    print(f"  {out_name}")


def render(tag, volumes, *, out=None, view="posterior", multi=False,
           background="white", quiet=True, **kw):
    path = Path(out) if out else TMP / f"{tag}.png"
    create_brain_volume_plot(
        mesh=str(MESH), volumes=volumes,
        camera_view=view, zoom=ZOOM, background_color=background,
        no_html=True, quiet=quiet, check_space=False,
        image_dpi=FIG_DPI, export_size=(900, 900),
        multi_view=VIEWS if multi else None,
        multi_view_panel_size=PANEL,
        export_image=str(path), save_path=str(TMP / "d.html"), **kw)
    return path


def posneg(**over):
    """The recommended two-file setup: hot32 activation + ice28 deactivation."""
    base_pos = dict(path=str(POS), name="Activation", cmap="hot32",
                    threshold=THR, smooth_fwhm=ANISO, step=STEP)
    base_neg = dict(path=str(NEG), name="Deactivation", cmap="ice28",
                    threshold=THR, smooth_fwhm=ANISO, step=STEP)
    base_pos.update(over)
    base_neg.update(over)
    return [base_pos, base_neg]


def main() -> None:
    if not MESH.exists():
        print(f"mouse data not found under {MOUSE}; skipping", file=sys.stderr)
        return
    print(f"voxel figures -> {IMG}  ({FIG_DPI} DPI)")

    # --- 01 quick start: single superior view -------------------------
    render("qs", posneg(), out=IMG / "01_quickstart_superior.png",
           view="superior", quiet=False)
    print("  01_quickstart_superior.png")

    # --- 02 the same, as a 3-panel multi-view -------------------------
    render("mv", posneg(), out=IMG / "02_multiview.png", multi=True)
    print("  02_multiview.png")

    # --- 03 smoothing: the 16-slice creases, before and after ---------
    sheet([render("sm_none", posneg(smooth_fwhm=None, level="fixed"),
                  view="superior"),
           render("sm_aniso", posneg(), view="superior")],
          ["no smoothing -- the 16 coronal slices show as steps",
           f"--volume-smooth-fwhm {ANISO} (the source voxel size)"],
          "03_smoothing.png",
          "Smoothing removes the creases left by resampling 16 thick slices "
          "onto a 25 um grid")

    # --- 04 fixed vs volume-preserving level --------------------------
    sheet([render("lv_fixed", posneg(smooth_fwhm="0.54,0.54,0.54",
                                     level="fixed"), view="superior"),
           render("lv_pres", posneg(smooth_fwhm="0.54,0.54,0.54",
                                    level="preserve"), view="superior")],
          ["--volume-level fixed -- the blur eats 40% of the cluster",
           "--volume-level preserve (default) -- same volume as unsmoothed"],
          "04_level_correction.png",
          "Blurring lowers the peak, so a FIXED level pulls the boundary "
          "inward")

    # --- 05 black vs white background ---------------------------------
    sheet([render("bg_black", posneg(), view="superior", background="black"),
           render("bg_white", posneg(), view="superior", background="white")],
          ["--background-color black (hot32 runs to white-hot)",
           "--background-color white (top truncated so the peak stays visible)"],
          "05_background.png",
          "The colorscale adapts: on white the top of hot32 is truncated",
          bg="black")

    # --- 06 two files vs one signed file ------------------------------
    sheet([render("two", posneg(), view="superior"),
           render("one", [dict(path=str(SIGNED), name="signed z",
                               threshold=THR, smooth_fwhm=ANISO, step=STEP)],
                  view="superior")],
          ["two files: hot32 + ice28, independently toggleable",
           "one signed file: a single diverging scale"],
          "06_two_files_vs_signed.png",
          "Two-sided data: two files (recommended) or one signed file")

    # --- 07 full vs top-10% ------------------------------------------
    sheet([render("full", [dict(path=str(POS), name="Activation",
                                cmap="hot32", threshold=THR,
                                smooth_fwhm=ANISO, step=STEP)],
                  view="superior"),
           render("t10", [dict(path=str(POS), name="Activation (top 10%)",
                               cmap="hot32", top_percent=10,
                               smooth_fwhm=ANISO, step=STEP)],
                  view="superior")],
          ["--volume-threshold 3.1 (the whole suprathreshold map)",
           "--volume-top-percent 10 (only the strongest decile)"],
          "07_thresholds.png",
          "Threshold modes: an absolute value, or the strongest N%")

    # --- reference sheets: the look knobs -----------------------------
    for knob, values, flag, blurb in (
        ("opacity", [0.25, 0.5, 0.75, 1.0], "--volume-opacity",
         "opacity CEILING of the VOXEL MAP at its peak (not the brain)"),
        ("gamma", [0.4, 0.7, 1.0, 1.6], "--volume-gamma",
         "shape of the opacity ramp: low lights the whole cluster, high only "
         "the core"),
        ("surfaces", [25, 60, 100, 200], "--volume-surfaces",
         "shells the ray-cast steps through: more is smoother AND slower"),
        ("floor", [0.0, 0.1, 0.15, 0.3], "--volume-opacity-floor",
         "opacity AT the threshold: 0 makes the cluster fringe vanish"),
    ):
        key = {"opacity": "opacity", "gamma": "gamma",
               "surfaces": "surfaces", "floor": "opacity_floor"}[knob]
        panels = [render(f"{knob}{v}", posneg(**{key: v}), view="superior")
                  for v in values]
        sheet(panels, [f"{flag} {v}" for v in values],
              f"ref_{knob}.png", f"{flag} : {blurb}", cols=2)

    # --- ghost (the BRAIN's opacity) ----------------------------------
    ghosts = [0.0, 0.04, 0.10, 0.20]
    sheet([render(f"gh{g}", posneg(), view="superior", ghost_opacity=g)
           for g in ghosts],
          [f"--ghost-opacity {g}" for g in ghosts],
          "ref_ghost.png",
          "--ghost-opacity : the BRAIN shell's opacity (not the voxel map's)",
          cols=2)

    # --- views --------------------------------------------------------
    views = ["left", "right", "superior", "inferior", "anterior", "posterior"]
    sheet([render(f"vw_{v}", posneg(), view=v) for v in views],
          views, "ref_views.png", "Camera presets", cols=3)

    print("done")


if __name__ == "__main__":
    main()
