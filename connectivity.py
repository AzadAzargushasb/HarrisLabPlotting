"""
Brain Connectivity Visualization
================================
Brain connectivity visualization functions with camera controls.
Version 3 features: Camera controls, toggleable edges, scaled edge widths,
                   vector node sizes, node metrics hover, edge-linked node hiding.
"""

import pandas as pd
import numpy as np
import networkx as nx
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
import colorsys
from typing import Dict, Tuple, Optional, Union, List

# Import shared modules from package
from .mesh import load_mesh_file
from .camera import CameraController
from .utils import (
    convert_node_size_input,
    convert_node_color_input,
    load_connectivity_input,
    load_node_metrics,
    calculate_edge_width,
    generate_module_colors,
    load_edge_color_matrix,
    transform_pvalue_matrix,
)


# ----------------------------------------------------------------------
# Mesh lighting presets
# ----------------------------------------------------------------------
# Each preset is a dict that maps directly onto plotly's Mesh3d.lighting
# parameters. They are intentionally hand-tuned so that the named style
# matches what most users would expect when they hear the word.
#
# Plotly defaults (for reference): ambient=0.8, diffuse=0.8, specular=0.05,
#                                  roughness=0.5, fresnel=0.2.
#
# 'default' is intentionally NOT in this dict -- "default" means "do not
# touch lighting at all", which preserves whatever the existing code path
# was doing (None unless fast_render, in which case ambient=0.8).
MESH_LIGHTING_PRESETS: Dict[str, Dict[str, float]] = {
    'flat': dict(
        # Pure ambient, no shading at all -- mesh looks like a paper cutout.
        ambient=1.0, diffuse=0.0, specular=0.0, roughness=1.0, fresnel=0.0,
    ),
    'matte': dict(
        # Diffuse-only, no specular highlight at all. Looks like chalk.
        ambient=0.5, diffuse=0.9, specular=0.0, roughness=1.0, fresnel=0.0,
    ),
    'smooth': dict(
        # Balanced -- a tiny bit of specular and a hint of fresnel rim.
        ambient=0.4, diffuse=0.8, specular=0.3, roughness=0.7, fresnel=0.2,
    ),
    'glossy': dict(
        # Sharp specular highlight + visible rim light. The "shiny plastic"
        # look most users mean when they say "glossy".
        ambient=0.3, diffuse=0.7, specular=1.2, roughness=0.2, fresnel=0.5,
    ),
    'mirror': dict(
        # Maximum specular, near-zero roughness, strong fresnel. Looks
        # almost chrome.
        ambient=0.2, diffuse=0.5, specular=2.0, roughness=0.05, fresnel=0.8,
    ),
}


def export_multi_view_stitched_png(
    fig: go.Figure,
    output_path: Union[str, Path],
    *,
    views: List[Union[str, Dict]],
    panel_width: int = 800,
    panel_height: int = 800,
    image_dpi: int = 300,
    title: str = "",
    panel_labels: Optional[List[str]] = None,
    keep_first_legend: bool = True,
    bg_color: str = 'white',
    label_font_size: int = 18,
    title_font_size: int = 22,
    zoom: float = 1.0,
    autocrop: bool = True,
    autocrop_padding_px: int = 8,
) -> Path:
    """
    Render ``fig`` from N camera angles and stitch them into a single
    horizontal PNG strip (1 row x N columns).

    The stitched output is built by:

    1. Cloning the input figure N times (via JSON serialization, so the
       original is left untouched).
    2. Stripping each clone's title, dropdown, camera-control annotation,
       and any live camera-readout post-script.
    3. For panels 2..N (when ``keep_first_legend=True``) also stripping
       the legend so the brain occupies the full panel width and the
       three brains line up visually.
    4. Overriding ``scene.camera`` on each clone to the requested view.
       Each entry in ``views`` is either:

       - the name of a built-in preset (``'left'``, ``'superior'``, ...);
       - or a dict ``{'name': str, 'eye': dict, 'center': dict, 'up': dict}``
         describing a custom camera (the format used by
         ``create_brain_connectivity_plot``'s ``custom_camera`` parameter).

    5. Rendering each clone to a temporary PNG via kaleido.
    6. Pasting the panels into a single image with Pillow, with optional
       small text labels under each panel and a combined title above.

    Parameters
    ----------
    fig : plotly.graph_objects.Figure
        The figure to re-render. NOT mutated.
    output_path : str or Path
        Where the stitched PNG should be written.
    views : list of (str | dict)
        Names of preset camera views (or full custom-camera dicts). The
        list length determines how many panels the strip has.
    panel_width, panel_height : int
        Pixel size of each individual panel BEFORE the DPI scale factor
        is applied. Default 800 x 800.
    image_dpi : int
        DPI for the output. Acts as a scale factor on the panel/strip
        dimensions: each panel is rendered at ``scale = dpi / 72``,
        capped at 8.0 (effective ~576 DPI). At ``image_dpi=300`` the
        scale factor is ~4.17. At ``image_dpi=600`` it's ~8.0 (the cap).
    title : str
        Combined title text drawn above the strip. Empty string = no
        title bar.
    panel_labels : list of str, optional
        Per-panel labels drawn below each panel. Length must equal
        ``len(views)``. When ``None``, the function pulls a sensible
        label out of each view (preset name -> capitalized form, custom
        dict -> ``view['name']`` if present, otherwise ``"View N"``).
    keep_first_legend : bool
        When ``True`` (default) the FIRST panel keeps its plotly legend
        and the rest are rendered with ``showlegend=False``. The
        first panel will therefore look slightly different from the
        others (legend taking up some left space) but you only have to
        look at the legend once. Set to ``False`` to strip the legend
        from every panel.
    bg_color : str
        Background color of the strip canvas. Default ``'white'``.
    label_font_size, title_font_size : int
        Font sizes (px) for per-panel labels and the combined title.
    zoom : float
        Camera zoom multiplier applied uniformly to every panel. The
        camera ``eye`` vector is scaled by ``1.0 / zoom`` -- so values
        ABOVE 1.0 bring the camera closer (brain looks bigger) and
        values BELOW 1.0 push it further away. Default ``1.0`` (no
        change). Use ``zoom=1.5`` to make the brain about 50% bigger
        in each panel without changing the panel pixel size.
    autocrop : bool
        Trim the white border around each panel before stitching, so
        the panels sit tightly next to each other with no wasted
        whitespace. Default ``True``. The cropped panels are then
        padded back to a common size so the strip remains uniform.
        Set to ``False`` to keep plotly's default panel padding.
    autocrop_padding_px : int
        When ``autocrop=True``, leave this many pixels of padding
        around each cropped brain instead of cropping right against
        the edge. Default ``8``.

    Returns
    -------
    Path
        Path to the written stitched PNG.

    Notes
    -----
    Multi-view export is **PNG-only** by design. SVG and PDF stitching
    require separate code paths and significantly more complexity for a
    use case (one-page panel of brain views) where raster output is
    almost always what you actually want anyway.

    The function does NOT mutate the input figure -- it serializes it
    via ``fig.to_dict()`` and re-creates an independent ``go.Figure``
    for every panel. The original figure can be saved as HTML
    afterward without picking up any of the per-panel modifications.
    """
    import copy
    import tempfile
    try:
        from PIL import Image, ImageDraw, ImageFont
    except ImportError as e:
        raise ImportError(
            "Multi-view export requires Pillow. Install with: pip install pillow"
        ) from e

    if not views:
        raise ValueError("`views` must contain at least one entry")

    # Resolve each view entry to a full {name, eye, center, up} dict.
    resolved_views: List[Dict] = []
    for v in views:
        if isinstance(v, str):
            cam = CameraController.get_camera_position(v)
            if 'name' not in cam:
                cam['name'] = v.replace('_', ' ').title()
            resolved_views.append(cam)
        elif isinstance(v, dict) and 'eye' in v:
            cam = dict(v)
            if 'center' not in cam:
                cam['center'] = {'x': 0, 'y': 0, 'z': 0}
            if 'up' not in cam:
                cam['up'] = {'x': 0, 'y': 0, 'z': 1}
            if 'name' not in cam:
                cam['name'] = 'Custom View'
            resolved_views.append(cam)
        else:
            raise ValueError(
                f"Each entry in `views` must be a preset name (str) or a "
                f"camera dict with at least an 'eye' key. Got: {v!r}"
            )

    if panel_labels is None:
        panel_labels = [v['name'] for v in resolved_views]
    elif len(panel_labels) != len(resolved_views):
        raise ValueError(
            f"`panel_labels` length ({len(panel_labels)}) does not match "
            f"`views` length ({len(resolved_views)})"
        )

    # Compute the scale factor used by kaleido for DPI. Cap at 8.0
    # (effective ~576 DPI) -- enough for most publication needs while
    # keeping memory usage reasonable for very large strips.
    scale = min(image_dpi / 72.0, 8.0)
    pwidth_px = int(panel_width * scale)
    pheight_px = int(panel_height * scale)

    # Apply the zoom multiplier to every panel's eye vector. The plotly
    # convention is that smaller |eye| brings the camera closer to the
    # origin (and the brain looks bigger), so to make zoom=1.5 mean
    # "1.5x bigger brain" we divide eye coords by zoom.
    if zoom and zoom != 1.0:
        eye_scale = 1.0 / float(zoom)
        for cam in resolved_views:
            cam['eye'] = {
                'x': float(cam['eye']['x']) * eye_scale,
                'y': float(cam['eye']['y']) * eye_scale,
                'z': float(cam['eye']['z']) * eye_scale,
            }

    # Render each panel to a temporary file.
    fig_dict = fig.to_dict()
    panel_pngs: List[Path] = []
    tmpdir = Path(tempfile.mkdtemp(prefix='hlplot_multiview_'))
    try:
        for i, cam in enumerate(resolved_views):
            sub = copy.deepcopy(fig_dict)
            layout = sub.setdefault('layout', {})
            scene = layout.setdefault('scene', {})
            scene['camera'] = {
                'eye': cam['eye'],
                'center': cam['center'],
                'up': cam['up'],
            }
            # Strip per-panel decorations
            layout['title'] = {'text': ''}
            layout['updatemenus'] = []
            layout['annotations'] = []
            layout['shapes'] = []
            if not (keep_first_legend and i == 0):
                layout['showlegend'] = False
            # Tight per-panel margins so the brain fills the panel and
            # the bg isn't padded with extra whitespace before autocrop.
            layout['margin'] = {'l': 0, 'r': 0, 't': 0, 'b': 0}
            sub_fig = go.Figure(sub)
            tmp_png = tmpdir / f'panel_{i:02d}.png'
            sub_fig.write_image(
                str(tmp_png),
                format='png',
                width=panel_width,
                height=panel_height,
                scale=scale,
            )
            panel_pngs.append(tmp_png)

        # ----- Auto-crop white borders around each panel ---------------
        # Pillow auto-cropping: detect the bounding box of non-white
        # pixels in each panel, crop, then pad ALL cropped panels back
        # to the same uniform size so the strip stays a clean grid.
        def _autocrop_white(im, bg_rgb=(255, 255, 255), tol=8, pad=0):
            arr = np.array(im.convert('RGB'))
            diff = np.abs(arr.astype(int) - np.array(bg_rgb)).sum(axis=2)
            mask = diff > (tol * 3)
            if not mask.any():
                return im
            rows = np.any(mask, axis=1)
            cols = np.any(mask, axis=0)
            r0, r1 = int(np.where(rows)[0][0]), int(np.where(rows)[0][-1])
            c0, c1 = int(np.where(cols)[0][0]), int(np.where(cols)[0][-1])
            h, w = arr.shape[:2]
            r0 = max(0, r0 - pad)
            c0 = max(0, c0 - pad)
            r1 = min(h - 1, r1 + pad)
            c1 = min(w - 1, c1 + pad)
            return im.crop((c0, r0, c1 + 1, r1 + 1))

        bg_rgb = (255, 255, 255) if bg_color in ('white', '#ffffff', '#fff') else (255, 255, 255)
        loaded_panels = [Image.open(p).convert('RGB') for p in panel_pngs]

        if autocrop:
            cropped = [
                _autocrop_white(im, bg_rgb=bg_rgb, pad=autocrop_padding_px)
                for im in loaded_panels
            ]
            # Find the largest cropped width / height across panels and
            # pad EVERY cropped panel to that uniform size, brain
            # centered. This keeps the strip aligned but eliminates the
            # plotly default padding.
            max_w = max(im.size[0] for im in cropped)
            max_h = max(im.size[1] for im in cropped)
            uniform_panels = []
            for im in cropped:
                canvas = Image.new('RGB', (max_w, max_h), bg_color)
                px = (max_w - im.size[0]) // 2
                py = (max_h - im.size[1]) // 2
                canvas.paste(im, (px, py))
                uniform_panels.append(canvas)
            pwidth_px = max_w
            pheight_px = max_h
        else:
            uniform_panels = loaded_panels

        # Reserved vertical space for combined title and per-panel labels.
        title_h = int((title_font_size * 2.5) * scale) if title else 0
        label_h = int((label_font_size * 2.0) * scale)
        strip_w = pwidth_px * len(uniform_panels)
        strip_h = title_h + pheight_px + label_h

        # Stitch with Pillow.
        strip = Image.new('RGB', (strip_w, strip_h), bg_color)
        draw = ImageDraw.Draw(strip)

        # Try to load a real TrueType font; fall back to the default
        # bitmap font if no TTF is available on the system.
        def _load_font(px_size):
            for candidate in (
                'arial.ttf', 'Arial.ttf', 'DejaVuSans.ttf',
                'LiberationSans-Regular.ttf', 'Helvetica.ttf',
            ):
                try:
                    return ImageFont.truetype(candidate, int(px_size))
                except (OSError, IOError):
                    continue
            return ImageFont.load_default()

        title_font = _load_font(title_font_size * scale) if title else None
        label_font = _load_font(label_font_size * scale)

        # Combined title row
        if title:
            try:
                bbox = draw.textbbox((0, 0), title, font=title_font)
                tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
            except AttributeError:
                tw, th = draw.textsize(title, font=title_font)
            draw.text(
                ((strip_w - tw) / 2, (title_h - th) / 2),
                title,
                fill='black',
                font=title_font,
            )

        # Paste the (autocropped + uniformly padded) panels into the strip
        # and draw the per-panel labels below each one.
        for i, img in enumerate(uniform_panels):
            x_offset = i * pwidth_px
            strip.paste(img, (x_offset, title_h))

            label = panel_labels[i]
            if label:
                try:
                    bbox = draw.textbbox((0, 0), label, font=label_font)
                    lw, lh = bbox[2] - bbox[0], bbox[3] - bbox[1]
                except AttributeError:
                    lw, lh = draw.textsize(label, font=label_font)
                lx = x_offset + (pwidth_px - lw) / 2
                ly = title_h + pheight_px + (label_h - lh) / 2
                draw.text((lx, ly), label, fill='black', font=label_font)

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        strip.save(str(output_path), format='PNG')
        return output_path
    finally:
        # Clean up the temp panel pngs.
        for p in panel_pngs:
            try:
                p.unlink()
            except OSError:
                pass
        try:
            tmpdir.rmdir()
        except OSError:
            pass


def _add_size_width_legend(
    fig: go.Figure,
    *,
    node_sizes: Optional[np.ndarray] = None,
    node_size_legend_title: str = "Node size",
    node_size_legend_values: Optional[np.ndarray] = None,
    edge_widths: Optional[np.ndarray] = None,
    edge_width_legend_title: str = "Edge weight",
    edge_width_legend_values: Optional[np.ndarray] = None,
    edge_color: str = '#444',
    n_entries: int = 5,
    x_center: float = 0.5,
) -> None:
    """
    Mutate ``fig`` in place to add a paper-coordinate legend strip showing
    sample dots (for vector node sizes) and/or sample line widths (for
    weight-scaled edges) above the existing camera-controls annotation.

    Implementation: uses ``layout.shapes`` (paper-coord circles and lines)
    plus ``layout.annotations`` (paper-coord text). All elements live in
    paper coordinates so they overlay on top of the 3D ``scene`` regardless
    of camera angle, and they render in static PNG/SVG/PDF exports.

    Each legend block is laid out horizontally with ``n_entries`` evenly
    spaced sample dots/lines, the value labels just below them, and the
    legend title centered above. When both blocks are requested they
    stack vertically (node size on top, edge width below).

    Parameters
    ----------
    fig : plotly.graph_objects.Figure
        Figure to mutate. Existing shapes and annotations are preserved.
    node_sizes : np.ndarray, optional
        Array of per-node pixel sizes. When provided, a 5-entry size
        legend is added.
    node_size_legend_title : str
        Title shown above the size legend (default ``"Node size"``).
    node_size_legend_values : np.ndarray, optional
        Per-node values to use for the SIZE LEGEND LABELS instead of the
        literal pixel sizes. Same length as ``node_sizes``. When provided,
        the legend labels are 5 evenly-spaced values from this array, and
        the dots themselves are drawn at the pixel sizes that correspond
        to those metric values (looked up by index of the closest match).
        Use this when the user passes ``--node-size-legend-metric pc`` to
        get a legend like "PC: 0.10, 0.30, 0.50, 0.70, 0.90" instead of
        "Node size: 8, 12, 16, 20, 25".
    edge_widths : np.ndarray, optional
        Array of edge weights/strengths. When provided, a 5-entry width
        legend is added.
    edge_width_legend_title : str
        Title shown above the width legend.
    edge_width_legend_values : np.ndarray, optional
        Same as ``node_size_legend_values`` but for edges. In p-value mode
        the caller passes the original p-values here so the legend reads
        e.g. "p-value: 0.05, 0.04, 0.02, 0.01, 0.001" rather than the
        ``-log10(p)`` weights.
    edge_color : str
        Color of the sample line segments in the width legend.
    n_entries : int
        Number of sample entries per legend block (default 5).
    y_top : float
        Top of the legend strip in paper coordinates [0, 1]. Default
        ``0.18`` -- just above the "Camera controls enabled" annotation.
    x_center : float
        Horizontal center of the strip in paper coordinates. Default 0.5.
    """
    if node_sizes is None and edge_widths is None:
        return  # nothing to draw

    new_shapes = list(fig.layout.shapes) if fig.layout.shapes else []
    new_annotations = list(fig.layout.annotations) if fig.layout.annotations else []

    # Read the actual figure dimensions so we can map pixels <-> paper
    # coords correctly. The plot functions hardcode 1200 x 900 but we
    # defensively fall back if those aren't set yet.
    fig_w = float(fig.layout.width) if fig.layout.width else 1200.0
    fig_h = float(fig.layout.height) if fig.layout.height else 900.0

    # ------------------------------------------------------------------
    # Reserve room for the legend in the figure's bottom margin so the
    # 3D scene shrinks slightly upward and the brain can never overlap
    # the legend, regardless of camera angle.
    #
    # Plotly's default bottom margin is small (~80 px). We bump it by
    # ~95 px per legend block, then place the legend in that newly
    # reserved area.
    # ------------------------------------------------------------------
    n_blocks = (1 if node_sizes is not None else 0) + (
        1 if edge_widths is not None else 0
    )
    if n_blocks == 0:
        return

    margin_per_block_px = 95
    extra_b_px = margin_per_block_px * n_blocks

    current_b = 80
    if fig.layout.margin is not None and fig.layout.margin.b is not None:
        current_b = int(fig.layout.margin.b)
    fig.update_layout(margin=dict(b=current_b + extra_b_px))

    # ------------------------------------------------------------------
    # Block layout (paper coords).
    #
    # Stack the blocks vertically from the BOTTOM up so the lowest label
    # is always at the same fixed y, just above the small "V2: Camera
    # controls enabled" annotation in the corner. This way single-block
    # and double-block layouts both look right.
    # ------------------------------------------------------------------
    title_dy = 0.024            # title sits this far ABOVE marker row
    label_dy = 0.024            # label sits this far BELOW marker row
    block_height = 0.075        # vertical span of one block
    entry_spacing = 0.060       # horizontal distance between consecutive samples
    total_width = entry_spacing * (n_entries - 1)
    x_start = x_center - total_width / 2.0

    # Lowest (bottom-most) block's marker row y-coord. Below it sits the
    # label row at y_label_bottom = y_marker_bottom - label_dy. We want
    # that label row to clear the V2 annotation at y=0.01, so set:
    y_marker_bottom = 0.05
    # Block 1 (the topmost) marker is at:
    y_marker_top = y_marker_bottom + (n_blocks - 1) * block_height

    current_y = y_marker_top  # we'll subtract block_height as we go

    # ------------------------------------------------------------------
    # Block 1: node size
    # ------------------------------------------------------------------
    if node_sizes is not None and len(node_sizes) > 0:
        # Choose the n entries.
        if (
            node_size_legend_values is not None
            and len(node_size_legend_values) == len(node_sizes)
        ):
            # Metric-driven labels: pick n_entries evenly spaced values
            # from the metric column, then look up each one's matching
            # node and use that node's pixel size for the sample dot.
            mvals = np.asarray(node_size_legend_values, dtype=float)
            valid = ~np.isnan(mvals)
            if valid.sum() >= 2:
                mvals_v = mvals[valid]
                sizes_v = np.asarray(node_sizes, dtype=float)[valid]
                lo, hi = np.nanmin(mvals_v), np.nanmax(mvals_v)
                ticks = np.linspace(lo, hi, n_entries)
                sample_labels = [f"{t:.3g}" for t in ticks]
                sample_sizes_px = []
                for t in ticks:
                    idx = int(np.argmin(np.abs(mvals_v - t)))
                    sample_sizes_px.append(float(sizes_v[idx]))
            else:
                sample_labels = []
                sample_sizes_px = []
        else:
            # Literal pixel-size labels.
            sizes_arr = np.asarray(node_sizes, dtype=float)
            sizes_arr = sizes_arr[~np.isnan(sizes_arr)]
            if sizes_arr.size >= 2:
                lo, hi = float(np.min(sizes_arr)), float(np.max(sizes_arr))
                ticks = np.linspace(lo, hi, n_entries)
                sample_labels = [f"{t:.3g}" for t in ticks]
                sample_sizes_px = list(ticks)
            else:
                sample_labels = []
                sample_sizes_px = []

        if sample_labels:
            # Title
            new_annotations.append(dict(
                text=f"<b>{node_size_legend_title}</b>",
                showarrow=False,
                xref='paper', yref='paper',
                x=x_center, y=current_y + title_dy,
                xanchor='center', yanchor='middle',
                font=dict(size=11, color='black'),
            ))
            # Sample dots: drawn as filled circle shapes in paper coords.
            # PAPER COORDS ARE NOT SQUARE -- they go from 0..1 in both
            # axes but the figure isn't (1200 x 900 by default), so a
            # paper-coord "circle" with rx=ry comes out as an ellipse.
            # We compensate by computing rx and ry from the SAME pixel
            # diameter via the figure's actual width/height. The diameter
            # we use is the LITERAL pixel size of the corresponding node
            # marker so the legend dots match the brain dots visually.
            for i, (lab, px) in enumerate(zip(sample_labels, sample_sizes_px)):
                xc = x_start + i * entry_spacing
                diameter_px = float(px)
                rx_paper = (diameter_px / 2.0) / fig_w
                ry_paper = (diameter_px / 2.0) / fig_h
                new_shapes.append(dict(
                    type='circle',
                    xref='paper', yref='paper',
                    x0=xc - rx_paper, x1=xc + rx_paper,
                    y0=current_y - ry_paper, y1=current_y + ry_paper,
                    fillcolor='rgba(110, 70, 180, 0.85)',
                    line=dict(color='rgba(40, 20, 80, 0.95)', width=1),
                ))
                new_annotations.append(dict(
                    text=lab,
                    showarrow=False,
                    xref='paper', yref='paper',
                    x=xc, y=current_y - label_dy,
                    xanchor='center', yanchor='middle',
                    font=dict(size=10, color='black'),
                ))
            current_y -= block_height

    # ------------------------------------------------------------------
    # Block 2: edge width
    # ------------------------------------------------------------------
    if edge_widths is not None and len(edge_widths) > 0:
        widths_arr = np.asarray(edge_widths, dtype=float)
        widths_arr = widths_arr[~np.isnan(widths_arr)]
        if widths_arr.size >= 2:
            # Choose ticks based on either the legend values (e.g. p-values)
            # or the raw widths.
            if (
                edge_width_legend_values is not None
                and len(edge_width_legend_values) > 0
            ):
                lvals = np.asarray(edge_width_legend_values, dtype=float)
                lvals = lvals[~np.isnan(lvals)]
                if lvals.size >= 2:
                    lo, hi = float(np.min(lvals)), float(np.max(lvals))
                    ticks = np.linspace(lo, hi, n_entries)
                    sample_labels = [f"{t:.3g}" for t in ticks]
                    # The line widths themselves: pick from the actual
                    # widths array at the same index as the closest
                    # tick value in lvals (we don't have a 1:1 mapping
                    # between widths_arr and lvals beyond ordering).
                    # Simpler: use n_entries evenly spaced widths.
                    sample_widths_px = list(np.linspace(
                        float(np.min(widths_arr)),
                        float(np.max(widths_arr)),
                        n_entries,
                    ))
                else:
                    sample_labels = []
                    sample_widths_px = []
            else:
                lo, hi = float(np.min(widths_arr)), float(np.max(widths_arr))
                ticks = np.linspace(lo, hi, n_entries)
                sample_labels = [f"{t:.3g}" for t in ticks]
                sample_widths_px = list(ticks)

            if sample_labels:
                new_annotations.append(dict(
                    text=f"<b>{edge_width_legend_title}</b>",
                    showarrow=False,
                    xref='paper', yref='paper',
                    x=x_center, y=current_y + title_dy,
                    xanchor='center', yanchor='middle',
                    font=dict(size=11, color='black'),
                ))
                # Sample line segments: paper-coord line shapes. Map
                # the px width to a plotly line.width (we can use it
                # directly because layout.shapes line.width IS in pixels).
                max_w = max(sample_widths_px) if sample_widths_px else 1.0
                for i, (lab, w) in enumerate(zip(sample_labels, sample_widths_px)):
                    xc = x_start + i * entry_spacing
                    half = 0.020  # half-length of the sample line in paper coords
                    new_shapes.append(dict(
                        type='line',
                        xref='paper', yref='paper',
                        x0=xc - half, x1=xc + half,
                        y0=current_y, y1=current_y,
                        line=dict(
                            color=edge_color,
                            width=max(1.0, float(w)),
                        ),
                    ))
                    new_annotations.append(dict(
                        text=lab,
                        showarrow=False,
                        xref='paper', yref='paper',
                        x=xc, y=current_y - label_dy,
                        xanchor='center', yanchor='middle',
                        font=dict(size=10, color='black'),
                    ))

    fig.update_layout(shapes=new_shapes, annotations=new_annotations)


def _build_camera_readout_js() -> str:
    """
    Return a small inline JavaScript snippet that adds a live camera-position
    overlay to the saved HTML. The overlay sits in the top-right corner of
    the plot container (not the browser viewport), updates in real time as
    the user rotates the brain, and shows the current ``scene.camera``
    eye/center/up plus a copy-pastable block of CLI flags that reproduces
    the current view.

    Implementation notes:

    - The overlay div is appended to the plotly graph div ("gd") with
      ``position: absolute``, so it scrolls with the plot and is contained
      within its bounding box. ``gd.style.position`` is set to ``relative``
      defensively so the absolute child anchors to gd rather than to some
      ancestor.
    - The overlay is fully selectable (``user-select: text``) so users can
      drag-select and copy any portion of the text. There is intentionally
      no copy button -- selection is the only interaction.
    - The overlay is HTML-only. ``post_script`` is only attached when we
      call ``fig.write_html(...)`` from the plot functions, NOT when the
      user calls ``fig.show()`` in Jupyter, so this never affects in-notebook
      rendering. It also never appears in static PNG/SVG/PDF exports because
      kaleido renders the figure JSON without running JS.
    """
    return r"""
(function() {
    var gd = document.getElementsByClassName('plotly-graph-div')[0];
    if (!gd) { return; }

    // Make the plot container the positioning context for the absolute
    // overlay below. Plotly usually leaves gd as static / unset, so set
    // it explicitly. This does NOT affect plotly's own internal layout
    // because plotly's children are absolutely positioned themselves.
    if (!gd.style.position || gd.style.position === 'static') {
        gd.style.position = 'relative';
    }

    var div = document.createElement('div');
    div.id = 'hlplot-camera-readout';
    div.style.cssText = (
        // Anchored inside gd, top-right corner. The 60px top offset
        // leaves room for plotly's modebar (camera/zoom/reset icons)
        // which sits flush with the top of the plot div.
        'position:absolute;top:60px;right:10px;'
        + 'background:rgba(255,255,255,0.95);border:1px solid #444;'
        + 'padding:8px 10px;font-family:monospace;font-size:11px;'
        + 'line-height:1.35;white-space:pre;z-index:1000;'
        + 'max-width:360px;border-radius:4px;'
        + 'box-shadow:0 2px 6px rgba(0,0,0,0.15);'
        // Make the text drag-selectable so the user can copy any portion
        // of it (including just the CLI flag block). Cross-vendor prefixes
        // are included so this works in older browsers / Safari.
        + 'user-select:text;-webkit-user-select:text;-moz-user-select:text;'
        + '-ms-user-select:text;cursor:text;'
    );
    gd.appendChild(div);

    function fmt(v) { return Number(v).toFixed(3); }
    function update() {
        var cam = (gd.layout && gd.layout.scene && gd.layout.scene.camera) || {};
        var e = cam.eye || {x:0, y:0, z:0};
        var c = cam.center || {x:0, y:0, z:0};
        var u = cam.up || {x:0, y:0, z:1};
        var ex = fmt(e.x), ey = fmt(e.y), ez = fmt(e.z);
        var cx = fmt(c.x), cy = fmt(c.y), cz = fmt(c.z);
        var ux = fmt(u.x), uy = fmt(u.y), uz = fmt(u.z);
        div.textContent =
            'Camera (live):\n'
            + '  eye    = ' + ex + ', ' + ey + ', ' + ez + '\n'
            + '  center = ' + cx + ', ' + cy + ', ' + cz + '\n'
            + '  up     = ' + ux + ', ' + uy + ', ' + uz + '\n'
            + '\nCLI flags to reproduce this view:\n'
            + '  --custom-camera-eye '    + ex + ',' + ey + ',' + ez + ' \\\n'
            + '  --custom-camera-center ' + cx + ',' + cy + ',' + cz + ' \\\n'
            + '  --custom-camera-up '     + ux + ',' + uy + ',' + uz + ' \\\n'
            + '  --custom-camera-name "My View"';
    }
    update();
    gd.on('plotly_relayout', update);
})();
"""


def _resolve_mesh_lighting(
    style: Optional[str],
    ambient: Optional[float],
    diffuse: Optional[float],
    specular: Optional[float],
    roughness: Optional[float],
    fresnel: Optional[float],
    fast_render: bool,
) -> Optional[Dict[str, float]]:
    """
    Build the ``lighting`` dict to pass to ``plotly.graph_objects.Mesh3d``.

    Resolution order:

    1. If the user passed nothing (no style, no raw knob) we preserve the
       existing default behavior: ``None`` normally, or ``{ambient: 0.8}``
       when ``fast_render`` is on.
    2. Otherwise we start from the named ``style`` preset (or an empty
       dict if ``style`` is None / 'default').
    3. Any raw knob the user supplied overrides the corresponding preset
       value.

    This means ``style='glossy', specular=2.0`` is glossy with the
    specular cranked higher, and the user never has to remember the full
    set of parameters when they only want to tweak one.
    """
    untouched = (
        style is None
        and ambient is None
        and diffuse is None
        and specular is None
        and roughness is None
        and fresnel is None
    )
    if untouched:
        return dict(ambient=0.8) if fast_render else None

    if style is not None and style != 'default' and style not in MESH_LIGHTING_PRESETS:
        valid = ['default'] + list(MESH_LIGHTING_PRESETS.keys())
        raise ValueError(
            f"Unknown mesh_style {style!r}. Must be one of: {valid}"
        )

    base: Dict[str, float] = {}
    if style and style != 'default':
        base = dict(MESH_LIGHTING_PRESETS[style])

    if ambient is not None:
        base['ambient'] = float(ambient)
    if diffuse is not None:
        base['diffuse'] = float(diffuse)
    if specular is not None:
        base['specular'] = float(specular)
    if roughness is not None:
        base['roughness'] = float(roughness)
    if fresnel is not None:
        base['fresnel'] = float(fresnel)

    return base or None


def _export_figure_static(
    fig,
    export_image,
    multi_view=None,
    multi_view_panel_size=(800, 800),
    multi_view_panel_labels=None,
    multi_view_keep_first_legend=True,
    multi_view_zoom=1.0,
    image_dpi=300,
    image_format='png',
    plot_title='',
    export_show_title=True,
    export_show_legend=True,
):
    """Render ``fig`` to a static image file.

    When ``multi_view`` is set, ``export_image`` is reinterpreted as the
    path of a stitched 1xN PNG produced by
    :func:`export_multi_view_stitched_png`. Otherwise the figure is
    exported as a single PNG/SVG/PDF/JPEG/WEBP via kaleido.

    Used by both ``create_brain_connectivity_plot`` and
    ``create_brain_connectivity_plot_with_modularity`` so the modular
    plot's per-module trace rebuild is reflected in the exported image.
    """
    if multi_view:
        if export_image is None:
            print(
                "Warning: multi_view was set but export_image is None; "
                "no stitched PNG will be written. Pass export_image=... "
                "to specify the output path."
            )
            return
        stitched_path = Path(export_image)
        if stitched_path.suffix.lower() not in ('', '.png'):
            print(
                f"Note: multi_view stitched export is PNG-only. "
                f"Forcing extension on {stitched_path} to .png."
            )
            stitched_path = stitched_path.with_suffix('.png')
        print(
            f"Multi-view: rendering {len(multi_view)} panels into "
            f"{stitched_path}"
        )
        export_multi_view_stitched_png(
            fig,
            output_path=stitched_path,
            views=list(multi_view),
            panel_width=multi_view_panel_size[0],
            panel_height=multi_view_panel_size[1],
            image_dpi=image_dpi,
            title=plot_title if export_show_title else "",
            panel_labels=multi_view_panel_labels,
            keep_first_legend=multi_view_keep_first_legend
                              and export_show_legend,
            zoom=multi_view_zoom,
        )
        print(f"Wrote stitched multi-view PNG to: {stitched_path}")
        return

    if export_image is None:
        return

    export_path = Path(export_image)
    export_path.parent.mkdir(parents=True, exist_ok=True)

    if export_path.suffix:
        fmt = export_path.suffix[1:].lower()
    else:
        fmt = image_format.lower()
        export_path = export_path.with_suffix(f'.{fmt}')

    if fmt not in ['png', 'svg', 'pdf', 'jpeg', 'webp']:
        print(f"Warning: Unsupported format '{fmt}', using PNG")
        fmt = 'png'
        export_path = export_path.with_suffix('.png')

    fig_dict = fig.to_dict()
    if 'layout' in fig_dict:
        fig_dict['layout']['updatemenus'] = []
        fig_dict['layout']['annotations'] = []
        fig_dict['layout']['paper_bgcolor'] = 'white'
        fig_dict['layout']['plot_bgcolor'] = 'white'
        if not export_show_title:
            fig_dict['layout']['title'] = {'text': ''}
        if not export_show_legend:
            fig_dict['layout']['showlegend'] = False

    fig_export = go.Figure(fig_dict)

    if fmt in ['svg', 'pdf']:
        scale = 1.0
    else:
        scale = min(image_dpi / 72.0, 8.0)
        if image_dpi / 72.0 > 8.0:
            print("Note: Scale capped at 8x (effective ~576 DPI) to avoid memory issues")

    print(f"Exporting {fmt.upper()} image...")

    export_width = 1200
    export_height = 900

    try:
        fig_export.write_image(
            str(export_path),
            format=fmt,
            width=export_width,
            height=export_height,
            scale=scale,
        )
        if export_path.exists():
            file_size = export_path.stat().st_size
            print(f"Exported static image to: {export_path}")
            print(f"  Format: {fmt.upper()}, Size: {file_size/1024:.1f} KB")
            if fmt not in ['svg', 'pdf']:
                print(f"  Dimensions: {int(export_width*scale)}x{int(export_height*scale)} pixels")
        else:
            print(f"ERROR: Export failed - file was not created at {export_path}")
    except Exception as e:
        print(f"ERROR exporting image: {e}")
        print("=" * 60)
        print("Troubleshooting:")
        print("1. Make sure kaleido is installed: pip install -U kaleido")
        print("2. For PDF export, you may also need: pip install -U kaleido[pdf]")
        print("3. Try using a lower image_dpi value if memory issues occur")
        print("=" * 60)


def create_brain_connectivity_plot(
    vertices: np.ndarray,
    faces: np.ndarray,
    roi_coords_df: pd.DataFrame,
    connectivity_matrix: Union[np.ndarray, str, pd.DataFrame],
    plot_title: str = "Brain Connectivity Network",
    save_path: str = "brain_connectivity.html",
    node_size: Union[int, float, np.ndarray, pd.Series, pd.DataFrame, List, Dict, str] = 8,
    node_color: Union[str, np.ndarray, pd.Series, pd.DataFrame, List] = 'purple',
    node_border_color: str = 'magenta',
    pos_edge_color: str = 'red',
    neg_edge_color: str = 'blue',
    edge_width: Union[float, Tuple[float, float]] = (1.0, 5.0),
    edge_width_scale: float = 1.0,
    node_size_scale: float = 1.0,
    edge_threshold: float = 0.0,
    mesh_color: str = 'lightgray',
    mesh_opacity: float = 0.15,
    label_font_size: int = 8,
    fast_render: bool = False,
    camera_view: str = 'oblique',
    custom_camera: Optional[Dict] = None,
    enable_camera_controls: bool = True,
    show_only_connected_nodes: bool = True,
    node_metrics: Optional[Union[str, pd.DataFrame]] = None,
    hide_nodes_with_hidden_edges: bool = True,
    export_image: Optional[str] = None,
    image_format: str = 'png',
    image_dpi: int = 300,
    export_show_title: bool = True,
    export_show_legend: bool = True,
    edge_color_matrix: Optional[Union[str, np.ndarray, pd.DataFrame]] = None,
    matrix_type: str = 'weight',
    pvalue_threshold: float = 0.05,
    sign_matrix: Optional[Union[str, np.ndarray, pd.DataFrame]] = None,
    mesh_style: Optional[str] = None,
    mesh_ambient: Optional[float] = None,
    mesh_diffuse: Optional[float] = None,
    mesh_specular: Optional[float] = None,
    mesh_roughness: Optional[float] = None,
    mesh_fresnel: Optional[float] = None,
    mesh_light_position: Optional[Tuple[float, float, float]] = None,
    custom_camera_name: Optional[str] = None,
    show_camera_readout: bool = False,
    show_size_legend: bool = True,
    show_width_legend: bool = True,
    node_size_legend_metric: Optional[str] = None,
    multi_view: Optional[List[Union[str, Dict]]] = None,
    multi_view_panel_size: Tuple[int, int] = (800, 800),
    multi_view_panel_labels: Optional[List[str]] = None,
    multi_view_keep_first_legend: bool = True,
    multi_view_zoom: float = 1.0,
):
    """
    Create an interactive 3D brain connectivity visualization.

    Parameters
    ----------
    vertices : numpy.ndarray
        Mesh vertices array of shape (n_vertices, 3)
    faces : numpy.ndarray
        Mesh faces array of shape (n_faces, 3)
    roi_coords_df : pandas.DataFrame
        DataFrame containing ROI coordinates with columns:
        - 'cog_x', 'cog_y', 'cog_z': world coordinates
        - 'roi_name': name of the ROI
    connectivity_matrix : numpy.ndarray, str, or pd.DataFrame
        Connectivity matrix or path to file. Supports:
        - numpy array: Used directly
        - str: Path to file (.npy, .csv, .txt, .mat, .edge)
        - pd.DataFrame: Converted to numpy array
    plot_title : str, optional
        Title for the plot
    save_path : str, optional
        Path where to save the HTML file
    node_size : int, float, array-like, dict, or str, optional
        Size of the ROI nodes. Can be:
        - Scalar (int/float): All nodes same size
        - numpy array/list/Series: Per-node sizes
        - dict: {node_idx: size}
        - str: Path to file (.csv, .txt, .npy, .mat)
    node_color : str, array-like, or file path, optional
        Color of the ROI nodes. Can be:
        - Single color string: All nodes same color (e.g., 'purple', '#FF0000')
        - numpy array of integers: Module assignments (1-indexed), colors auto-generated
        - numpy array of color strings: Per-node colors
        - pandas Series/DataFrame: Colors or module assignments
        - list: Colors or module assignments
        - str: Path to file (.csv, .npy) containing assignments or colors
    node_border_color : str, optional
        Border color of the ROI nodes
    pos_edge_color : str, optional
        Color for positive connections
    neg_edge_color : str, optional
        Color for negative connections
    edge_width : float or tuple, optional
        Edge width specification:
        - float: Fixed width for all edges
        - tuple (min, max): Scale edge widths based on absolute weight
    edge_width_scale : float, optional
        Uniform multiplier applied to every edge width AFTER all other
        scaling. Use this when you've tuned ``edge_width`` and just want
        every edge proportionally thicker (or thinner) for a particular
        figure. Default ``1.0`` (no change). Examples:

        - ``edge_width=(1, 5), edge_width_scale=5`` -> final widths in [5, 25]
        - ``edge_width=2.0, edge_width_scale=5`` -> every edge has width 10
    node_size_scale : float, optional
        Uniform multiplier applied to every node size AFTER all other
        size resolution (scalar / vector / file path). Mirror of
        ``edge_width_scale`` for nodes. Use this when you've tuned
        ``node_size`` and want every dot proportionally larger or
        smaller. Default ``1.0`` (no change). Examples:

        - ``node_size=10, node_size_scale=2`` -> every dot at 20 px
        - ``node_size='sizes.csv', node_size_scale=1.5`` -> every dot 50%
          bigger; the size legend's sample dots also scale up so they
          continue to match the brain dots visually.
    edge_threshold : float, optional
        Threshold for showing edges (default 0.0 shows all non-zero)
    mesh_color : str, optional
        Color of the brain mesh
    mesh_opacity : float, optional
        Opacity of the brain mesh
    label_font_size : int, optional
        Font size for ROI labels
    fast_render : bool, optional
        If True, uses optimizations for faster rendering
    camera_view : str, optional
        Camera view preset name (default 'oblique')
    custom_camera : dict, optional
        Custom camera position dict with 'eye', 'center', 'up' keys
    enable_camera_controls : bool, optional
        Whether to enable camera view dropdown controls (default True)
    show_only_connected_nodes : bool, optional
        If True (default), only show nodes with at least one edge
    node_metrics : str or pd.DataFrame, optional
        Node metrics for hover display. Rows = nodes, columns = metric names.
        Can be CSV path or DataFrame.
    hide_nodes_with_hidden_edges : bool, optional
        If True (default), nodes with only positive or only negative edges
        will be hidden when those edge types are toggled off in the legend.
    export_image : str, optional
        Path to export static image (e.g., 'output.png'). If None, no image exported.
        Supported formats: png, svg, pdf (requires kaleido package).
        Note: Exported images use fixed 1200x900 dimensions for consistency.
    image_format : str, optional
        Image format if export_image doesn't have extension: 'png', 'svg', 'pdf'
    image_dpi : int, optional
        DPI for exported PNG/JPEG images (default 300 for publication quality).
        Vector formats (SVG, PDF) ignore this setting.
    export_show_title : bool, optional
        Whether to show the title in exported images (default True).
    export_show_legend : bool, optional
        Whether to show the legend in exported images (default True).
    edge_color_matrix : str, np.ndarray, or pd.DataFrame, optional
        Per-edge color matrix of the SAME shape as ``connectivity_matrix``.
        Each cell ``[i, j]`` may be:

        - a CSS color name (``"red"``), hex code (``"#FF0000"``), or
          ``"rgb(R,G,B)"`` string -- used directly as the color of the
          edge between ROI i and ROI j;
        - an integer label (``1, 2, 3, ...``) -- all edges sharing the
          label get the same auto-generated color from the same palette
          used for module assignments;
        - empty / NaN / 0 -- the edge between i and j is **skipped**
          (not drawn) for that cell.

        When provided, ``edge_color_matrix`` overrides the default
        positive/negative coloring controlled by ``pos_edge_color`` and
        ``neg_edge_color``. Each edge becomes its own trace in the figure
        so the per-edge color is preserved.
    matrix_type : str, optional
        Interpretation of ``connectivity_matrix``. One of:

        - ``'weight'`` (default): the matrix holds connection strengths /
          correlations / weights. Sign determines pos/neg coloring.
        - ``'pvalue'``: the matrix holds p-values in ``(0, 1]``. The matrix
          is internally transformed via ``-log10(p)`` so that smaller
          p-values produce thicker edges. The actual p-value is shown in
          the hover text. ``pvalue_threshold`` filters out non-significant
          cells, and ``sign_matrix`` (optional) makes the result signed
          for pos/neg coloring.
    pvalue_threshold : float, optional
        Only used when ``matrix_type='pvalue'``. Cells with
        ``p > pvalue_threshold`` are dropped (not drawn). Default
        ``0.05``. Set to ``1.0`` to draw every p-value.
    sign_matrix : str, np.ndarray, or pd.DataFrame, optional
        Only used when ``matrix_type='pvalue'``. A matrix of the same
        shape as ``connectivity_matrix`` whose sign indicates the sign of
        the underlying effect (``+1`` for positive, ``-1`` for negative,
        ``0`` for unsigned). When provided, the transformed weight matrix
        is multiplied by the sign so positive effects render as positive
        edges (``pos_edge_color``) and negative effects render as negative
        edges (``neg_edge_color``). When omitted, every p-value renders as
        a positive edge.
    mesh_style : str, optional
        High-level lighting preset for the brain mesh. One of:

        - ``None`` / ``'default'`` (default): preserve the legacy behavior
          (no lighting params unless ``fast_render`` is on, in which case
          ``ambient=0.8`` is set). Looks identical to all previous releases.
        - ``'flat'``: pure ambient, no shading at all. Mesh looks like a
          paper cutout.
        - ``'matte'``: chalk-like surface, no specular highlight.
        - ``'smooth'``: balanced default with a small specular component
          and a hint of fresnel rim light.
        - ``'glossy'``: shiny plastic look with a sharp specular highlight.
        - ``'mirror'``: maximum specular, near-zero roughness, near-chrome.

        Any of the raw knobs below (``mesh_ambient`` ... ``mesh_fresnel``)
        OVERRIDE individual values from the chosen preset, so
        ``mesh_style='glossy', mesh_specular=2.0`` is "glossy with the
        specular cranked higher".
    mesh_ambient : float, optional
        Override of the mesh ``lighting.ambient`` value (0-1). Higher
        values make the mesh self-illuminate (no shadows). Plotly default
        is ``0.8``.
    mesh_diffuse : float, optional
        Override of the mesh ``lighting.diffuse`` value (0-1). Higher
        values give a stronger directional shading contour. Plotly
        default is ``0.8``.
    mesh_specular : float, optional
        Override of the mesh ``lighting.specular`` value (0-2). This is
        the "glossy" knob: higher values produce brighter highlights.
        Plotly default is ``0.05`` (almost matte).
    mesh_roughness : float, optional
        Override of the mesh ``lighting.roughness`` value (0-1). This is
        the "shiny" knob: lower values produce sharper, smaller highlights
        (polished surface), higher values produce broader, dimmer
        highlights (sandpaper). Plotly default is ``0.5``.
    mesh_fresnel : float, optional
        Override of the mesh ``lighting.fresnel`` value (0-5). Controls
        the rim light at glancing angles -- the "edge glow" you see on
        shiny plastic. Plotly default is ``0.2``.
    mesh_light_position : tuple of (x, y, z), optional
        Where the directional light is located in 3D world space. When
        ``None`` (default) plotly's default light position is used.
    custom_camera_name : str, optional
        Display name to use for ``custom_camera`` when one is provided.
        Shows up both in the plot title (e.g. ``"View: My Custom Angle"``)
        and as the label of an extra entry appended to the camera dropdown
        in the HTML, so the user can flip back to the custom view at any
        time. Ignored when ``custom_camera`` is ``None``. When omitted but
        ``custom_camera`` is set, defaults to ``"Custom View"``.
    show_camera_readout : bool, optional
        When ``True``, inject a small fixed-position overlay into the saved
        HTML that shows the **live** camera ``eye``, ``center``, and ``up``
        as the user rotates the brain, plus a copy-pastable block of CLI
        flags (``--custom-camera-eye ... --custom-camera-center ...``)
        that reproduce the current view. Use this to find an angle
        interactively, then paste the printed flags into a future
        invocation to reproduce that view in PNG/SVG/PDF exports.
        Default ``False``. **Set to False (or omit) if you do not want
        the overlay to appear in your saved HTML.** The overlay is HTML
        only -- it never appears in static image exports because those
        are rendered by kaleido on the server, not from the HTML.
    show_size_legend : bool, optional
        Render a 5-entry "node size" key (sample dots labeled with the
        sizes they represent) as a paper-coordinate overlay near the
        bottom of the plot. Default ``True``. The key is automatically
        skipped when ``node_size`` is a scalar (because every dot would
        be the same size and the key would be pointless). Set to
        ``False`` to suppress it for a clean publication figure.
    show_width_legend : bool, optional
        Render a 5-entry "edge width" key (sample line segments labeled
        with the weight they represent) as a paper-coordinate overlay
        near the bottom of the plot. Default ``True``. Automatically
        skipped when ``edge_width`` is a scalar. In
        ``matrix_type='pvalue'`` mode the labels show the **original
        p-values**, not the ``-log10(p)`` transform.
    node_size_legend_metric : str, optional
        Name of a column in ``node_metrics`` to use as the SOURCE OF
        LABELS in the node size key. When set:

        - the size key title becomes that column name
          (e.g. ``"participation_coef"``);
        - the 5 sample labels are 5 evenly-spaced values from that
          column (e.g. ``"0.10, 0.30, 0.50, 0.70, 0.90"``);
        - the sample DOTS are still drawn at the actual pixel sizes,
          paired by index of the node whose metric value is closest to
          each tick.

        When unset (default), the key shows literal pixel sizes from
        ``node_size``. Use this when you've pre-scaled a metric (e.g.
        PC) into pixel sizes and want the key to show the metric
        values, not the pixel sizes.
    multi_view : list of (str | dict), optional
        When set, ALSO render a stitched 1xN PNG strip of the brain
        from each of the requested views, in addition to the normal
        single HTML/static export. Each entry is either:

        - the name of a built-in preset (``'left'``, ``'superior'``,
          ``'anterior'``, ``'right'``, ``'posterior'``, ``'inferior'``,
          ``'oblique'``, ``'anterolateral_left'``, ...); or
        - a custom camera dict ``{'name': str, 'eye': dict, 'center':
          dict, 'up': dict}`` -- the same shape that ``custom_camera``
          accepts.

        The output path for the stitched PNG is ``export_image``: when
        ``multi_view`` is set, ``export_image`` is REINTERPRETED as the
        stitched output path (the single-image export is suppressed).
        Multi-view output is **PNG only** -- SVG/PDF stitching is not
        supported. The function uses
        :func:`export_multi_view_stitched_png` under the hood.
    multi_view_panel_size : tuple of (int, int), optional
        Pixel size of EACH individual panel before the DPI scale factor
        is applied. Default ``(800, 800)``. The final stitched image is
        ``len(multi_view) * panel_width`` wide by ``panel_height + label
        rows + title row`` tall.
    multi_view_panel_labels : list of str, optional
        Per-panel labels drawn below each panel in the stitched image.
        Length must equal ``len(multi_view)``. When ``None``, sensible
        defaults are pulled from each view (preset name -> capitalized
        form, custom dict -> ``view['name']`` if present).
    multi_view_keep_first_legend : bool, optional
        When ``True`` (default), the first panel of the stitched strip
        keeps its plotly legend (positive/negative edges, modules, ...)
        and the remaining panels are rendered without it. The first
        panel will look slightly different from the others (the legend
        eats some left-side space). Set to ``False`` to strip the
        legend from every panel for a clean visual repeat.
    multi_view_zoom : float, optional
        Camera zoom multiplier applied uniformly to every panel of the
        stitched strip. Values ABOVE ``1.0`` bring the camera closer
        and make the brain look bigger; values BELOW ``1.0`` push it
        further away. Default ``1.0`` (no change). Examples:
        ``multi_view_zoom=1.5`` makes the brain about 50% bigger;
        ``multi_view_zoom=2.0`` doubles its apparent size.

    Returns
    -------
    fig : plotly.graph_objects.Figure
        The plotly figure object
    graph_stats : dict
        Dictionary containing graph statistics
    """

    print(f"Creating brain connectivity visualization: {plot_title}")

    # Load connectivity matrix from various input types
    conn_matrix = load_connectivity_input(connectivity_matrix, n_expected_nodes=len(roi_coords_df))
    print(f"Connectivity matrix shape: {conn_matrix.shape}")

    # ----- p-value mode -----------------------------------------------
    # If matrix_type='pvalue', interpret conn_matrix as a p-value matrix
    # and convert it (via -log10) into a signed weight matrix that the
    # rest of this function can plot normally.
    pvalue_lookup: Optional[np.ndarray] = None
    if matrix_type not in ('weight', 'pvalue'):
        raise ValueError(
            f"matrix_type must be 'weight' or 'pvalue', got {matrix_type!r}"
        )
    if matrix_type == 'pvalue':
        sign_arr = None
        if sign_matrix is not None:
            sign_arr = load_connectivity_input(
                sign_matrix, n_expected_nodes=conn_matrix.shape[0]
            )
        weight_matrix, pvalue_lookup = transform_pvalue_matrix(
            conn_matrix,
            pvalue_threshold=pvalue_threshold,
            sign_matrix=sign_arr,
        )
        n_kept = int(np.sum(weight_matrix != 0))
        print(
            f"matrix_type='pvalue': transformed via -log10(p), kept "
            f"{n_kept // 2} edges with p <= {pvalue_threshold}"
            + (" (signed)" if sign_arr is not None else "")
        )
        conn_matrix = weight_matrix

    # ----- per-edge color matrix --------------------------------------
    edge_color_arr: Optional[np.ndarray] = None
    if edge_color_matrix is not None:
        edge_color_arr, _ = load_edge_color_matrix(
            edge_color_matrix, n_expected_nodes=conn_matrix.shape[0]
        )
        print(
            f"Loaded edge color matrix: "
            f"{int(np.sum(edge_color_arr != ''))} colored cells"
        )

    # Get number of nodes
    n_nodes = len(roi_coords_df)

    # Convert node_size to array
    node_sizes = convert_node_size_input(node_size, n_nodes, default_size=8.0)

    # Uniform multiplier applied AFTER all other size resolution. Mirrors
    # edge_width_scale on the edge side. node_size_scale=1.0 is a no-op.
    if node_size_scale != 1.0:
        node_sizes = node_sizes.astype(float) * float(node_size_scale)

    print(f"Node sizes: min={node_sizes.min():.1f}, max={node_sizes.max():.1f}")

    # Convert node_color to appropriate format
    node_colors, module_color_map, module_assignments = convert_node_color_input(
        node_color, n_nodes, default_color='purple'
    )
    is_single_color = isinstance(node_colors, str)
    if not is_single_color:
        print(f"Node colors: {len(set(node_colors))} unique colors")

    # Load node metrics if provided
    metrics_df = None
    if node_metrics is not None:
        metrics_df = load_node_metrics(node_metrics, n_expected_nodes=n_nodes)
        print(f"Loaded node metrics with columns: {list(metrics_df.columns)}")

    # Determine edge width scaling
    if isinstance(edge_width, tuple):
        min_edge_width, max_edge_width = edge_width
        scale_edge_width = True
    else:
        min_edge_width = max_edge_width = float(edge_width)
        scale_edge_width = False

    # Apply the uniform multiplier ONCE here so every downstream code path
    # (consolidated edge traces, per-edge color groups, modular rebuild)
    # picks it up automatically without having to multiply again.
    if edge_width_scale != 1.0:
        min_edge_width = float(min_edge_width) * float(edge_width_scale)
        max_edge_width = float(max_edge_width) * float(edge_width_scale)

    # Create NetworkX graph
    G_all = nx.Graph()

    # Add all nodes with valid coordinates
    valid_nodes = 0
    for idx, row in roi_coords_df.iterrows():
        if not pd.isna(row['cog_x']):
            G_all.add_node(idx,
                          pos=[row['cog_x'], row['cog_y'], row['cog_z']],
                          label=row['roi_name'],
                          x=row['cog_x'],
                          y=row['cog_y'],
                          z=row['cog_z'],
                          size=node_sizes[idx] if idx < len(node_sizes) else 8.0)
            valid_nodes += 1

    print(f"Added {valid_nodes} nodes with valid coordinates")

    # Add edges based on connectivity matrix
    edge_count = 0
    all_weights = []
    for i in range(conn_matrix.shape[0]):
        for j in range(i + 1, conn_matrix.shape[1]):
            weight = conn_matrix[i, j]
            if abs(weight) > edge_threshold and weight != 0:
                if i in G_all.nodes() and j in G_all.nodes():
                    # If an edge color matrix is provided, skip cells with
                    # no color (empty/NaN/0). This makes the color matrix
                    # an additional filter on which edges are drawn.
                    if edge_color_arr is not None and edge_color_arr[i, j] == "":
                        continue
                    edge_attrs = {'weight': weight}
                    if pvalue_lookup is not None:
                        edge_attrs['pvalue'] = float(pvalue_lookup[i, j])
                    if edge_color_arr is not None:
                        edge_attrs['color'] = str(edge_color_arr[i, j])
                    G_all.add_edge(i, j, **edge_attrs)
                    all_weights.append(weight)
                    edge_count += 1

    all_weights = np.array(all_weights) if all_weights else np.array([0])
    print(f"Added {edge_count} edges above threshold {edge_threshold}")

    # Classify nodes by their edge types
    nodes_with_pos_only = set()
    nodes_with_neg_only = set()
    nodes_with_both = set()

    for node in G_all.nodes():
        has_pos = False
        has_neg = False
        for neighbor in G_all.neighbors(node):
            weight = G_all[node][neighbor]['weight']
            if weight > 0:
                has_pos = True
            elif weight < 0:
                has_neg = True

        if has_pos and has_neg:
            nodes_with_both.add(node)
        elif has_pos:
            nodes_with_pos_only.add(node)
        elif has_neg:
            nodes_with_neg_only.add(node)

    print(f"Node classification: {len(nodes_with_pos_only)} pos-only, "
          f"{len(nodes_with_neg_only)} neg-only, {len(nodes_with_both)} both")

    # Create connected-only graph
    G_connected = G_all.copy()
    isolated_nodes = list(nx.isolates(G_connected))
    G_connected.remove_nodes_from(isolated_nodes)

    # Helper to build hover text for a node
    def build_node_hover(node_idx, label):
        hover_parts = [f"<b>{label}</b>"]
        if metrics_df is not None and node_idx < len(metrics_df):
            row = metrics_df.iloc[node_idx]
            for col in metrics_df.columns:
                if col not in ['roi_name', 'roi_index', 'node_idx']:
                    val = row[col]
                    if isinstance(val, (int, float)):
                        hover_parts.append(f"{col}: {val:.4f}" if isinstance(val, float) else f"{col}: {val}")
                    else:
                        hover_parts.append(f"{col}: {val}")
        return "<br>".join(hover_parts)

    # Prepare edges with variable width and hover information
    def prepare_edges_with_width(G):
        """Prepare edge traces with scaled widths."""
        pos_traces = []  # List of (x, y, z, width, hover) tuples for each positive edge
        neg_traces = []  # List of (x, y, z, width, hover) tuples for each negative edge

        for edge in G.edges(data=True):
            i, j, data = edge
            weight = data['weight']
            node_i = G.nodes[i]
            node_j = G.nodes[j]

            # Calculate edge width
            if scale_edge_width:
                edge_w = calculate_edge_width(weight, all_weights, min_edge_width, max_edge_width)
            else:
                edge_w = min_edge_width

            # Hover text — show the original p-value when in pvalue mode,
            # otherwise the raw weight.
            if 'pvalue' in data:
                hover_text = (
                    f"{node_i['label']} <-> {node_j['label']}<br>"
                    f"p-value: {data['pvalue']:.4g}<br>"
                    f"-log10(p): {abs(weight):.3f}"
                )
            else:
                hover_text = (
                    f"{node_i['label']} <-> {node_j['label']}<br>"
                    f"Strength: {weight:.4f}"
                )

            edge_data = {
                'x': [node_i['x'], node_j['x']],
                'y': [node_i['y'], node_j['y']],
                'z': [node_i['z'], node_j['z']],
                'width': edge_w,
                'hover': hover_text,
                'color': data.get('color'),  # may be None
            }

            if weight > 0:
                pos_traces.append(edge_data)
            else:
                neg_traces.append(edge_data)

        return pos_traces, neg_traces

    pos_edges, neg_edges = prepare_edges_with_width(G_all)

    # Create figure
    fig = go.Figure()

    # Resolve mesh lighting (preset + raw knob overrides). When the user
    # passed nothing this returns the legacy default so the figure looks
    # identical to previous releases.
    resolved_lighting = _resolve_mesh_lighting(
        style=mesh_style,
        ambient=mesh_ambient,
        diffuse=mesh_diffuse,
        specular=mesh_specular,
        roughness=mesh_roughness,
        fresnel=mesh_fresnel,
        fast_render=fast_render,
    )

    # Build the Mesh3d kwargs piecemeal so 'lighting' / 'lightposition'
    # are only present when actually set -- otherwise plotly should fall
    # back to its own defaults.
    mesh_kwargs = dict(
        x=vertices[:, 0],
        y=vertices[:, 1],
        z=vertices[:, 2],
        i=faces[:, 0],
        j=faces[:, 1],
        k=faces[:, 2],
        opacity=mesh_opacity,
        color=mesh_color,
        name='Brain Surface',
        showlegend=False,
        hoverinfo='skip',
    )
    if resolved_lighting is not None:
        mesh_kwargs['lighting'] = resolved_lighting
    if mesh_light_position is not None:
        lpx, lpy, lpz = mesh_light_position
        mesh_kwargs['lightposition'] = dict(
            x=float(lpx), y=float(lpy), z=float(lpz)
        )

    fig.add_trace(go.Mesh3d(**mesh_kwargs))

    # ----- Edge trace emission ---------------------------------------
    # When an edge color matrix is supplied, group edges by their cell
    # color (one trace per unique color) so per-edge colors are preserved
    # while still keeping the trace count reasonable. Otherwise fall back
    # to the legacy behavior of one consolidated 'positive' trace and one
    # consolidated 'negative' trace.
    if edge_color_arr is not None:
        from collections import defaultdict
        color_groups: Dict[str, List[Dict]] = defaultdict(list)
        for edge in pos_edges + neg_edges:
            c = edge.get('color') or 'gray'
            color_groups[c].append(edge)

        for c_idx, (color, group) in enumerate(sorted(color_groups.items())):
            gx, gy, gz, ghover = [], [], [], []
            avg_w = float(np.mean([e['width'] for e in group])) if group else min_edge_width
            for edge in group:
                gx.extend(edge['x'] + [None])
                gy.extend(edge['y'] + [None])
                gz.extend(edge['z'] + [None])
                ghover.extend([edge['hover'], edge['hover'], ''])
            fig.add_trace(go.Scatter3d(
                x=gx, y=gy, z=gz,
                mode='lines',
                line=dict(color=color, width=avg_w),
                opacity=0.7,
                hoverinfo='text',
                hovertext=ghover,
                showlegend=True,
                visible=True,
                name=f'Edges {color} ({len(group)})',
                legendgroup=f'edge_color_{c_idx}',
            ))
    else:
        # Add positive edges - consolidated into single trace with average width for legend
        if pos_edges:
            # Consolidate all positive edges into one trace
            pos_x, pos_y, pos_z, pos_hover = [], [], [], []
            avg_pos_width = np.mean([e['width'] for e in pos_edges])
            for edge in pos_edges:
                pos_x.extend(edge['x'] + [None])
                pos_y.extend(edge['y'] + [None])
                pos_z.extend(edge['z'] + [None])
                pos_hover.extend([edge['hover'], edge['hover'], ''])

            pos_label = (
                f'Significant edges ({len(pos_edges)})'
                if matrix_type == 'pvalue' and not neg_edges
                else f'Positive Edges ({len(pos_edges)})'
            )
            fig.add_trace(go.Scatter3d(
                x=pos_x,
                y=pos_y,
                z=pos_z,
                mode='lines',
                line=dict(color=pos_edge_color, width=avg_pos_width),
                opacity=0.6,
                hoverinfo='text',
                hovertext=pos_hover,
                showlegend=True,
                visible=True,
                name=pos_label,
                legendgroup='pos_edges'
            ))

        # Add negative edges - consolidated into single trace
        if neg_edges:
            neg_x, neg_y, neg_z, neg_hover = [], [], [], []
            avg_neg_width = np.mean([e['width'] for e in neg_edges])
            for edge in neg_edges:
                neg_x.extend(edge['x'] + [None])
                neg_y.extend(edge['y'] + [None])
                neg_z.extend(edge['z'] + [None])
                neg_hover.extend([edge['hover'], edge['hover'], ''])

            fig.add_trace(go.Scatter3d(
                x=neg_x,
                y=neg_y,
                z=neg_z,
                mode='lines',
                line=dict(color=neg_edge_color, width=avg_neg_width),
                opacity=0.6,
                hoverinfo='text',
                hovertext=neg_hover,
                showlegend=True,
                visible=True,
                name=f'Negative Edges ({len(neg_edges)})',
                legendgroup='neg_edges'
            ))

    # Determine which nodes to show based on show_only_connected_nodes
    if show_only_connected_nodes:
        active_nodes = list(G_connected.nodes())
    else:
        active_nodes = list(G_all.nodes())

    # If hide_nodes_with_hidden_edges is True, we need to create separate node traces
    # for nodes linked to positive edges and nodes linked to negative edges.
    # Nodes with BOTH edge types are added to BOTH traces so they hide when
    # BOTH edge types are hidden (they overlap when both are visible, which is fine).
    if hide_nodes_with_hidden_edges and show_only_connected_nodes:
        # Nodes with ONLY positive edges OR nodes with both (linked to pos_edges)
        pos_linked_nodes = [n for n in active_nodes if n in nodes_with_pos_only or n in nodes_with_both]
        # Nodes with ONLY negative edges OR nodes with both (linked to neg_edges)
        neg_linked_nodes = [n for n in active_nodes if n in nodes_with_neg_only or n in nodes_with_both]

        # Add nodes linked to positive edges (includes nodes with both edge types)
        if pos_linked_nodes:
            node_x = [G_all.nodes[n]['x'] for n in pos_linked_nodes]
            node_y = [G_all.nodes[n]['y'] for n in pos_linked_nodes]
            node_z = [G_all.nodes[n]['z'] for n in pos_linked_nodes]
            node_labels = [G_all.nodes[n]['label'] for n in pos_linked_nodes]
            node_sizes_pos = [G_all.nodes[n]['size'] for n in pos_linked_nodes]
            node_hovers = [build_node_hover(n, G_all.nodes[n]['label']) for n in pos_linked_nodes]

            # Get colors for these nodes
            if is_single_color:
                pos_node_colors = node_colors
            else:
                pos_node_colors = [node_colors[n] for n in pos_linked_nodes]

            fig.add_trace(go.Scatter3d(
                x=node_x, y=node_y, z=node_z,
                mode='markers+text',
                marker=dict(
                    size=node_sizes_pos,
                    color=pos_node_colors,
                    opacity=0.9,
                    line=dict(color=node_border_color, width=1)
                ),
                text=node_labels,
                textposition='top center',
                textfont=dict(size=label_font_size, color='black', family='Arial'),
                hoverinfo='text',
                hovertext=node_hovers,
                showlegend=False,
                visible=True,
                name='nodes_pos_linked',
                legendgroup='pos_edges'  # Linked to positive edges
            ))

        # Add nodes linked to negative edges (includes nodes with both edge types)
        if neg_linked_nodes:
            node_x = [G_all.nodes[n]['x'] for n in neg_linked_nodes]
            node_y = [G_all.nodes[n]['y'] for n in neg_linked_nodes]
            node_z = [G_all.nodes[n]['z'] for n in neg_linked_nodes]
            node_labels = [G_all.nodes[n]['label'] for n in neg_linked_nodes]
            node_sizes_neg = [G_all.nodes[n]['size'] for n in neg_linked_nodes]
            node_hovers = [build_node_hover(n, G_all.nodes[n]['label']) for n in neg_linked_nodes]

            # Get colors for these nodes
            if is_single_color:
                neg_node_colors = node_colors
            else:
                neg_node_colors = [node_colors[n] for n in neg_linked_nodes]

            fig.add_trace(go.Scatter3d(
                x=node_x, y=node_y, z=node_z,
                mode='markers+text',
                marker=dict(
                    size=node_sizes_neg,
                    color=neg_node_colors,
                    opacity=0.9,
                    line=dict(color=node_border_color, width=1)
                ),
                text=node_labels,
                textposition='top center',
                textfont=dict(size=label_font_size, color='black', family='Arial'),
                hoverinfo='text',
                hovertext=node_hovers,
                showlegend=False,
                visible=True,
                name='nodes_neg_linked',
                legendgroup='neg_edges'  # Linked to negative edges
            ))
    else:
        # Simple mode - all nodes in one trace
        node_x = [G_all.nodes[n]['x'] for n in active_nodes]
        node_y = [G_all.nodes[n]['y'] for n in active_nodes]
        node_z = [G_all.nodes[n]['z'] for n in active_nodes]
        node_labels = [G_all.nodes[n]['label'] for n in active_nodes]
        node_sizes_all = [G_all.nodes[n]['size'] for n in active_nodes]
        node_hovers = [build_node_hover(n, G_all.nodes[n]['label']) for n in active_nodes]

        # Get colors for these nodes
        if is_single_color:
            all_node_colors = node_colors
        else:
            all_node_colors = [node_colors[n] for n in active_nodes]

        fig.add_trace(go.Scatter3d(
            x=node_x, y=node_y, z=node_z,
            mode='markers+text',
            marker=dict(
                size=node_sizes_all,
                color=all_node_colors,
                opacity=0.9,
                line=dict(color=node_border_color, width=1)
            ),
            text=node_labels,
            textposition='top center',
            textfont=dict(size=label_font_size, color='black', family='Arial'),
            hoverinfo='text',
            hovertext=node_hovers,
            showlegend=False,
            visible=True,
            name='nodes_all'
        ))

    # Set camera position. When the caller passed an explicit custom
    # camera dict, we make sure it carries a 'name' field so the title
    # and dropdown both have a sensible label to show.
    if custom_camera is not None:
        camera = dict(custom_camera)  # don't mutate caller's dict
        if 'name' not in camera:
            camera['name'] = custom_camera_name or 'Custom View'
    else:
        camera = CameraController.get_camera_position(camera_view)

    # Add camera view to title
    if 'name' in camera:
        full_title = f"{plot_title}<br><i>View: {camera['name']}</i>"
    else:
        full_title = plot_title

    # Camera control instructions
    camera_control_text = ""
    if enable_camera_controls:
        camera_control_text = (
            "<b>Camera Controls:</b><br>"
            "* Drag to rotate<br>"
            "* Scroll to zoom<br>"
            "* Right-click drag to pan<br>"
            "* Double-click to reset"
        )

    # Build camera preset buttons if enabled
    updatemenus = []
    active_button_idx = 0  # Default to first button
    if enable_camera_controls:
        camera_buttons = []
        view_keys = [k for k in CameraController.PRESET_VIEWS.keys() if k != 'custom']

        for idx, view_key in enumerate(view_keys):
            view_data = CameraController.PRESET_VIEWS[view_key]
            # Track which button should be active based on initial camera_view
            if view_key == camera_view:
                active_button_idx = idx

            # Update both camera AND title subtitle
            new_title = f"{plot_title}<br><i>View: {view_data['name']}</i>"
            camera_buttons.append(
                dict(
                    args=[{
                        'scene.camera.eye.x': view_data['eye']['x'],
                        'scene.camera.eye.y': view_data['eye']['y'],
                        'scene.camera.eye.z': view_data['eye']['z'],
                        'scene.camera.center.x': view_data['center']['x'],
                        'scene.camera.center.y': view_data['center']['y'],
                        'scene.camera.center.z': view_data['center']['z'],
                        'scene.camera.up.x': view_data['up']['x'],
                        'scene.camera.up.y': view_data['up']['y'],
                        'scene.camera.up.z': view_data['up']['z'],
                        'title.text': new_title
                    }],
                    label=view_data['name'],
                    method='relayout'
                )
            )

        # Append the user's custom view as the LAST button in the
        # dropdown and make it the active one. This way the dropdown
        # opens on "Oblique View" / etc. by default but the moment the
        # user provides a custom camera the dropdown surfaces it as the
        # active selection.
        if custom_camera is not None:
            custom_label = camera['name']  # already populated above
            new_title_custom = f"{plot_title}<br><i>View: {custom_label}</i>"
            camera_buttons.append(
                dict(
                    args=[{
                        'scene.camera.eye.x': camera['eye']['x'],
                        'scene.camera.eye.y': camera['eye']['y'],
                        'scene.camera.eye.z': camera['eye']['z'],
                        'scene.camera.center.x': camera['center']['x'],
                        'scene.camera.center.y': camera['center']['y'],
                        'scene.camera.center.z': camera['center']['z'],
                        'scene.camera.up.x': camera['up']['x'],
                        'scene.camera.up.y': camera['up']['y'],
                        'scene.camera.up.z': camera['up']['z'],
                        'title.text': new_title_custom,
                    }],
                    label=custom_label,
                    method='relayout',
                )
            )
            active_button_idx = len(camera_buttons) - 1

        updatemenus = [
            dict(
                type='dropdown',
                showactive=True,
                active=active_button_idx,
                buttons=camera_buttons,
                x=0.01,
                xanchor='left',
                y=0.99,
                yanchor='top',
                bgcolor='rgba(255, 255, 255, 0.9)',
                bordercolor='black',
                borderwidth=1,
                font=dict(size=11)
            )
        ]

    # Update layout with camera controls
    fig.update_layout(
        scene=dict(
            xaxis=dict(showgrid=False, zeroline=False, visible=False),
            yaxis=dict(showgrid=False, zeroline=False, visible=False),
            zaxis=dict(showgrid=False, zeroline=False, visible=False),
            bgcolor='white',
            camera=dict(
                eye=camera['eye'],
                center=camera['center'],
                up=camera['up']
            ),
            dragmode='orbit',
            aspectmode='data'
        ),
        width=1200,
        height=900,
        title={
            'text': full_title,
            'x': 0.5,
            'xanchor': 'center',
            'font': dict(size=20)
        },
        showlegend=True,
        legend=dict(
            x=0.01,
            y=0.85,
            bgcolor='rgba(255,255,255,0.9)',
            bordercolor='black',
            borderwidth=1
        ),
        updatemenus=updatemenus,
        annotations=[
            dict(
                text=camera_control_text,
                showarrow=False,
                xref="paper", yref="paper",
                x=0.01, y=0.50,
                xanchor="left",
                yanchor="top",
                font=dict(size=10),
                bgcolor='rgba(255,255,255,0.9)' if camera_control_text else 'rgba(255,255,255,0)',
                bordercolor='black' if camera_control_text else 'rgba(0,0,0,0)',
                borderwidth=1 if camera_control_text else 0,
                borderpad=4
            ),
            dict(
                text=f"V2: Camera controls enabled" if enable_camera_controls else "",
                showarrow=False,
                xref="paper", yref="paper",
                x=0.99, y=0.01,
                xanchor="right",
                yanchor="bottom",
                font=dict(size=11, color='gray')
            )
        ]
    )

    # Save the figure
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    config = {
        'displayModeBar': True,
        'displaylogo': False,
        'toImageButtonOptions': {
            'format': 'png',
            'filename': save_path.stem,
            'height': 900,
            'width': 1200,
            'scale': 2
        }
    }

    if fast_render:
        config['staticPlot'] = False
        config['scrollZoom'] = True

    # ----- Optional size + width "legend keys" --------------------
    # Pure paper-coordinate overlay (shapes + annotations) so it
    # renders identically in HTML and in static PNG/SVG/PDF exports.
    # Both are auto-skipped for the trivial cases (scalar size /
    # fixed width) where the key would be pointless.
    is_size_vector = not isinstance(node_size, (int, float))
    if not is_size_vector and isinstance(node_sizes, np.ndarray):
        # convert_node_size_input may have produced a constant array
        # from a scalar input -- detect that case so we don't draw a
        # 5-entry legend whose entries are all the same.
        if np.allclose(node_sizes, node_sizes[0]):
            is_size_vector = False
        else:
            is_size_vector = True

    size_for_legend = node_sizes if (show_size_legend and is_size_vector) else None

    # Resolve the size-legend metric column to a value array.
    size_legend_values = None
    size_legend_title = "Node size"
    if size_for_legend is not None and node_size_legend_metric:
        if metrics_df is None:
            print(
                f"Warning: --node-size-legend-metric={node_size_legend_metric!r} "
                f"was passed but no node_metrics file was provided; falling "
                f"back to literal pixel-size labels."
            )
        elif node_size_legend_metric not in metrics_df.columns:
            print(
                f"Warning: --node-size-legend-metric={node_size_legend_metric!r} "
                f"is not a column of node_metrics. Available columns: "
                f"{list(metrics_df.columns)}. Falling back to literal pixel-size labels."
            )
        else:
            size_legend_values = metrics_df[node_size_legend_metric].values
            size_legend_title = node_size_legend_metric

    width_for_legend = None
    width_legend_values = None
    width_legend_title = "Edge weight"
    if show_width_legend and scale_edge_width and len(all_weights) >= 2:
        # Use the absolute weight magnitudes (these are what drove the widths).
        width_for_legend = np.abs(np.asarray(all_weights, dtype=float))
        if matrix_type == 'pvalue' and pvalue_lookup is not None:
            # Pull the original p-values for the same set of edges out of
            # pvalue_lookup. Edges live in the upper triangle so we walk
            # the matrix once.
            edge_pvals = []
            for i in range(conn_matrix.shape[0]):
                for j in range(i + 1, conn_matrix.shape[1]):
                    if (
                        conn_matrix[i, j] != 0
                        and abs(conn_matrix[i, j]) > edge_threshold
                    ):
                        edge_pvals.append(float(pvalue_lookup[i, j]))
            if edge_pvals:
                width_legend_values = np.asarray(edge_pvals)
                width_legend_title = "p-value"

    if size_for_legend is not None or width_for_legend is not None:
        _add_size_width_legend(
            fig,
            node_sizes=size_for_legend,
            node_size_legend_title=size_legend_title,
            node_size_legend_values=size_legend_values,
            edge_widths=width_for_legend,
            edge_width_legend_title=width_legend_title,
            edge_width_legend_values=width_legend_values,
            edge_color=pos_edge_color,
        )

    # Optional inline JS that adds a live camera-position overlay to the
    # HTML. Off by default; only injected when the caller explicitly
    # opts in via show_camera_readout=True. Static image exports
    # (kaleido) are unaffected -- they don't run JS.
    write_html_kwargs = dict(config=config)
    if show_camera_readout:
        write_html_kwargs['post_script'] = _build_camera_readout_js()

    fig.write_html(save_path, **write_html_kwargs)

    print(f"Saved interactive visualization to: {save_path}")

    # Export static image if requested. The single-image vs multi-view
    # branching lives inside the helper so the modular plot can call the
    # same code path AFTER it has rebuilt traces per-module.
    _export_figure_static(
        fig,
        export_image=export_image,
        multi_view=multi_view,
        multi_view_panel_size=multi_view_panel_size,
        multi_view_panel_labels=multi_view_panel_labels,
        multi_view_keep_first_legend=multi_view_keep_first_legend,
        multi_view_zoom=multi_view_zoom,
        image_dpi=image_dpi,
        image_format=image_format,
        plot_title=plot_title,
        export_show_title=export_show_title,
        export_show_legend=export_show_legend,
    )

    # Calculate graph statistics
    graph_stats = {
        'total_nodes': G_all.number_of_nodes(),
        'total_edges': G_all.number_of_edges(),
        'connected_nodes': G_connected.number_of_nodes(),
        'isolated_nodes': len(isolated_nodes),
        'positive_edges': len(pos_edges),
        'negative_edges': len(neg_edges),
        'nodes_with_pos_only': len(nodes_with_pos_only),
        'nodes_with_neg_only': len(nodes_with_neg_only),
        'nodes_with_both': len(nodes_with_both),
        'network_density': nx.density(G_all),
        'average_degree': np.mean([d for n, d in G_all.degree()]) if G_all.number_of_nodes() > 0 else 0,
        'edge_width_scaled': scale_edge_width,
        'node_size_is_vector': not isinstance(node_size, (int, float)),
        'node_color_is_vector': not is_single_color,
        'module_color_map': module_color_map
    }

    if G_all.number_of_nodes() > 0:
        degree_dict = dict(G_all.degree())
        sorted_nodes = sorted(degree_dict.items(), key=lambda x: x[1], reverse=True)[:10]
        graph_stats['top_connected_nodes'] = [(G_all.nodes[node_id]['label'], degree)
                                              for node_id, degree in sorted_nodes]

    return fig, graph_stats


def quick_brain_plot(vertices, faces, roi_coords_df, connectivity_matrix,
                     title="Brain Network", save_name="brain_plot.html"):
    """
    Quick plotting function with default parameters.

    Parameters
    ----------
    vertices : numpy.ndarray
        Mesh vertices
    faces : numpy.ndarray
        Mesh faces
    roi_coords_df : pandas.DataFrame
        ROI coordinates dataframe
    connectivity_matrix : numpy.ndarray
        Connectivity matrix
    title : str, optional
        Plot title
    save_name : str, optional
        Save filename

    Returns
    -------
    fig : plotly.graph_objects.Figure
        The plotly figure
    stats : dict
        Graph statistics
    """
    return create_brain_connectivity_plot(
        vertices=vertices,
        faces=faces,
        roi_coords_df=roi_coords_df,
        connectivity_matrix=connectivity_matrix,
        plot_title=title,
        save_path=save_name
    )


def create_brain_connectivity_plot_with_modularity(
    vertices: np.ndarray,
    faces: np.ndarray,
    roi_coords_df: pd.DataFrame,
    connectivity_matrix: Union[np.ndarray, str, pd.DataFrame],
    module_assignments: Union[np.ndarray, pd.Series, pd.DataFrame, List, str],
    plot_title: str = "Brain Connectivity with Modularity",
    save_path: str = "brain_modularity.html",
    Q_score: Optional[float] = None,
    Z_score: Optional[float] = None,
    node_size: Union[int, float, np.ndarray, pd.Series, pd.DataFrame, List, Dict, str] = 8,
    node_border_color: str = 'darkgray',
    edge_color_mode: str = 'module',
    pos_edge_color: str = 'red',
    neg_edge_color: str = 'blue',
    edge_width: Union[float, Tuple[float, float]] = (1.0, 5.0),
    edge_width_scale: float = 1.0,
    node_size_scale: float = 1.0,
    edge_threshold: float = 0.0,
    mesh_color: str = 'lightgray',
    mesh_opacity: float = 0.15,
    label_font_size: int = 8,
    fast_render: bool = False,
    camera_view: str = 'oblique',
    custom_camera: Optional[Dict] = None,
    enable_camera_controls: bool = True,
    show_only_connected_nodes: bool = True,
    node_metrics: Optional[Union[str, pd.DataFrame]] = None,
    hide_nodes_with_hidden_edges: bool = True,
    export_image: Optional[str] = None,
    image_format: str = 'png',
    image_dpi: int = 300,
    export_show_title: bool = True,
    export_show_legend: bool = True,
    edge_color_matrix: Optional[Union[str, np.ndarray, pd.DataFrame]] = None,
    matrix_type: str = 'weight',
    pvalue_threshold: float = 0.05,
    sign_matrix: Optional[Union[str, np.ndarray, pd.DataFrame]] = None,
    mesh_style: Optional[str] = None,
    mesh_ambient: Optional[float] = None,
    mesh_diffuse: Optional[float] = None,
    mesh_specular: Optional[float] = None,
    mesh_roughness: Optional[float] = None,
    mesh_fresnel: Optional[float] = None,
    mesh_light_position: Optional[Tuple[float, float, float]] = None,
    custom_camera_name: Optional[str] = None,
    show_camera_readout: bool = False,
    show_size_legend: bool = True,
    show_width_legend: bool = True,
    node_size_legend_metric: Optional[str] = None,
    multi_view: Optional[List[Union[str, Dict]]] = None,
    multi_view_panel_size: Tuple[int, int] = (800, 800),
    multi_view_panel_labels: Optional[List[str]] = None,
    multi_view_keep_first_legend: bool = True,
    multi_view_zoom: float = 1.0,
) -> Tuple[go.Figure, Dict]:
    """
    Create brain connectivity visualization with modularity-based node coloring.

    This function colors nodes based on module assignments and includes a
    module legend. It wraps create_brain_connectivity_plot with modularity
    specific features.

    Parameters
    ----------
    vertices : numpy.ndarray
        Mesh vertices array of shape (n_vertices, 3)
    faces : numpy.ndarray
        Mesh faces array of shape (n_faces, 3)
    roi_coords_df : pandas.DataFrame
        DataFrame containing ROI coordinates with columns:
        - 'cog_x', 'cog_y', 'cog_z': world coordinates
        - 'roi_name': name of the ROI
    connectivity_matrix : numpy.ndarray, str, or pd.DataFrame
        Connectivity matrix or path to file
    module_assignments : numpy.ndarray, pd.Series, pd.DataFrame, list, or str
        Module assignment for each ROI (1-indexed). Can be:
        - numpy array of integers
        - pandas Series/DataFrame with 'module' column
        - list of integers
        - str: Path to CSV file with module assignments
    plot_title : str, optional
        Title for the plot
    save_path : str, optional
        Path where to save the HTML file
    Q_score : float, optional
        Modularity Q score to display in title (if provided)
    Z_score : float, optional
        Modularity Z score to display in title (if provided)
    node_size : int, float, array-like, dict, or str, optional
        Size of the ROI nodes
    node_border_color : str, optional
        Border color of the ROI nodes (default 'darkgray')
    edge_color_mode : str, optional
        How to color edges: 'module' (default) colors edges by the module
        of the source node, 'sign' colors by positive/negative
    pos_edge_color : str, optional
        Color for positive connections (only used when edge_color_mode='sign')
    neg_edge_color : str, optional
        Color for negative connections (only used when edge_color_mode='sign')
    edge_width : float or tuple, optional
        Edge width specification (see ``create_brain_connectivity_plot``)
    edge_width_scale : float, optional
        Uniform multiplier applied to every edge width AFTER all other
        scaling. ``1.0`` (default) leaves widths unchanged; ``5.0`` makes
        every edge five times thicker. Forwarded to
        ``create_brain_connectivity_plot``.
    node_size_scale : float, optional
        Uniform multiplier applied to every node size AFTER all other
        size resolution. ``1.0`` (default) leaves sizes unchanged.
        Forwarded to ``create_brain_connectivity_plot``. The legend's
        sample dots also scale so they continue to match the brain dots.
    edge_threshold : float, optional
        Threshold for showing edges
    mesh_color : str, optional
        Color of the brain mesh
    mesh_opacity : float, optional
        Opacity of the brain mesh
    label_font_size : int, optional
        Font size for ROI labels
    fast_render : bool, optional
        If True, uses optimizations for faster rendering
    camera_view : str, optional
        Camera view preset name
    custom_camera : dict, optional
        Custom camera position
    enable_camera_controls : bool, optional
        Whether to enable camera view dropdown controls
    show_only_connected_nodes : bool, optional
        If True, only show nodes with at least one edge
    node_metrics : str or pd.DataFrame, optional
        Node metrics for hover display
    hide_nodes_with_hidden_edges : bool, optional
        If True, nodes will be hidden when their edge types are toggled off
    export_image : str, optional
        Path to export static image
    image_format : str, optional
        Image format if export_image doesn't have extension
    image_dpi : int, optional
        DPI for exported images
    export_show_title : bool, optional
        Whether to show the title in exported images
    export_show_legend : bool, optional
        Whether to show the legend in exported images
    edge_color_matrix : str, np.ndarray, or pd.DataFrame, optional
        Per-edge color matrix forwarded to
        :func:`create_brain_connectivity_plot`. Same shape as
        ``connectivity_matrix``; cells may be color strings, integer
        labels, or empty/0/NaN to skip the edge. When set, this overrides
        both the per-module edge coloring and the pos/neg sign coloring.
    matrix_type : str, optional
        Either ``'weight'`` (default) or ``'pvalue'``. When ``'pvalue'``,
        ``connectivity_matrix`` is interpreted as a matrix of p-values
        and transformed via ``-log10(p)`` so that smaller p-values draw
        thicker edges. Hover text shows the original p-value. See
        ``pvalue_threshold`` and ``sign_matrix``.
    pvalue_threshold : float, optional
        Only used when ``matrix_type='pvalue'``. Edges with
        ``p > pvalue_threshold`` are dropped. Default ``0.05``.
    sign_matrix : str, np.ndarray, or pd.DataFrame, optional
        Only used when ``matrix_type='pvalue'``. Same-shape matrix whose
        sign indicates the sign of the underlying effect; positive
        effects render with ``pos_edge_color`` and negative effects with
        ``neg_edge_color``.
    mesh_style : str, optional
        Brain mesh lighting preset. Forwarded to
        ``create_brain_connectivity_plot``. See its docstring for the
        full list (``flat`` / ``matte`` / ``smooth`` / ``glossy`` /
        ``mirror``). ``None`` (default) preserves legacy behavior.
    mesh_ambient, mesh_diffuse, mesh_specular, mesh_roughness, mesh_fresnel : float, optional
        Per-knob overrides of the mesh ``lighting`` dict, applied on top
        of ``mesh_style``. See ``create_brain_connectivity_plot`` for the
        accepted ranges and effects.
    mesh_light_position : tuple of (x, y, z), optional
        Position of the directional light in 3D space. ``None`` (default)
        uses plotly's default light position.
    custom_camera_name : str, optional
        Display name for the custom camera. Forwarded to
        ``create_brain_connectivity_plot``. Shows up in the title and as
        a new entry in the camera dropdown.
    show_camera_readout : bool, optional
        Inject the live camera-position overlay into the saved HTML.
        Off by default. Set to ``False`` (or omit) to keep the saved
        HTML clean. See ``create_brain_connectivity_plot`` for details.
    show_size_legend, show_width_legend : bool, optional
        Render the node-size / edge-width legend keys near the bottom of
        the plot. Forwarded to ``create_brain_connectivity_plot``. See
        its docstring for the auto-skip rules.
    node_size_legend_metric : str, optional
        Column of ``node_metrics`` to use as the source of LABELS in the
        node-size key (instead of the literal pixel sizes). Forwarded to
        ``create_brain_connectivity_plot``.
    multi_view : list of (str | dict), optional
        Render a stitched 1xN PNG strip of the brain from multiple
        camera angles. Forwarded to ``create_brain_connectivity_plot``.
        ``export_image`` is reinterpreted as the stitched output path.
    multi_view_panel_size : tuple of (int, int), optional
        Pixel size of each panel before DPI scaling. Default
        ``(800, 800)``.
    multi_view_panel_labels : list of str, optional
        Per-panel labels drawn under each panel of the stitched strip.
    multi_view_keep_first_legend : bool, optional
        When ``True`` (default), the first panel keeps its plotly legend
        and the rest are rendered without one.
    multi_view_zoom : float, optional
        Camera zoom multiplier applied uniformly to every panel of the
        stitched strip. ``1.0`` (default) = no change. Higher values
        bring the camera closer (brain looks bigger).

    Returns
    -------
    fig : plotly.graph_objects.Figure
        The plotly figure object
    graph_stats : dict
        Dictionary containing graph statistics and module information
    """

    print(f"Creating modularity visualization: {plot_title}")

    # Get number of nodes
    n_nodes = len(roi_coords_df)

    # Process module assignments
    module_arr = None

    if isinstance(module_assignments, str):
        # File path
        path = Path(module_assignments)
        if not path.exists():
            raise FileNotFoundError(f"Module assignments file not found: {module_assignments}")
        df = pd.read_csv(module_assignments)
        if 'module' in df.columns:
            module_arr = df['module'].values
        else:
            # Use last column (assuming it's the module column)
            module_arr = df.iloc[:, -1].values
    elif isinstance(module_assignments, pd.DataFrame):
        if 'module' in module_assignments.columns:
            module_arr = module_assignments['module'].values
        else:
            numeric_cols = module_assignments.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                module_arr = module_assignments[numeric_cols[0]].values
            else:
                module_arr = module_assignments.iloc[:, 0].values
    elif isinstance(module_assignments, pd.Series):
        module_arr = module_assignments.values
    elif isinstance(module_assignments, np.ndarray):
        module_arr = module_assignments.flatten()
    elif isinstance(module_assignments, list):
        module_arr = np.array(module_assignments)
    else:
        raise TypeError(
            f"Unsupported module_assignments type: {type(module_assignments)}. "
            f"Expected np.ndarray, pd.Series, pd.DataFrame, list, or file path."
        )

    # Ensure integer type
    module_arr = module_arr.astype(int)

    # Validate length
    if len(module_arr) != n_nodes:
        raise ValueError(
            f"Module assignments length ({len(module_arr)}) does not match "
            f"number of nodes ({n_nodes})"
        )

    # Build title with Q and Z scores if provided
    full_title = plot_title
    score_parts = []
    if Q_score is not None:
        score_parts.append(f"Q={Q_score:.3f}")
    if Z_score is not None:
        score_parts.append(f"Z={Z_score:.2f}")
    if score_parts:
        full_title = f"{plot_title} ({', '.join(score_parts)})"

    # Validate edge_color_mode
    if edge_color_mode not in ['sign', 'module']:
        raise ValueError(f"edge_color_mode must be 'sign' or 'module', got '{edge_color_mode}'")

    # Generate module colors for edge coloring if needed
    unique_modules = np.unique(module_arr)
    module_colors = generate_module_colors(len(unique_modules))
    module_color_map_internal = {m: module_colors[i] for i, m in enumerate(sorted(unique_modules))}

    # Call the main function with module assignments as node_color
    fig, graph_stats = create_brain_connectivity_plot(
        vertices=vertices,
        faces=faces,
        roi_coords_df=roi_coords_df,
        connectivity_matrix=connectivity_matrix,
        plot_title=full_title,
        save_path=save_path,
        node_size=node_size,
        node_color=module_arr,  # Pass module assignments as node colors
        node_border_color=node_border_color,
        pos_edge_color=pos_edge_color,
        neg_edge_color=neg_edge_color,
        edge_width=edge_width,
        edge_width_scale=edge_width_scale,
        node_size_scale=node_size_scale,
        edge_threshold=edge_threshold,
        mesh_color=mesh_color,
        mesh_opacity=mesh_opacity,
        label_font_size=label_font_size,
        fast_render=fast_render,
        camera_view=camera_view,
        custom_camera=custom_camera,
        enable_camera_controls=enable_camera_controls,
        show_only_connected_nodes=show_only_connected_nodes,
        node_metrics=node_metrics,
        hide_nodes_with_hidden_edges=hide_nodes_with_hidden_edges,
        # Suppress export from the inner call: the inner figure has
        # sign-mode edges, but the outer function rebuilds them per
        # module below. The export must happen on the rebuilt figure,
        # so we run it ourselves at the end of this function.
        export_image=None,
        image_format=image_format,
        image_dpi=image_dpi,
        export_show_title=export_show_title,
        export_show_legend=export_show_legend,
        edge_color_matrix=edge_color_matrix,
        matrix_type=matrix_type,
        pvalue_threshold=pvalue_threshold,
        sign_matrix=sign_matrix,
        mesh_style=mesh_style,
        mesh_ambient=mesh_ambient,
        mesh_diffuse=mesh_diffuse,
        mesh_specular=mesh_specular,
        mesh_roughness=mesh_roughness,
        mesh_fresnel=mesh_fresnel,
        mesh_light_position=mesh_light_position,
        custom_camera_name=custom_camera_name,
        show_camera_readout=show_camera_readout,
        show_size_legend=show_size_legend,
        show_width_legend=show_width_legend,
        node_size_legend_metric=node_size_legend_metric,
        multi_view=None,
        multi_view_panel_size=multi_view_panel_size,
        multi_view_panel_labels=multi_view_panel_labels,
        multi_view_keep_first_legend=multi_view_keep_first_legend,
        multi_view_zoom=multi_view_zoom,
    )

    # ------------------------------------------------------------------
    # Rebuild traces grouped by MODULE so the legend correctly toggles
    # both the module's nodes and the module's edges in one click.
    #
    # The inner call to create_brain_connectivity_plot built node traces
    # in legendgroups 'pos_edges'/'neg_edges' (sign-based) and edge traces
    # in those same groups. That makes clicking a module legend entry a
    # no-op for nodes/edges. Here we strip those traces and rebuild them
    # per-module so that every per-module trace shares
    # legendgroup=f'module_{module_id}' with the visible legend entry.
    # ------------------------------------------------------------------

    # Load connectivity matrix once for rebuilding. Apply the same
    # transforms (p-value -> -log10) and the same edge color matrix
    # filter that the inner create_brain_connectivity_plot used, so the
    # rebuilt per-module traces stay consistent with the originals.
    conn_matrix = load_connectivity_input(connectivity_matrix, n_expected_nodes=n_nodes)

    pvalue_lookup_mod: Optional[np.ndarray] = None
    if matrix_type == 'pvalue':
        sign_arr_mod = None
        if sign_matrix is not None:
            sign_arr_mod = load_connectivity_input(
                sign_matrix, n_expected_nodes=conn_matrix.shape[0]
            )
        conn_matrix, pvalue_lookup_mod = transform_pvalue_matrix(
            conn_matrix,
            pvalue_threshold=pvalue_threshold,
            sign_matrix=sign_arr_mod,
        )

    edge_color_arr_mod: Optional[np.ndarray] = None
    if edge_color_matrix is not None:
        edge_color_arr_mod, _ = load_edge_color_matrix(
            edge_color_matrix, n_expected_nodes=conn_matrix.shape[0]
        )

    # Drop existing edge traces AND existing node traces from the inner
    # connectivity plot. We keep the brain mesh trace (Mesh3d) and any
    # other unrelated traces.
    #
    # Inner node traces are named 'nodes_pos_linked', 'nodes_neg_linked',
    # or 'nodes_all'. Inner edge trace names depend on matrix_type:
    #   - weight mode -> 'Positive Edges (...)' / 'Negative Edges (...)'
    #   - pvalue mode -> 'Significant edges (...)' (note the lowercase
    #     'e' in 'edges')
    # so the substring match must be case-insensitive.
    kept_data = []
    for trace in fig.data:
        trace_name = trace.name if hasattr(trace, 'name') and trace.name else ''
        lower = trace_name.lower()
        if (
            'edges' in lower
            or trace_name.startswith('nodes_')
        ):
            continue
        kept_data.append(trace)
    fig.data = kept_data

    # Compute scaled edge widths over all edges (so widths are comparable
    # across modules). Apply the uniform edge_width_scale multiplier here
    # so the rebuilt per-module traces match what create_brain_connectivity_plot
    # produces internally.
    if isinstance(edge_width, tuple):
        min_ew, max_ew = edge_width
        scale_ew = True
    else:
        min_ew = max_ew = float(edge_width)
        scale_ew = False
    if edge_width_scale != 1.0:
        min_ew = float(min_ew) * float(edge_width_scale)
        max_ew = float(max_ew) * float(edge_width_scale)
    all_weights_full = conn_matrix[conn_matrix != 0]
    if all_weights_full.size == 0:
        all_weights_full = np.array([0.0])

    # For each module: build one trace for its nodes and one trace for its
    # edges (edge endpoint i, lower-index, defines ownership). Both share
    # legendgroup=f'module_{module_id}', and the node trace is the legend
    # entry (showlegend=True).
    sorted_modules = sorted(unique_modules)
    for module_id in sorted_modules:
        module_color = module_color_map_internal[module_id]
        legend_group = f'module_{module_id}'
        module_node_idx = np.where(module_arr == module_id)[0]
        n_in_module = len(module_node_idx)

        # ---------- edges owned by this module ----------
        edge_x, edge_y, edge_z, edge_hover = [], [], [], []
        edge_widths_for_avg = []
        # Per-edge color used only when edge_color_matrix is supplied;
        # in that case we emit one trace per unique color below.
        per_edge_colors: List[str] = []
        # Collect (i, j, weight) of edges where source (lower index) is in module
        for i in range(conn_matrix.shape[0]):
            if module_arr[i] != module_id:
                continue
            for j in range(i + 1, conn_matrix.shape[1]):
                weight = conn_matrix[i, j]
                if abs(weight) <= edge_threshold or weight == 0:
                    continue
                if edge_color_arr_mod is not None and edge_color_arr_mod[i, j] == "":
                    continue
                try:
                    xi = roi_coords_df.loc[i, 'cog_x']
                    yi = roi_coords_df.loc[i, 'cog_y']
                    zi = roi_coords_df.loc[i, 'cog_z']
                    xj = roi_coords_df.loc[j, 'cog_x']
                    yj = roi_coords_df.loc[j, 'cog_y']
                    zj = roi_coords_df.loc[j, 'cog_z']
                except (KeyError, IndexError):
                    continue

                if scale_ew:
                    w = calculate_edge_width(weight, all_weights_full, min_ew, max_ew)
                else:
                    w = min_ew
                edge_widths_for_avg.append(w)

                edge_x.extend([xi, xj, None])
                edge_y.extend([yi, yj, None])
                edge_z.extend([zi, zj, None])
                if pvalue_lookup_mod is not None:
                    hover_text = (
                        f"{roi_coords_df.loc[i, 'roi_name']} (M{module_arr[i]}) "
                        f"<-> {roi_coords_df.loc[j, 'roi_name']} (M{module_arr[j]})<br>"
                        f"p-value: {pvalue_lookup_mod[i, j]:.4g}<br>"
                        f"-log10(p): {abs(weight):.3f}"
                    )
                else:
                    hover_text = (
                        f"{roi_coords_df.loc[i, 'roi_name']} (M{module_arr[i]}) "
                        f"<-> {roi_coords_df.loc[j, 'roi_name']} (M{module_arr[j]})<br>"
                        f"Strength: {weight:.4f}"
                    )
                edge_hover.extend([hover_text, hover_text, ''])
                if edge_color_arr_mod is not None:
                    per_edge_colors.append(str(edge_color_arr_mod[i, j]))

        if edge_x:
            # When the user supplied an explicit edge color matrix, those
            # colors override the module / sign coloring. Group this
            # module's edges by color and emit one sub-trace per color so
            # plotly can render them correctly while still keeping all of
            # them in this module's legendgroup.
            if edge_color_arr_mod is not None:
                from collections import defaultdict
                # Walk per-edge state in lockstep: each edge contributes
                # 3 entries to (edge_x, edge_y, edge_z) and edge_hover.
                grouped_x: Dict[str, List] = defaultdict(list)
                grouped_y: Dict[str, List] = defaultdict(list)
                grouped_z: Dict[str, List] = defaultdict(list)
                grouped_h: Dict[str, List] = defaultdict(list)
                grouped_w: Dict[str, List] = defaultdict(list)
                for k, ec in enumerate(per_edge_colors):
                    sx = edge_x[k * 3:k * 3 + 3]
                    sy = edge_y[k * 3:k * 3 + 3]
                    sz = edge_z[k * 3:k * 3 + 3]
                    sh = edge_hover[k * 3:k * 3 + 3]
                    grouped_x[ec].extend(sx)
                    grouped_y[ec].extend(sy)
                    grouped_z[ec].extend(sz)
                    grouped_h[ec].extend(sh)
                    grouped_w[ec].append(edge_widths_for_avg[k])
                for ec in sorted(grouped_x.keys()):
                    avg_ew = (
                        float(np.mean(grouped_w[ec])) if grouped_w[ec] else max_ew
                    )
                    fig.add_trace(go.Scatter3d(
                        x=grouped_x[ec],
                        y=grouped_y[ec],
                        z=grouped_z[ec],
                        mode='lines',
                        line=dict(color=ec, width=avg_ew),
                        opacity=0.7,
                        hoverinfo='text',
                        hovertext=grouped_h[ec],
                        showlegend=False,
                        name=f'Module {module_id} Edges ({ec})',
                        legendgroup=legend_group,
                    ))
            else:
                # Pick the line color based on edge_color_mode
                if edge_color_mode == 'module':
                    line_color = module_color
                else:
                    # 'sign' mode -- the trace is grouped by module but
                    # each individual edge keeps the pos/neg color of its
                    # weight. Plotly Scatter3d only supports a single
                    # color per trace, so we split into two sub-traces
                    # below instead. We'll handle that case after this
                    # block.
                    line_color = None

                if line_color is not None:
                    avg_w = float(np.mean(edge_widths_for_avg)) if edge_widths_for_avg else max_ew
                    fig.add_trace(go.Scatter3d(
                        x=edge_x, y=edge_y, z=edge_z,
                        mode='lines',
                        line=dict(color=line_color, width=avg_w),
                        opacity=0.6,
                        hoverinfo='text',
                        hovertext=edge_hover,
                        showlegend=False,
                        name=f'Module {module_id} Edges',
                        legendgroup=legend_group,
                    ))

        if edge_color_mode == 'sign' and edge_color_arr_mod is None:
            # Re-walk and split this module's edges into positive and
            # negative sub-traces, both still tagged with the module's
            # legendgroup so they hide together with the module nodes.
            # (Skipped when edge_color_arr_mod is set: those edges were
            # already emitted with their per-cell colors above.)
            pos_x, pos_y, pos_z, pos_hover, pos_widths = [], [], [], [], []
            neg_x, neg_y, neg_z, neg_hover, neg_widths = [], [], [], [], []
            for i in range(conn_matrix.shape[0]):
                if module_arr[i] != module_id:
                    continue
                for j in range(i + 1, conn_matrix.shape[1]):
                    weight = conn_matrix[i, j]
                    if abs(weight) <= edge_threshold or weight == 0:
                        continue
                    try:
                        xi = roi_coords_df.loc[i, 'cog_x']
                        yi = roi_coords_df.loc[i, 'cog_y']
                        zi = roi_coords_df.loc[i, 'cog_z']
                        xj = roi_coords_df.loc[j, 'cog_x']
                        yj = roi_coords_df.loc[j, 'cog_y']
                        zj = roi_coords_df.loc[j, 'cog_z']
                    except (KeyError, IndexError):
                        continue

                    if scale_ew:
                        w = calculate_edge_width(weight, all_weights_full, min_ew, max_ew)
                    else:
                        w = min_ew

                    hover_text = (
                        f"{roi_coords_df.loc[i, 'roi_name']} (M{module_arr[i]}) "
                        f"<-> {roi_coords_df.loc[j, 'roi_name']} (M{module_arr[j]})<br>"
                        f"Strength: {weight:.4f}"
                    )
                    if weight > 0:
                        pos_x.extend([xi, xj, None])
                        pos_y.extend([yi, yj, None])
                        pos_z.extend([zi, zj, None])
                        pos_hover.extend([hover_text, hover_text, ''])
                        pos_widths.append(w)
                    else:
                        neg_x.extend([xi, xj, None])
                        neg_y.extend([yi, yj, None])
                        neg_z.extend([zi, zj, None])
                        neg_hover.extend([hover_text, hover_text, ''])
                        neg_widths.append(w)

            if pos_x:
                fig.add_trace(go.Scatter3d(
                    x=pos_x, y=pos_y, z=pos_z,
                    mode='lines',
                    line=dict(
                        color=pos_edge_color,
                        width=float(np.mean(pos_widths)) if pos_widths else max_ew,
                    ),
                    opacity=0.6,
                    hoverinfo='text',
                    hovertext=pos_hover,
                    showlegend=False,
                    name=f'Module {module_id} (+ edges)',
                    legendgroup=legend_group,
                ))
            if neg_x:
                fig.add_trace(go.Scatter3d(
                    x=neg_x, y=neg_y, z=neg_z,
                    mode='lines',
                    line=dict(
                        color=neg_edge_color,
                        width=float(np.mean(neg_widths)) if neg_widths else max_ew,
                    ),
                    opacity=0.6,
                    hoverinfo='text',
                    hovertext=neg_hover,
                    showlegend=False,
                    name=f'Module {module_id} (- edges)',
                    legendgroup=legend_group,
                ))

        # ---------- nodes belonging to this module ----------
        # Only show nodes that have at least one connection if
        # show_only_connected_nodes is True
        if show_only_connected_nodes:
            connected_mask = (
                np.any(conn_matrix != 0, axis=0) | np.any(conn_matrix != 0, axis=1)
            )
            module_node_visible = [
                i for i in module_node_idx if connected_mask[i]
            ]
        else:
            module_node_visible = list(module_node_idx)

        if module_node_visible:
            node_x = [roi_coords_df.loc[i, 'cog_x'] for i in module_node_visible]
            node_y = [roi_coords_df.loc[i, 'cog_y'] for i in module_node_visible]
            node_z = [roi_coords_df.loc[i, 'cog_z'] for i in module_node_visible]
            node_labels = [roi_coords_df.loc[i, 'roi_name'] for i in module_node_visible]
            # Use the node sizes the inner function computed (stored on the
            # graph in graph_stats? not directly; recompute here). Apply
            # node_size_scale here too so the rebuilt module markers
            # match the scaled sizes the inner call produced.
            node_sizes = convert_node_size_input(node_size, n_nodes, default_size=8.0)
            if node_size_scale != 1.0:
                node_sizes = node_sizes.astype(float) * float(node_size_scale)
            sizes_for_module = [node_sizes[i] for i in module_node_visible]

            fig.add_trace(go.Scatter3d(
                x=node_x, y=node_y, z=node_z,
                mode='markers+text',
                marker=dict(
                    size=sizes_for_module,
                    color=module_color,
                    opacity=0.9,
                    line=dict(color=node_border_color, width=1),
                ),
                text=node_labels,
                textposition='top center',
                textfont=dict(size=label_font_size, color='black', family='Arial'),
                hoverinfo='text',
                hovertext=[f"<b>{nm}</b><br>Module: {int(module_id)}"
                           for nm in node_labels],
                showlegend=True,
                name=f'Module {int(module_id)} ({n_in_module})',
                legendgroup=legend_group,
            ))
        else:
            # No visible nodes for this module — still emit a dummy legend
            # entry so the user can see the module exists in the legend.
            fig.add_trace(go.Scatter3d(
                x=[None], y=[None], z=[None],
                mode='markers',
                marker=dict(size=10, color=module_color),
                name=f'Module {int(module_id)} ({n_in_module})',
                showlegend=True,
                legendgroup=legend_group,
            ))

    graph_stats['edge_color_mode'] = edge_color_mode

    # Add module statistics to graph_stats
    unique_modules = np.unique(module_arr)
    module_sizes = {f'module_{m}': int(np.sum(module_arr == m)) for m in unique_modules}
    graph_stats['module_assignments'] = module_arr
    graph_stats['n_modules'] = len(unique_modules)
    graph_stats['module_sizes'] = module_sizes
    graph_stats['Q_score'] = Q_score
    graph_stats['Z_score'] = Z_score

    # Re-save the figure with the legend. The inner call already wrote
    # the file once via create_brain_connectivity_plot, but we then
    # rebuilt the per-module legend traces and need to overwrite.
    save_path = Path(save_path)
    config = {
        'displayModeBar': True,
        'displaylogo': False,
        'toImageButtonOptions': {
            'format': 'png',
            'filename': save_path.stem,
            'height': 900,
            'width': 1200,
            'scale': 2
        }
    }
    write_html_kwargs = dict(config=config)
    if show_camera_readout:
        write_html_kwargs['post_script'] = _build_camera_readout_js()
    fig.write_html(save_path, **write_html_kwargs)

    # Export static image now that the per-module traces are in place.
    # The inner call was passed export_image=None / multi_view=None, so
    # this is the only export step and it sees the correct figure.
    _export_figure_static(
        fig,
        export_image=export_image,
        multi_view=multi_view,
        multi_view_panel_size=multi_view_panel_size,
        multi_view_panel_labels=multi_view_panel_labels,
        multi_view_keep_first_legend=multi_view_keep_first_legend,
        multi_view_zoom=multi_view_zoom,
        image_dpi=image_dpi,
        image_format=image_format,
        plot_title=full_title,
        export_show_title=export_show_title,
        export_show_legend=export_show_legend,
    )

    return fig, graph_stats
