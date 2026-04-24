"""
Basic brain connectivity plotting commands.
"""

import click
from pathlib import Path
import os

from ..console import console, print_success, print_error, print_warning, print_info, create_stats_table


# Available camera view presets
CAMERA_VIEWS = [
    "oblique", "anterior", "posterior", "left", "right",
    "superior", "inferior", "dorsal", "ventral",
    "lateral-left", "lateral-right"
]


@click.command()
# === Required inputs ===
@click.option("--mesh", "-m", required=True, type=click.Path(exists=True),
              help="Path to brain mesh file. Supported formats: .gii (GIFTI), .obj, .mz3, .ply")
@click.option("--coords", "-c", required=True, type=click.Path(exists=True),
              help="Path to ROI coordinates CSV file. Must contain columns: cog_x, cog_y, cog_z, roi_name")
@click.option("--matrix", "-x", required=True, type=click.Path(exists=True),
              help="Path to connectivity matrix. Supported formats: .npy, .csv, .txt, .mat, .edge")

# === Output options ===
@click.option("--output", "-o", default="brain_connectivity.html", type=click.Path(),
              help="Output HTML file path for interactive visualization. Default: brain_connectivity.html")
@click.option("--title", "-t", default="Brain Connectivity Network",
              help="Plot title displayed at top of visualization.")

# === Node appearance ===
@click.option("--node-size", default="8",
              help="""Node size specification. Accepts:
              - Single number: All nodes same size (e.g., '10')
              - File path: CSV/NPY file with per-node sizes
              - 'pc' or 'zscore': Use metrics column if --node-metrics provided""")
@click.option("--node-color", default="purple",
              help="""Node color specification. Accepts:
              - Color name: 'purple', 'red', 'steelblue'
              - Hex code: '#FF5733'
              - File path: CSV with module assignments (integers 1-N)
              - Module assignments auto-generate distinct colors""")
@click.option("--node-border-color", default="magenta",
              help="Border/outline color for nodes. Default: magenta")
@click.option("--node-size-scale", default=1.0, type=float,
              help=(
                  "Uniform multiplier applied to every node size AFTER "
                  "all other size resolution. Default 1.0 (no change). "
                  "Use this when you've tuned --node-size and just want "
                  "every dot proportionally larger or smaller for a "
                  "particular figure. Examples: --node-size 10 "
                  "--node-size-scale 2 -> every dot at 20 px; "
                  "--node-size sizes.csv --node-size-scale 1.5 -> every "
                  "dot 50% bigger. The size legend's sample dots also "
                  "scale up so they continue to match the brain dots."
              ))

# === Edge appearance ===
@click.option("--pos-edge-color", default="red",
              help="Color for positive connections. Default: red")
@click.option("--neg-edge-color", default="blue",
              help="Color for negative connections. Default: blue")
@click.option("--edge-width-min", default=1.0, type=float,
              help="Minimum edge line width when scaling by weight. Default: 1.0")
@click.option("--edge-width-max", default=5.0, type=float,
              help="Maximum edge line width when scaling by weight. Default: 5.0")
@click.option("--edge-width-fixed", default=None, type=float,
              help="Fixed edge width (disables weight-based scaling). If set, ignores min/max.")
@click.option("--edge-width-scale", default=1.0, type=float,
              help=(
                  "Uniform multiplier applied to every edge width AFTER all other "
                  "scaling. Default 1.0 (no change). Use this when you've tuned "
                  "--edge-width-min/--edge-width-max (or --edge-width-fixed) and "
                  "you just want every edge proportionally thicker/thinner for a "
                  "particular figure. Examples: with --edge-width-min 1 "
                  "--edge-width-max 5 --edge-width-scale 5 the final widths land "
                  "in [5, 25]; with --edge-width-fixed 2 --edge-width-scale 5 "
                  "every edge is drawn at width 10."
              ))
@click.option("--edge-threshold", default=0.0, type=float,
              help="Minimum absolute edge weight to display. Default: 0.0 (show all non-zero)")

# === Brain mesh appearance ===
@click.option("--mesh-color", default="lightgray",
              help="Color of the brain mesh surface. Default: lightgray")
@click.option("--mesh-opacity", default=0.15, type=float,
              help="Opacity of brain mesh (0-1). Lower values show more of the network. Default: 0.15")

# === Brain mesh lighting / shininess ===
@click.option("--mesh-style", default=None,
              type=click.Choice(["default", "flat", "matte", "smooth", "glossy", "mirror"],
                                case_sensitive=False),
              help=(
                  "High-level lighting preset for the brain mesh. 'default' "
                  "(or omitted) preserves the legacy look (no extra lighting). "
                  "'flat' = pure ambient, no shading. 'matte' = chalk-like, "
                  "no specular highlight. 'smooth' = balanced default with a "
                  "small specular and a hint of fresnel rim. 'glossy' = shiny "
                  "plastic look with a sharp specular highlight. 'mirror' = "
                  "near-chrome (max specular, near-zero roughness). Any of "
                  "the --mesh-{ambient,diffuse,specular,roughness,fresnel} "
                  "knobs you also pass override individual values from the "
                  "preset."
              ))
@click.option("--mesh-ambient", default=None, type=float,
              help=(
                  "Mesh lighting.ambient (0-1). Non-directional fill light. "
                  "Higher values make the mesh self-illuminate (flatter, "
                  "fewer shadows). Plotly default is 0.8. Override on top "
                  "of --mesh-style."
              ))
@click.option("--mesh-diffuse", default=None, type=float,
              help=(
                  "Mesh lighting.diffuse (0-1). Standard angular shading. "
                  "Higher values produce a stronger directional contour. "
                  "Plotly default is 0.8. Override on top of --mesh-style."
              ))
@click.option("--mesh-specular", default=None, type=float,
              help=(
                  "Mesh lighting.specular (0-2). The 'glossy' knob: higher "
                  "values produce brighter specular highlights. Plotly "
                  "default is 0.05 (almost matte); 1.0+ is shiny plastic; "
                  "2.0 is near-chrome. Override on top of --mesh-style."
              ))
@click.option("--mesh-roughness", default=None, type=float,
              help=(
                  "Mesh lighting.roughness (0-1). The 'shiny' knob: LOWER "
                  "values produce sharper, smaller highlights (polished "
                  "surface), higher values produce broader dimmer highlights "
                  "(sandpaper). Plotly default is 0.5. Override on top of "
                  "--mesh-style."
              ))
@click.option("--mesh-fresnel", default=None, type=float,
              help=(
                  "Mesh lighting.fresnel (0-5). Rim light at glancing "
                  "angles -- the 'edge glow' you see on shiny plastic. "
                  "Plotly default is 0.2. Override on top of --mesh-style."
              ))
@click.option("--mesh-light-position", default=None, type=str,
              help=(
                  "Position of the directional light in 3D world space, "
                  "given as 'x,y,z' (e.g. '1.5,1.5,1.0'). When omitted, "
                  "plotly's default light position is used. Affects where "
                  "the highlight lands on the mesh."
              ))

# === Labels and rendering ===
@click.option("--label-font-size", default=8, type=int,
              help="Font size for ROI labels on hover. Default: 8")
@click.option("--fast-render/--no-fast-render", default=False,
              help="Enable fast rendering optimizations for large networks.")

# === Camera and view ===
@click.option("--camera", default="oblique", type=click.Choice(CAMERA_VIEWS, case_sensitive=False),
              help="Camera view preset. Default: oblique")
@click.option("--enable-camera-controls/--no-camera-controls", default=True,
              help="Show camera view dropdown in the visualization. Default: enabled")

# === Custom camera (overrides --camera) ===
@click.option("--custom-camera-eye", default=None, type=str,
              help=(
                  "Custom camera EYE position as 'x,y,z' (e.g. '1.25,1.25,1.25'). "
                  "When provided, the plot opens at this exact view instead of the "
                  "--camera preset, the view is appended to the camera dropdown so "
                  "the user can flip back to it any time, and any --export-image "
                  "PNG/SVG/PDF inherits this same camera. If you also pass --camera "
                  "you'll get a warning and the custom camera wins. Tip: enable "
                  "--show-camera-readout to interactively pick an angle, copy the "
                  "printed flags, then re-run with them."
              ))
@click.option("--custom-camera-center", default="0,0,0", type=str,
              help=(
                  "Custom camera CENTER (look-at point) as 'x,y,z'. Only used "
                  "when --custom-camera-eye is given. Default: '0,0,0' (the "
                  "scene origin)."
              ))
@click.option("--custom-camera-up", default="0,0,1", type=str,
              help=(
                  "Custom camera UP vector as 'x,y,z'. Only used when "
                  "--custom-camera-eye is given. Default: '0,0,1' (z-up). Use "
                  "'0,1,0' if your data is y-up."
              ))
@click.option("--custom-camera-name", default="Custom View", type=str,
              help=(
                  "Display name for the custom camera. Shows up in the plot "
                  "title (e.g. 'View: My Saved Angle') and as the label of the "
                  "extra entry appended to the camera dropdown. Only used when "
                  "--custom-camera-eye is given. Default: 'Custom View'."
              ))
@click.option("--show-camera-readout/--no-camera-readout", default=False,
              help=(
                  "Inject a small live overlay into the saved HTML that shows "
                  "the current camera eye/center/up positions as you rotate the "
                  "brain, plus a copy-pastable block of '--custom-camera-...' "
                  "flags that reproduce the current view. Use this to find a "
                  "view interactively, then paste the printed flags into your "
                  "next invocation. Default: --no-camera-readout (overlay is "
                  "OMITTED). PASS --no-camera-readout (or just leave it off) IF "
                  "YOU DO NOT WANT THE OVERLAY TO APPEAR IN YOUR HTML. The "
                  "overlay is HTML-only and never shows up in static image "
                  "exports."
              ))

# === Node visibility ===
@click.option("--show-only-connected/--show-all-nodes", default=True,
              help="Only show nodes with at least one edge. Default: show only connected")
@click.option("--hide-nodes-with-hidden-edges/--keep-nodes-visible", default=True,
              help="Hide nodes when their edge type is toggled off in legend. Default: hide")

# === Node metrics for hover ===
@click.option("--node-metrics", default=None, type=click.Path(exists=True),
              help="CSV file with node metrics to display on hover. Rows=nodes, columns=metrics.")

# === Static image export ===
@click.option("--export-image", default=None, type=click.Path(),
              help="Export static image to this path. Supports: .png, .svg, .pdf")
@click.option("--image-format", default="png", type=click.Choice(["png", "svg", "pdf"]),
              help="Image format if --export-image path has no extension. Default: png")
@click.option("--image-dpi", default=300, type=int,
              help="DPI for PNG export (max ~288 for memory safety). Default: 300")
@click.option("--export-show-title/--export-no-title", default=True,
              help="Include title in exported image. Default: show title")
@click.option("--export-show-legend/--export-no-legend", default=True,
              help="Include legend in exported image. Default: show legend")

# === Per-edge color matrix ===
@click.option("--edge-color-matrix", default=None, type=click.Path(exists=True),
              help=(
                  "Path to a per-edge color matrix file (.csv/.txt/.npy) of the SAME "
                  "shape as --matrix. Each cell [i, j] specifies the color for the "
                  "edge between ROI i and ROI j. Cells may be: (1) a CSS color name "
                  "like 'red', a hex code like '#FF0000', or 'rgb(R,G,B)' -- used "
                  "directly; (2) an integer label (1, 2, 3, ...) -- all edges with "
                  "the same label get the same auto-generated color from a distinct "
                  "palette; or (3) empty / NaN / 0 -- the edge is SKIPPED (not "
                  "drawn). When provided this overrides --pos-edge-color and "
                  "--neg-edge-color."
              ))

# === P-value mode ===
@click.option("--matrix-type", default="weight",
              type=click.Choice(["weight", "pvalue"], case_sensitive=False),
              help=(
                  "How to interpret --matrix. 'weight' (default) treats it as a "
                  "weighted/correlation matrix where sign drives pos/neg coloring. "
                  "'pvalue' treats it as a matrix of p-values in (0, 1] and "
                  "internally transforms it via -log10(p) so that smaller "
                  "p-values produce thicker edges. Use --pvalue-threshold to "
                  "filter and --sign-matrix for signed effects."
              ))
@click.option("--pvalue-threshold", default=0.05, type=float,
              help=(
                  "Only used when --matrix-type pvalue. Edges with p-values STRICTLY "
                  "GREATER than this threshold are dropped (not drawn). Default: "
                  "0.05 (the conventional significance cutoff). To draw every "
                  "p-value regardless of significance pass --pvalue-threshold 1.0; "
                  "to be stricter, pass e.g. --pvalue-threshold 0.01."
              ))
@click.option("--sign-matrix", default=None, type=click.Path(exists=True),
              help=(
                  "Optional path to a same-shape matrix whose sign indicates the "
                  "direction of the underlying effect for each cell (+1 positive, "
                  "-1 negative, 0 unsigned). Only used when --matrix-type pvalue. "
                  "When provided, edges with positive sign are drawn in "
                  "--pos-edge-color and edges with negative sign in "
                  "--neg-edge-color, exactly like --matrix-type weight does."
              ))

# === Size / width legend keys ===
@click.option("--show-size-legend/--no-size-legend", default=True,
              help=(
                  "Render a 5-entry node-size key (sample dots labeled with "
                  "the sizes they represent) in the bottom-center of the "
                  "plot. Auto-skipped when --node-size is a single number "
                  "(no point showing 5 identical dots). Default: enabled. "
                  "Pass --no-size-legend for a clean publication figure."
              ))
@click.option("--show-width-legend/--no-width-legend", default=True,
              help=(
                  "Render a 5-entry edge-width key (sample line segments "
                  "labeled with the weight they represent) in the "
                  "bottom-center of the plot. Auto-skipped when "
                  "--edge-width-fixed is set. In --matrix-type pvalue mode "
                  "the labels show the ORIGINAL p-values, not the "
                  "-log10(p) transform. Default: enabled."
              ))
@click.option("--node-size-legend-metric", default=None, type=str,
              help=(
                  "Name of a column in --node-metrics whose values should "
                  "label the node-size key (instead of the literal pixel "
                  "sizes). When set, the key title becomes the column name "
                  "and the 5 sample labels are 5 evenly-spaced values from "
                  "that column (e.g. 'participation_coef: 0.10, 0.30, "
                  "0.50, 0.70, 0.90'). The dots themselves are still drawn "
                  "at the actual rendered pixel sizes from --node-size. "
                  "Use this when you've pre-scaled a metric into pixel "
                  "sizes and want the key to show the metric values, not "
                  "the pixel sizes. Requires --node-metrics to be passed. "
                  "Default: unset (key shows literal pixel sizes)."
              ))

# === Multi-view stitched export ===
@click.option("--multi-view", default=None, type=str,
              help=(
                  "Render a stitched 1xN PNG strip of the brain from "
                  "multiple camera angles INSTEAD of the normal single "
                  "static export. Pass a comma-separated list of view "
                  "names, e.g. --multi-view \"left,superior,posterior\". "
                  "Each entry can be: (1) a built-in preset name "
                  "(left/right/superior/inferior/anterior/posterior/"
                  "anterolateral_left/anterolateral_right/posterolateral_left/"
                  "posterolateral_right/oblique); or (2) the name of a "
                  "custom view registered with --custom-view. The "
                  "--export-image flag is REINTERPRETED as the path of "
                  "the stitched PNG when --multi-view is set; the "
                  "single-image export is suppressed. Multi-view output "
                  "is PNG-only -- SVG/PDF stitching is not supported."
              ))
@click.option("--custom-view", "custom_views", multiple=True, default=(), type=str,
              help=(
                  "Register a named custom view that can be referenced "
                  "from --multi-view. Format: 'NAME=eye_x,eye_y,eye_z' "
                  "(center defaults to '0,0,0' and up defaults to "
                  "'0,0,1'). Optionally append ';center=cx,cy,cz' and/or "
                  "';up=ux,uy,uz' for full control. Pass this flag MULTIPLE "
                  "times to register multiple views. Example: "
                  "--custom-view \"three_quarter=1.5,0.8,1.2\" "
                  "--custom-view \"low_anterior=0,1.8,0.3;up=0,0,1\" "
                  "--multi-view \"three_quarter,superior,low_anterior\""
              ))
@click.option("--multi-view-panel-size", default="800,800", type=str,
              help=(
                  "Pixel size of EACH individual panel in the stitched "
                  "strip, given as 'width,height'. Default '800,800'. The "
                  "final stitched image is N*width pixels wide. Multiplied "
                  "by the kaleido scale factor derived from --image-dpi."
              ))
@click.option("--multi-view-keep-first-legend/--multi-view-no-first-legend",
              default=True,
              help=(
                  "When --multi-view is set: keep the plotly legend in the "
                  "FIRST panel of the stitched strip and strip it from the "
                  "rest, so the user only sees the legend once. Default: "
                  "enabled. Pass --multi-view-no-first-legend to render "
                  "every panel without a legend."
              ))
@click.option("--multi-view-zoom", default=1.0, type=float,
              help=(
                  "Camera zoom multiplier applied uniformly to every panel "
                  "of the stitched strip. Values ABOVE 1.0 bring the camera "
                  "closer (brain looks bigger); values BELOW 1.0 push it "
                  "further away. Default 1.0 (no change). Examples: "
                  "--multi-view-zoom 1.5 makes the brain about 50% bigger, "
                  "--multi-view-zoom 2.0 doubles its apparent size."
              ))

# === Convenience ===
@click.option("--show/--no-show", default=False,
              help="Open the HTML file in browser after creation.")
def plot(mesh, coords, matrix, output, title, node_size, node_color,
         node_border_color, node_size_scale, pos_edge_color, neg_edge_color,
         edge_width_min, edge_width_max, edge_width_fixed, edge_width_scale,
         edge_threshold,
         mesh_color, mesh_opacity,
         mesh_style, mesh_ambient, mesh_diffuse, mesh_specular,
         mesh_roughness, mesh_fresnel, mesh_light_position,
         label_font_size, fast_render,
         camera, enable_camera_controls,
         custom_camera_eye, custom_camera_center, custom_camera_up,
         custom_camera_name, show_camera_readout,
         show_only_connected,
         hide_nodes_with_hidden_edges, node_metrics,
         export_image, image_format, image_dpi,
         export_show_title, export_show_legend,
         edge_color_matrix, matrix_type, pvalue_threshold, sign_matrix,
         show_size_legend, show_width_legend, node_size_legend_metric,
         multi_view, custom_views, multi_view_panel_size,
         multi_view_keep_first_legend, multi_view_zoom,
         show):
    """
    Create an interactive 3D brain connectivity visualization.

    This command creates an HTML file with an interactive 3D brain plot
    showing nodes (ROIs) and edges (connections) on a brain mesh surface.

    \b
    REQUIRED FILES:
      --mesh: Brain surface mesh (.gii, .obj, .mz3, .ply)
      --coords: ROI coordinates CSV with cog_x, cog_y, cog_z, roi_name columns
      --matrix: Connectivity matrix (.npy, .csv, .txt, .mat, .edge)

    \b
    NODE SIZE OPTIONS:
      - Fixed: --node-size 10
      - From file: --node-size path/to/sizes.csv
      - Scaled by metric: --node-size pc --node-metrics metrics.csv

    \b
    NODE COLOR OPTIONS:
      - Single color: --node-color purple
      - Hex code: --node-color "#FF5733"
      - From modules: --node-color path/to/modules.csv (integers 1-N)

    \b
    EDGE WIDTH OPTIONS:
      - Scaled by weight: --edge-width-min 1 --edge-width-max 8
      - Fixed width: --edge-width-fixed 2

    \b
    EXAMPLES:
      # Basic plot
      hlplot plot -m brain.gii -c rois.csv -x connectivity.npy

      # Customized appearance
      hlplot plot -m brain.gii -c rois.csv -x connectivity.npy \\
        --node-size 12 --node-color steelblue --mesh-opacity 0.2

      # With module coloring
      hlplot plot -m brain.gii -c rois.csv -x connectivity.npy \\
        --node-color modules.csv --node-border-color darkgray

      # Export static image
      hlplot plot -m brain.gii -c rois.csv -x connectivity.npy \\
        --export-image figure.png --image-dpi 300

      # Clean export (no title/legend)
      hlplot plot -m brain.gii -c rois.csv -x connectivity.npy \\
        --export-image clean.pdf --export-no-title --export-no-legend

      # Color edges from a custom color matrix (hex/rgb/integer labels)
      hlplot plot -m brain.gii -c rois.csv -x connectivity.npy \\
        --edge-color-matrix edge_colors.csv

      # Plot a p-value matrix (-log10 transform, drop p > 0.05)
      hlplot plot -m brain.gii -c rois.csv -x pvalues.csv \\
        --matrix-type pvalue --pvalue-threshold 0.05 \\
        --edge-width-min 1 --edge-width-max 8

      # Signed p-values: pos effects red, neg effects blue
      hlplot plot -m brain.gii -c rois.csv -x pvalues.csv \\
        --matrix-type pvalue --sign-matrix effect_signs.csv
    """
    try:
        # Import here to avoid slow startup
        import numpy as np
        import pandas as pd
        from HarrisLabPlotting import (
            load_mesh_file,
            create_brain_connectivity_plot,
            load_connectivity_input,
            convert_node_size_input
        )

        print_info(f"Loading mesh from {mesh}...")
        vertices, faces = load_mesh_file(mesh)
        print_success(f"Loaded mesh: {len(vertices)} vertices, {len(faces)} faces")

        print_info(f"Loading ROI coordinates from {coords}...")
        roi_df = pd.read_csv(coords)
        print_success(f"Loaded {len(roi_df)} ROI coordinates")

        print_info(f"Loading connectivity matrix from {matrix}...")
        connectivity = load_connectivity_input(matrix)
        print_success(f"Loaded matrix: {connectivity.shape}")

        # Check dimension compatibility
        if connectivity.shape[0] != len(roi_df):
            print_warning(f"Matrix size ({connectivity.shape[0]}) differs from ROI count ({len(roi_df)})")
            print_info("Consider using map_coordinate() to align ROI coordinates with your matrix")

        # Process node size
        try:
            node_size_val = float(node_size)
        except ValueError:
            # It's a file path
            if os.path.exists(node_size):
                node_size_val = node_size
            else:
                node_size_val = node_size  # Let the function handle it

        # Process edge width
        if edge_width_fixed is not None:
            edge_width_val = edge_width_fixed
        else:
            edge_width_val = (edge_width_min, edge_width_max)

        def _parse_xyz(flag_name: str, raw: str):
            """Parse a 'x,y,z' string into a tuple of three floats."""
            try:
                parts = [float(p.strip()) for p in raw.split(',')]
                if len(parts) != 3:
                    raise ValueError
                return tuple(parts)
            except ValueError:
                print_error(
                    f"{flag_name} must be three comma-separated numbers "
                    f"like '1.25,1.25,1.25' (got {raw!r})"
                )
                raise click.Abort()

        # Parse --mesh-light-position "x,y,z"
        mesh_light_position_val = None
        if mesh_light_position:
            mesh_light_position_val = _parse_xyz('--mesh-light-position',
                                                 mesh_light_position)

        def _parse_custom_view_spec(spec: str):
            """Parse one --custom-view 'NAME=ex,ey,ez[;center=cx,cy,cz][;up=ux,uy,uz]' string."""
            if '=' not in spec:
                print_error(
                    f"--custom-view must be 'NAME=eye_x,eye_y,eye_z[;center=...][;up=...]' "
                    f"(got {spec!r})"
                )
                raise click.Abort()
            head, _, tail = spec.partition('=')
            view_name = head.strip()
            if not view_name:
                print_error(f"--custom-view name cannot be empty (got {spec!r})")
                raise click.Abort()
            # Default center / up
            center = (0.0, 0.0, 0.0)
            up = (0.0, 0.0, 1.0)
            # The first segment (before any ';') is the eye triple.
            segs = [s.strip() for s in tail.split(';') if s.strip()]
            if not segs:
                print_error(
                    f"--custom-view {view_name!r} has no eye coordinates"
                )
                raise click.Abort()
            eye = _parse_xyz(f'--custom-view {view_name}', segs[0])
            for seg in segs[1:]:
                if '=' not in seg:
                    print_error(
                        f"--custom-view sub-segments must be 'center=...' or "
                        f"'up=...' (got {seg!r})"
                    )
                    raise click.Abort()
                key, _, val = seg.partition('=')
                key = key.strip().lower()
                if key == 'center':
                    center = _parse_xyz(f'--custom-view {view_name} center', val)
                elif key == 'up':
                    up = _parse_xyz(f'--custom-view {view_name} up', val)
                else:
                    print_error(
                        f"--custom-view sub-segment key must be 'center' or "
                        f"'up' (got {key!r})"
                    )
                    raise click.Abort()
            return view_name, dict(
                name=view_name,
                eye=dict(x=eye[0], y=eye[1], z=eye[2]),
                center=dict(x=center[0], y=center[1], z=center[2]),
                up=dict(x=up[0], y=up[1], z=up[2]),
            )

        # Build the registry of named custom views from --custom-view.
        custom_view_registry: dict = {}
        for cv in custom_views:
            name, cam = _parse_custom_view_spec(cv)
            custom_view_registry[name] = cam

        # Parse --multi-view "name1,name2,name3" into a list of camera
        # dicts (mixing built-in preset names with registered custom view
        # names). The actual preset lookup happens inside
        # export_multi_view_stitched_png; here we only resolve names from
        # the registry into full dicts so the helper sees a flat list.
        multi_view_list = None
        if multi_view:
            from HarrisLabPlotting import CameraController
            preset_names = set(CameraController.PRESET_VIEWS.keys())
            entries = [s.strip() for s in multi_view.split(',') if s.strip()]
            if not entries:
                print_error("--multi-view must contain at least one view name")
                raise click.Abort()
            multi_view_list = []
            for name in entries:
                if name in custom_view_registry:
                    multi_view_list.append(custom_view_registry[name])
                elif name in preset_names:
                    multi_view_list.append(name)
                else:
                    print_error(
                        f"--multi-view: unknown view {name!r}. Built-in "
                        f"presets: {sorted(preset_names)}. Registered "
                        f"custom views: {sorted(custom_view_registry.keys())}."
                    )
                    raise click.Abort()
            print_info(
                f"Multi-view: rendering {len(multi_view_list)} panels: "
                f"{[v if isinstance(v, str) else v.get('name', '?') for v in multi_view_list]}"
            )

        # Parse --multi-view-panel-size 'w,h'
        try:
            mvps_parts = [int(p.strip()) for p in multi_view_panel_size.split(',')]
            if len(mvps_parts) != 2:
                raise ValueError
            multi_view_panel_size_val = (mvps_parts[0], mvps_parts[1])
        except ValueError:
            print_error(
                f"--multi-view-panel-size must be 'width,height' (got {multi_view_panel_size!r})"
            )
            raise click.Abort()

        # Parse --custom-camera-* flags. The eye flag drives everything;
        # center/up/name only matter when eye is provided.
        custom_camera_dict = None
        if custom_camera_eye:
            # B3: warn-and-prefer when both --camera and --custom-camera-eye
            # are provided. (We can't reliably detect "user explicitly
            # passed --camera" vs "click default" so we always warn that
            # the custom view is winning.)
            if camera and camera.lower() != 'oblique':
                print_warning(
                    f"Both --camera {camera!r} and --custom-camera-eye were "
                    f"provided. The custom camera wins; --camera is ignored."
                )
            eye = _parse_xyz('--custom-camera-eye', custom_camera_eye)
            center = _parse_xyz('--custom-camera-center', custom_camera_center)
            up = _parse_xyz('--custom-camera-up', custom_camera_up)
            custom_camera_dict = dict(
                eye=dict(x=eye[0], y=eye[1], z=eye[2]),
                center=dict(x=center[0], y=center[1], z=center[2]),
                up=dict(x=up[0], y=up[1], z=up[2]),
                name=custom_camera_name,
            )

        print_info("Creating brain connectivity plot...")

        # Create the plot
        fig, stats = create_brain_connectivity_plot(
            vertices=vertices,
            faces=faces,
            roi_coords_df=roi_df,
            connectivity_matrix=connectivity,
            plot_title=title,
            save_path=output,
            node_size=node_size_val,
            node_color=node_color,
            node_border_color=node_border_color,
            pos_edge_color=pos_edge_color,
            neg_edge_color=neg_edge_color,
            edge_width=edge_width_val,
            edge_width_scale=edge_width_scale,
            node_size_scale=node_size_scale,
            edge_threshold=edge_threshold,
            mesh_color=mesh_color,
            mesh_opacity=mesh_opacity,
            label_font_size=label_font_size,
            fast_render=fast_render,
            camera_view=camera,
            enable_camera_controls=enable_camera_controls,
            show_only_connected_nodes=show_only_connected,
            node_metrics=node_metrics,
            hide_nodes_with_hidden_edges=hide_nodes_with_hidden_edges,
            export_image=export_image,
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
            mesh_light_position=mesh_light_position_val,
            custom_camera=custom_camera_dict,
            custom_camera_name=custom_camera_name if custom_camera_dict else None,
            show_camera_readout=show_camera_readout,
            show_size_legend=show_size_legend,
            show_width_legend=show_width_legend,
            node_size_legend_metric=node_size_legend_metric,
            multi_view=multi_view_list,
            multi_view_panel_size=multi_view_panel_size_val,
            multi_view_keep_first_legend=multi_view_keep_first_legend,
            multi_view_zoom=multi_view_zoom,
        )

        print_success(f"Saved interactive visualization to {output}")

        if export_image:
            print_success(f"Exported static image to {export_image}")

        # Display statistics
        console.print()
        display_stats = {
            "Total nodes": stats.get("total_nodes", "N/A"),
            "Connected nodes": stats.get("connected_nodes", "N/A"),
            "Total edges": stats.get("total_edges", "N/A"),
            "Positive edges": stats.get("positive_edges", "N/A"),
            "Negative edges": stats.get("negative_edges", "N/A"),
            "Network density": stats.get("network_density", "N/A"),
        }
        table = create_stats_table(display_stats, title="Network Statistics")
        console.print(table)

        # Optionally open in browser
        if show:
            import webbrowser
            output_path = Path(output).resolve()
            webbrowser.open(f"file://{output_path}")

    except Exception as e:
        print_error(f"Error creating plot: {e}")
        raise click.Abort()
