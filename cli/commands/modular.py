"""
Modularity visualization commands.
"""

import click
from pathlib import Path
import os

from ..console import console, print_success, print_error, print_warning, print_info, create_stats_table
from .plot import _parse_show_node_labels_arg
from ..pager import PagerCommand


# Available camera view presets
CAMERA_VIEWS = [
    "oblique", "anterior", "posterior", "left", "right",
    "superior", "inferior", "dorsal", "ventral",
    "lateral-left", "lateral-right"
]


@click.command(cls=PagerCommand)
# === Required inputs ===
@click.option("--mesh", "-m", required=True, type=click.Path(exists=True),
              help="Path to brain mesh file. Supported formats: .gii (GIFTI), .obj, .mz3, .ply")
@click.option("--coords", "-c", required=True, type=click.Path(exists=True),
              help="Path to ROI coordinates CSV file. Must contain columns: cog_x, cog_y, cog_z, roi_name")
@click.option("--matrix", "-x", required=True, type=click.Path(exists=True),
              help="Path to connectivity matrix. Supported formats: .npy, .csv, .txt, .mat, .edge")
@click.option("--modules", "-d", required=True, type=click.Path(exists=True),
              help="Path to module assignments file. CSV with 'module' column or single column of integers (1-indexed)")

# === Output options ===
@click.option("--output", "-o", default="brain_modularity.html", type=click.Path(),
              help="Output HTML file path. Default: brain_modularity.html")
@click.option("--title", "-t", default="Brain Connectivity with Modularity",
              help="Plot title (Q and Z scores appended if provided).")

# === Modularity scores ===
@click.option("--q-score", default=None, type=float,
              help="Modularity Q score to display in title. Example: 0.452")
@click.option("--z-score", default=None, type=float,
              help="Z-rand score to display in title. Example: 3.21")

# === Node appearance ===
@click.option("--node-size", default="8",
              help="""Node size specification. Accepts:
              - Single number: All nodes same size (e.g., '10')
              - File path: CSV/NPY file with per-node sizes
              Nodes are automatically colored by module assignment.""")
@click.option("--node-border-color", default="darkgray",
              help="Border/outline color for nodes. Default: darkgray")
@click.option("--node-size-scale", default=1.0, type=float,
              help=(
                  "Uniform multiplier applied to every node size AFTER "
                  "all other size resolution. Default 1.0 (no change). "
                  "Use this when you've tuned --node-size and just want "
                  "every dot proportionally larger or smaller. Examples: "
                  "--node-size 10 --node-size-scale 2 -> every dot at "
                  "20 px; --node-size sizes.csv --node-size-scale 1.5 "
                  "-> every dot 50% bigger."
              ))

# === Edge coloring ===
@click.option("--edge-color-mode", default="module",
              type=click.Choice(["module", "sign"], case_sensitive=False),
              help="""Edge coloring mode:
              - 'module': Edges colored by source node's module (default)
              - 'sign': Red for positive, blue for negative connections""")
@click.option("--pos-edge-color", default="red",
              help="Color for positive connections (when edge-color-mode=sign). Default: red")
@click.option("--neg-edge-color", default="blue",
              help="Color for negative connections (when edge-color-mode=sign). Default: blue")

# === Edge width ===
@click.option("--edge-width-min", default=1.0, type=float,
              help="Minimum edge line width when scaling by weight. Default: 1.0")
@click.option("--edge-width-max", default=5.0, type=float,
              help="Maximum edge line width when scaling by weight. Default: 5.0")
@click.option("--edge-width-fixed", default=None, type=float,
              help="Fixed edge width (disables weight-based scaling). If set, ignores min/max.")
@click.option("--edge-width-scale", default=1.0, type=float,
              help=(
                  "Uniform multiplier applied to every edge width AFTER all other "
                  "scaling. Default 1.0 (no change). Useful when you've tuned "
                  "--edge-width-min/--edge-width-max (or --edge-width-fixed) and "
                  "you just want every edge proportionally thicker/thinner. "
                  "E.g. --edge-width-min 1 --edge-width-max 5 --edge-width-scale "
                  "5 -> final widths in [5, 25]."
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
@click.option("--show-node-labels", default="true", type=str,
              help=(
                  "Controls which ROI text labels are rendered next to "
                  "their node markers. Hover tooltips are always shown "
                  "regardless of this setting. Accepts: 'true' (default; "
                  "every ROI gets a label), 'false' (no labels), or a "
                  "path to a CSV/TXT/NPY file containing a per-node 0/1 "
                  "(or True/False) vector of length N where 1 = show the "
                  "label and 0 = hide it. The CSV may have a header "
                  "column ('show_label') or be headerless. Use this to "
                  "label only a subset of regions in publication figures "
                  "(e.g. only hub nodes)."
              ))
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
                  "--custom-camera-eye is given. Default: '0,0,1' (z-up)."
              ))
@click.option("--custom-camera-name", default="Custom View", type=str,
              help=(
                  "Display name for the custom camera. Shows up in the plot "
                  "title and as the label of the extra entry appended to the "
                  "camera dropdown. Only used when --custom-camera-eye is given. "
                  "Default: 'Custom View'."
              ))
@click.option("--show-camera-readout/--no-camera-readout", default=False,
              help=(
                  "Inject a small live overlay into the saved HTML that shows "
                  "the current camera eye/center/up positions as you rotate the "
                  "brain, plus a copy-pastable block of '--custom-camera-...' "
                  "flags that reproduce the current view. Default: "
                  "--no-camera-readout (overlay is OMITTED). PASS "
                  "--no-camera-readout (or just leave it off) IF YOU DO NOT "
                  "WANT THE OVERLAY TO APPEAR IN YOUR HTML. The overlay is "
                  "HTML-only and never shows up in static image exports."
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
              help="DPI for PNG export (no hard cap; very high values make very "
                   "large images and use more memory). Default: 300")
@click.option("--export-show-title/--export-no-title", default=True,
              help="Include title in exported image. Default: show title")
@click.option("--export-show-legend/--export-no-legend", default=True,
              help="Include legend in exported image. Default: show legend")
@click.option("--export-size", default="1200,1200", type=str,
              help="Exported image canvas size as 'width,height'. Default "
                   "'1200,1200' (square) which leaves even margins on both sides. "
                   "KEEP WIDTH == HEIGHT if you plan to change --image-dpi: on a "
                   "non-square canvas the rendered 3D aspect varies with the DPI "
                   "scale factor, so the brain's proportions shift between DPIs.")
@click.option("--export-autocrop/--no-export-autocrop", default=False,
              help="Trim the uniform background border around the exported PNG so "
                   "the figure frames its content tightly (no even margins). Pure "
                   "crop, no aspect warp. Raster only (SVG/PDF unaffected). "
                   "Default: OFF -- the square --export-size canvas already gives "
                   "even margins and a DPI-stable aspect.")
@click.option("--background-color", default="white",
              help="Figure background color (applies to the saved HTML AND the "
                   "exported image). Accepts a named color, a hex code like "
                   "'#1e1e1e', or 'transparent' for a transparent PNG. "
                   "Default: white.")

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
                  "drawn). When provided this overrides --edge-color-mode."
              ))

# === P-value mode ===
@click.option("--matrix-type", default="weight",
              type=click.Choice(["weight", "pvalue"], case_sensitive=False),
              help=(
                  "How to interpret --matrix. 'weight' (default) treats it as a "
                  "weighted/correlation matrix. 'pvalue' treats it as a matrix of "
                  "p-values in (0, 1] and internally transforms it via -log10(p) "
                  "so that smaller p-values produce thicker edges. Use "
                  "--pvalue-threshold to filter and --sign-matrix for signed effects."
              ))
@click.option("--pvalue-threshold", default=0.05, type=float,
              help=(
                  "Only used when --matrix-type pvalue. Edges with p-values STRICTLY "
                  "GREATER than this threshold are dropped (not drawn). Default: "
                  "0.05 (the conventional significance cutoff). To draw every "
                  "p-value pass --pvalue-threshold 1.0; to be stricter pass "
                  "e.g. --pvalue-threshold 0.01."
              ))
@click.option("--sign-matrix", default=None, type=click.Path(exists=True),
              help=(
                  "Optional path to a same-shape matrix whose sign indicates the "
                  "direction of the underlying effect (+1 positive, -1 negative, "
                  "0 unsigned). Only used when --matrix-type pvalue. When provided "
                  "and --edge-color-mode is 'sign', positive effects are drawn in "
                  "--pos-edge-color and negative effects in --neg-edge-color."
              ))

# === Size / width legend keys ===
@click.option("--show-size-legend/--no-size-legend", default=True,
              help=(
                  "Render a 5-entry node-size key (sample dots labeled with "
                  "the sizes they represent) in the bottom-center of the "
                  "plot. Auto-skipped when --node-size is a single number. "
                  "Default: enabled. Pass --no-size-legend for a clean "
                  "publication figure."
              ))
@click.option("--show-width-legend/--no-width-legend", default=True,
              help=(
                  "Render a 5-entry edge-width key (sample line segments "
                  "labeled with the weight they represent) in the "
                  "bottom-center of the plot. Auto-skipped when "
                  "--edge-width-fixed is set. In --matrix-type pvalue mode "
                  "the labels show the ORIGINAL p-values. Default: enabled."
              ))
@click.option("--node-size-legend-metric", default=None, type=str,
              help=(
                  "Name of a column in --node-metrics whose values should "
                  "label the node-size key (instead of the literal pixel "
                  "sizes). When set, the key title becomes the column name "
                  "and the 5 sample labels are 5 evenly-spaced values from "
                  "that column. The dots themselves are still drawn at the "
                  "actual rendered pixel sizes from --node-size. Requires "
                  "--node-metrics. Default: unset."
              ))

# === Multi-view stitched export ===
@click.option("--multi-view", default=None, type=str,
              help=(
                  "Render a stitched 1xN PNG strip of the brain from "
                  "multiple camera angles INSTEAD of the normal single "
                  "static export. Pass a comma-separated list of view "
                  "names, e.g. --multi-view \"left,superior,posterior\". "
                  "Each entry can be a built-in preset name or the name "
                  "of a custom view registered with --custom-view. The "
                  "--export-image flag is REINTERPRETED as the path of "
                  "the stitched PNG when --multi-view is set. PNG-only."
              ))
@click.option("--custom-view", "custom_views", multiple=True, default=(), type=str,
              help=(
                  "Register a named custom view that can be referenced "
                  "from --multi-view. Format: 'NAME=eye_x,eye_y,eye_z' "
                  "(center defaults to '0,0,0' and up defaults to "
                  "'0,0,1'). Optionally append ';center=cx,cy,cz' and/or "
                  "';up=ux,uy,uz'. Pass MULTIPLE times to register multiple "
                  "views."
              ))
@click.option("--multi-view-panel-size", default="800,800", type=str,
              help=(
                  "Pixel size of EACH panel in the stitched strip, given "
                  "as 'width,height'. Default '800,800'."
              ))
@click.option("--multi-view-keep-first-legend/--multi-view-no-first-legend",
              default=True,
              help=(
                  "When --multi-view is set: keep the plotly legend in the "
                  "FIRST panel of the stitched strip and strip it from the "
                  "rest. Default: enabled."
              ))
@click.option("--multi-view-grid", default=None, type=str,
              help=(
                  "Layout for --multi-view panels, given as 'rows,cols' "
                  "(e.g. '2,3' for a 2-row by 3-column grid). When omitted "
                  "(the default), every panel is placed in a single "
                  "horizontal row (1xN) -- the original behavior, "
                  "unchanged. Panels fill ROW-MAJOR (left-to-right, then "
                  "top-to-bottom). If rows*cols exceeds the number of "
                  "views, the trailing cells render as blank panels with "
                  "the background color. If rows*cols is less than the "
                  "number of views, the command errors out."
              ))
@click.option("--zoom", default=1.0, type=float,
              help=(
                  "Camera zoom multiplier applied to the rendered brain. "
                  "Applies to BOTH the single-view export (HTML and any "
                  "static PNG/SVG/PDF) AND every panel of a --multi-view "
                  "stitched export. Values ABOVE 1.0 bring the camera "
                  "closer (brain looks bigger); values BELOW 1.0 push it "
                  "further away. Default 1.0 (no change). Replaces the "
                  "old --multi-view-zoom flag, which has been removed."
              ))

# === Node-role classification (PC + within-module Z-score) ===
@click.option("--node-roles/--no-node-roles", default=False,
              help=(
                  "Enable per-node role classification using the "
                  "cartographic two-cut of Guimera & Amaral, Functional "
                  "cartography of complex metabolic networks, Nature "
                  "433:895 (2005). Seven regions: Ultra-peripheral / "
                  "Peripheral / Non-hub connector / Non-hub kinless / "
                  "Provincial hub / Connector hub / Kinless hub. When "
                  "set, each node renders as a dual-layer marker: outer "
                  "ring is the role color, inner fill is the module "
                  "color. Requires --node-metrics with "
                  "'participation_coef' and 'within_module_zscore' "
                  "columns. A small role legend is added in the "
                  "bottom-left corner. Default: off."
              ))
@click.option("--node-size-mode",
              default="fixed",
              type=click.Choice(["fixed", "pc", "zscore", "both"], case_sensitive=False),
              help=(
                  "How to compute per-node sizes. 'fixed' (default) uses "
                  "--node-size as-is. 'pc' / 'zscore' / 'both' derive "
                  "sizes dynamically from participation coefficient, "
                  "within-module Z-score, or a 50/50 blend of both. "
                  "Requires --node-metrics. Use --base-node-size and "
                  "--max-node-multiplier to tune the dynamic range."
              ))
@click.option("--base-node-size", default=None, type=int,
              help=(
                  "Base size (px) for dynamic --node-size-mode. When unset, "
                  "falls back to --node-size if it's a scalar, else 8. "
                  "Default: unset."
              ))
@click.option("--max-node-multiplier", default=5.0, type=float,
              help=(
                  "Maximum size multiplier applied to --base-node-size in "
                  "dynamic --node-size-mode. Default 5.0 (i.e. the largest "
                  "node is up to 5x the base size)."
              ))
@click.option("--border-width", default=6, type=int,
              help=(
                  "Pixel width of the role-classification border ring "
                  "rendered when --node-roles is on. Larger values make "
                  "the role color more visible at the cost of obscuring "
                  "the module fill. Default 6."
              ))
@click.option("--viz-type",
              default="all",
              type=click.Choice(["all", "intra", "inter", "nodes_only"], case_sensitive=False),
              help=(
                  "Edge visualization filter. 'all' (default) draws every "
                  "edge. 'intra' keeps only within-module edges. 'inter' "
                  "keeps only between-module edges. 'nodes_only' drops all "
                  "edges so only the labelled nodes render."
              ))
@click.option("--inter-edge-color", default=None, type=str,
              help=(
                  "Override color for between-module (inter-module) "
                  "edges, e.g. 'black' or '#000000'. SCOPED: only takes "
                  "effect when --edge-color-mode is 'module' AND "
                  "--viz-type is 'all' or 'inter'. Ignored (with a one-"
                  "line note) in sign mode; silently moot with --viz-type "
                  "'intra' or 'nodes_only'. Useful for highlighting "
                  "cross-module connectivity as a single visual layer. "
                  "Default: unset (use --edge-color-mode behavior)."
              ))

# === Directed graphs (asymmetric matrices) ===
@click.option("--directed/--undirected", "directed", default=None,
              help=(
                  "Force directed (arrowheads) or undirected rendering. "
                  "DEFAULT: auto -- the matrix is tested for symmetry and "
                  "arrows are drawn only if it is asymmetric. The verdict is "
                  "printed on every run either way."
              ))
@click.option("--matrix-orientation",
              type=click.Choice(["row-to-col", "col-to-row"], case_sensitive=False),
              default="row-to-col",
              help=(
                  "Which index is the SOURCE. 'row-to-col' (DEFAULT) means "
                  "M[i,j] is the connection i -> j, the numpy/networkx "
                  "convention. 'col-to-row' transposes on load: use it for "
                  "SPM DCM, whose A(i,j) is the connection FROM j TO i. "
                  "Getting this wrong reverses every arrow silently. NOTE a "
                  "row-stochastic transition matrix (rows sum to 1) is "
                  "already row-to-col and needs no change."
              ))
@click.option("--symmetry-tol", type=float, default=None,
              help=(
                  "Absolute tolerance for calling a matrix symmetric. "
                  "DEFAULT: numpy's allclose (rtol 1e-5, atol 1e-8), which "
                  "treats CSV round-trip noise (~1e-16) as symmetric. Lower "
                  "it to be stricter. Units: same as the matrix values."
              ))
@click.option("--arrow-view-mode",
              type=click.Choice(["camera", "fixed"], case_sensitive=False),
              default="camera",
              help=(
                  "How reciprocal (two-way) pairs bow apart. 'camera' "
                  "(DEFAULT) bows each arc perpendicular to the viewing axis "
                  "so the pair separates as widely as that projection allows, "
                  "recomputed per multi-view panel. 'fixed' bows in an "
                  "anatomically fixed direction, identical from every angle. "
                  "Saved HTML always uses 'fixed' so it stays correct when "
                  "you rotate it."
              ))
@click.option("--arrow-size", type=float, default=1.08,
              help=(
                  "Arrowhead radius as a MULTIPLE of that edge's own line "
                  "half-width. 1.0 = exactly as wide as its line; larger = a "
                  "fatter head. Unitless ratio. DEFAULT 1.08. Typical range "
                  "1.0-2.0."
              ))
@click.option("--arrow-slenderness", type=float, default=0.18,
              help=(
                  "Arrowhead radius divided by its length. SMALLER = a "
                  "longer, thinner dart (more obviously an arrow); larger = "
                  "short and stubby. Unitless ratio. DEFAULT 0.18. Typical "
                  "range 0.12-0.35."
              ))
@click.option("--arrow-max-edge-frac", type=float, default=0.30,
              help=(
                  "Cap on arrowhead length as a FRACTION of the edge it caps, "
                  "so a short edge never gets a head longer than itself. "
                  "Unitless fraction of edge length. DEFAULT 0.30 (30%). "
                  "Typical range 0.2-0.5."
              ))
@click.option("--arrow-min-radius-px", type=float, default=1.2,
              help=(
                  "Floor on arrowhead radius, in SCREEN PIXELS, so the very "
                  "thinnest edges still show a visible head. DEFAULT 1.2 px. "
                  "Set 0 to let heads shrink with their line without limit."
              ))
@click.option("--arc-bow-frac", type=float, default=0.10,
              help=(
                  "How far a reciprocal pair's arcs bow apart, as a FRACTION "
                  "of the straight-line distance between the two nodes. "
                  "DEFAULT 0.10 (10% of the chord). One-way edges are always "
                  "straight and ignore this."
              ))
@click.option("--arc-bow-floor", type=float, default=0.03,
              help=(
                  "Minimum bow for a reciprocal pair, as a FRACTION of the "
                  "coordinate bounding-box diagonal. Without a floor, very "
                  "short pairs bow by almost nothing and overlap. DEFAULT "
                  "0.03 (3% of the diagonal)."
              ))
@click.option("--arrow-darken", type=float, default=0.74,
              help=(
                  "Arrowheads are drawn in this SHADE of their line colour so "
                  "they read against it: 1.0 = identical colour, lower = "
                  "darker. Unitless multiplier on RGB. DEFAULT 0.74."
              ))
@click.option("--no-html", is_flag=True, default=False,
              help=(
                  "Skip writing the interactive HTML and produce only the "
                  "static --export-image. Use this when you want a figure and "
                  "not a browser file -- especially with voxel overlays, "
                  "where the HTML is far larger than the PNG."
              ))
# === Voxel-map overlays (see `hlplot volume` for the full set) ===
@click.option("--volume", "-V", "volume_paths", multiple=True,
              type=click.Path(exists=True),
              help=("Overlay a statistical volume (.nii/.nii.gz) UNDER the "
                    "network. Repeatable. The map must be in the same world "
                    "space as the mesh -- check with `hlplot utils "
                    "check-alignment`. For the full set of per-map controls "
                    "use `hlplot volume`."))
@click.option("--volume-spec", type=click.Path(exists=True),
              help=("YAML file describing the overlay map(s); same format as "
                    "`hlplot volume`. A CLI flag overrides the same key for "
                    "every map in the file."))
@click.option("--volume-cmap", "volume_cmaps", multiple=True,
              help=("Colorscale per overlay, matched by position. 'hot32' "
                    "(activation) and 'ice28' (deactivation) are built in; "
                    "any plotly scale name also works."))
@click.option("--volume-name", "volume_names", multiple=True,
              help="Label for each overlay's colorbar, matched by position.")
@click.option("--volume-threshold", "volume_thresholds", multiple=True,
              type=float,
              help=("Threshold per overlay, as an ABSOLUTE value in the map's "
                    "own units (e.g. 3.1 for a z-map). Given once, applies to "
                    "every overlay."))
@click.option("--volume-smooth-fwhm", "volume_smooths", multiple=True,
              help=("Gaussian blur FWHM in MILLIMETRES: one number, or "
                    "'X,Y,Z' per axis. Pass the ORIGINAL pre-warp voxel size. "
                    "Default: none."))
@click.option("--volume-step", "volume_steps", multiple=True, type=int,
              help=("Take every Nth voxel of the overlay grid. Nothing is "
                    "downsampled by default and the projected cost is "
                    "printed; a step of 5-7 is visually identical when "
                    "smoothing."))
@click.option("--volume-opacity", "volume_opacities", multiple=True,
              type=float,
              help=("Opacity ceiling of the VOXEL MAP, 0-1 (default 1.0). "
                    "Lower it to ~0.6 so edges read through a bright cloud. "
                    "This is not the brain -- that is --mesh-opacity."))
# === Convenience ===
@click.option("--show/--no-show", default=False,
              help="Open the HTML file in browser after creation.")
def modular(mesh, coords, matrix, modules, output, title, q_score, z_score,
            node_size, node_border_color, node_size_scale,
            edge_color_mode, pos_edge_color, neg_edge_color,
            edge_width_min, edge_width_max, edge_width_fixed, edge_width_scale,
            edge_threshold,
            mesh_color, mesh_opacity,
            mesh_style, mesh_ambient, mesh_diffuse, mesh_specular,
            mesh_roughness, mesh_fresnel, mesh_light_position,
            label_font_size, show_node_labels, fast_render,
            camera, enable_camera_controls,
            custom_camera_eye, custom_camera_center, custom_camera_up,
            custom_camera_name, show_camera_readout,
            show_only_connected,
            hide_nodes_with_hidden_edges, node_metrics,
            export_image, image_format, image_dpi,
            export_show_title, export_show_legend, background_color, export_size, export_autocrop,
            edge_color_matrix, matrix_type, pvalue_threshold, sign_matrix,
            show_size_legend, show_width_legend, node_size_legend_metric,
            multi_view, custom_views, multi_view_panel_size,
            multi_view_keep_first_legend, multi_view_grid, zoom,
            node_roles, node_size_mode, base_node_size, max_node_multiplier,
            border_width, viz_type, inter_edge_color,
            show,
         directed, matrix_orientation, symmetry_tol, arrow_view_mode,
         arrow_size, arrow_slenderness, arrow_max_edge_frac,
         arrow_min_radius_px, arc_bow_frac, arc_bow_floor,
         arrow_darken, no_html,
         volume_paths, volume_spec, volume_cmaps, volume_names,
         volume_thresholds, volume_smooths, volume_steps,
         volume_opacities):
    """
    Create a brain connectivity plot with modularity visualization.

    This command creates a visualization where nodes are automatically colored
    by their module assignment, with a module legend. Edges can be colored
    either by module (matching source node) or by sign (positive/negative).

    \b
    REQUIRED FILES:
      --mesh: Brain surface mesh (.gii, .obj, .mz3, .ply)
      --coords: ROI coordinates CSV with cog_x, cog_y, cog_z, roi_name columns
      --matrix: Connectivity matrix (.npy, .csv, .txt, .mat, .edge)
      --modules: Module assignments CSV or NPY file (integers 1 to N)

    \b
    MODULE FILE FORMAT:
      Option 1: CSV with 'module' column
        roi_index,module
        0,1
        1,2
        2,1
        ...

      Option 2: Single column file (one module per line)
        1
        2
        1
        ...

    \b
    EDGE COLORING:
      --edge-color-mode module: Edges inherit color from source node's module
      --edge-color-mode sign: Red for positive, blue for negative

    \b
    EXAMPLES:
      # Basic modularity plot
      hlplot modular -m brain.gii -c rois.csv -x connectivity.npy -d modules.csv

      # With Q and Z scores in title
      hlplot modular -m brain.gii -c rois.csv -x connectivity.npy -d modules.csv \\
        --q-score 0.452 --z-score 3.21

      # Module-colored edges (same color as nodes)
      hlplot modular -m brain.gii -c rois.csv -x connectivity.npy -d modules.csv \\
        --edge-color-mode module

      # Sign-colored edges (red/blue for positive/negative)
      hlplot modular -m brain.gii -c rois.csv -x connectivity.npy -d modules.csv \\
        --edge-color-mode sign

      # Export publication-quality image
      hlplot modular -m brain.gii -c rois.csv -x connectivity.npy -d modules.csv \\
        --export-image figure.png --image-dpi 300 --camera superior

      # Color edges from a custom color matrix (overrides --edge-color-mode)
      hlplot modular -m brain.gii -c rois.csv -x connectivity.npy -d modules.csv \\
        --edge-color-matrix edge_colors.csv

      # Plot a p-value matrix (-log10 transform, drop p > 0.05)
      hlplot modular -m brain.gii -c rois.csv -x pvalues.csv -d modules.csv \\
        --matrix-type pvalue --pvalue-threshold 0.05

      # Signed p-values: pos effects red, neg effects blue
      hlplot modular -m brain.gii -c rois.csv -x pvalues.csv -d modules.csv \\
        --matrix-type pvalue --sign-matrix effect_signs.csv \\
        --edge-color-mode sign
    """
    try:
        # Import here to avoid slow startup
        import numpy as np
        import pandas as pd
        from HarrisLabPlotting import (
            load_mesh_file,
            create_brain_connectivity_plot_with_modularity,
            load_connectivity_input
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

        print_info(f"Loading module assignments from {modules}...")
        # Load module assignments - handle various formats
        modules_path = Path(modules)
        if modules_path.suffix.lower() == '.npy':
            module_assignments = np.load(modules)
        elif modules_path.suffix.lower() in ['.csv', '.txt']:
            module_data = pd.read_csv(modules)
            # Check if it has a 'module' column
            if 'module' in module_data.columns:
                module_assignments = module_data['module'].values
            elif module_data.shape[1] == 1:
                module_assignments = module_data.iloc[:, 0].values
            else:
                # Assume first numeric column is the assignments
                module_assignments = module_data.iloc[:, 0].values
        else:
            module_assignments = np.loadtxt(modules)

        n_modules = len(np.unique(module_assignments))
        print_success(f"Loaded {len(module_assignments)} assignments across {n_modules} modules")

        # Check dimension compatibility
        if connectivity.shape[0] != len(roi_df):
            print_warning(f"Matrix size ({connectivity.shape[0]}) differs from ROI count ({len(roi_df)})")
            print_info("Consider using map_coordinate() to align ROI coordinates with your matrix")

        if len(module_assignments) != connectivity.shape[0]:
            print_warning(f"Module count ({len(module_assignments)}) differs from matrix size ({connectivity.shape[0]})")

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
            center = (0.0, 0.0, 0.0)
            up = (0.0, 0.0, 1.0)
            segs = [s.strip() for s in tail.split(';') if s.strip()]
            if not segs:
                print_error(f"--custom-view {view_name!r} has no eye coordinates")
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

        custom_view_registry: dict = {}
        for cv in custom_views:
            name, cam = _parse_custom_view_spec(cv)
            custom_view_registry[name] = cam

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

        # Parse --export-size 'w,h'
        try:
            es_parts = [int(p.strip()) for p in export_size.split(',')]
            if len(es_parts) != 2 or es_parts[0] < 1 or es_parts[1] < 1:
                raise ValueError
            export_size_val = (es_parts[0], es_parts[1])
        except ValueError:
            print_error(f"--export-size must be 'width,height' (got {export_size!r})")
            raise click.Abort()

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

        # Parse --multi-view-grid 'rows,cols'. None == 1xN row (default).
        multi_view_grid_val = None
        if multi_view_grid:
            try:
                grid_parts = [int(p.strip()) for p in multi_view_grid.split(',')]
                if len(grid_parts) != 2 or grid_parts[0] < 1 or grid_parts[1] < 1:
                    raise ValueError
                multi_view_grid_val = (grid_parts[0], grid_parts[1])
            except ValueError:
                print_error(
                    f"--multi-view-grid must be 'rows,cols' with both values >= 1 "
                    f"(got {multi_view_grid!r})"
                )
                raise click.Abort()
            n_views = len(multi_view_list) if multi_view_list else 0
            if multi_view_grid_val[0] * multi_view_grid_val[1] < n_views:
                print_error(
                    f"--multi-view-grid {multi_view_grid_val} has only "
                    f"{multi_view_grid_val[0] * multi_view_grid_val[1]} cells "
                    f"but --multi-view supplied {n_views} views."
                )
                raise click.Abort()

        # Parse --custom-camera-* flags. The eye flag drives everything;
        # center/up/name only matter when eye is provided.
        custom_camera_dict = None
        if custom_camera_eye:
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

        # ----- voxel-map overlays -------------------------------------
        _vol_overlays = None
        if volume_paths or volume_spec:
            from HarrisLabPlotting.volume import load_volume_spec as _lvs

            def _spread(vals, n, name):
                if not vals:
                    return [None] * n
                if len(vals) == 1:
                    return [vals[0]] * n
                if len(vals) != n:
                    raise click.BadParameter(
                        f"{name} given {len(vals)} times but there are {n} "
                        f"overlay volumes; give it once or once per volume.")
                return list(vals)

            _vspecs = list(_lvs(volume_spec)) if volume_spec else []
            _vspecs += [{"path": p} for p in volume_paths]
            _n = len(_vspecs)
            _per = {
                "cmap": _spread(volume_cmaps, _n, "--volume-cmap"),
                "name": _spread(volume_names, _n, "--volume-name"),
                "threshold": _spread(volume_thresholds, _n, "--volume-threshold"),
                "smooth_fwhm": _spread(volume_smooths, _n, "--volume-smooth-fwhm"),
                "step": _spread(volume_steps, _n, "--volume-step"),
                "opacity": _spread(volume_opacities, _n, "--volume-opacity"),
            }
            for _i, _s in enumerate(_vspecs):
                for _k, _v in _per.items():
                    if _v[_i] is not None:
                        _s[_k] = _v[_i]
            _vol_overlays = _vspecs
            print_info(f"Overlaying {_n} volume(s) under the network")

        print_info("Creating modularity visualization...")

        # Create the plot
        fig, stats = create_brain_connectivity_plot_with_modularity(
            vertices=vertices,
            faces=faces,
            roi_coords_df=roi_df,
            connectivity_matrix=connectivity,
            module_assignments=module_assignments,
            plot_title=title,
            save_path=output,
            Q_score=q_score,
            Z_score=z_score,
            node_size=node_size_val,
            node_border_color=node_border_color,
            edge_color_mode=edge_color_mode,
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
            background_color=background_color,
            export_size=export_size_val,
            export_autocrop=export_autocrop,
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
            multi_view_grid=multi_view_grid_val,
            zoom=zoom,
            volume_overlays=_vol_overlays,
            directed=directed,
            matrix_orientation=matrix_orientation,
            symmetry_tol=symmetry_tol,
            arrow_view_mode=arrow_view_mode,
            arrow_size=arrow_size,
            arrow_slenderness=arrow_slenderness,
            arrow_max_edge_frac=arrow_max_edge_frac,
            arrow_min_radius_px=arrow_min_radius_px,
            arc_bow_frac=arc_bow_frac,
            arc_bow_floor=arc_bow_floor,
            arrow_darken=arrow_darken,
            no_html=no_html,
            node_roles=node_roles,
            node_size_mode=node_size_mode,
            base_node_size=base_node_size,
            max_node_multiplier=max_node_multiplier,
            border_width=border_width,
            viz_type=viz_type,
            inter_edge_color=inter_edge_color,
            show_node_labels=_parse_show_node_labels_arg(show_node_labels),
        )

        if no_html:
            print_info("--no-html: interactive HTML not written")
        else:
            print_success(f"Saved interactive visualization to {output}")

        if export_image:
            print_success(f"Exported static image to {export_image}")

        # Display statistics
        console.print()
        display_stats = {
            "Total nodes": stats.get("total_nodes", "N/A"),
            "Total edges": stats.get("total_edges", "N/A"),
            "Number of modules": stats.get("n_modules", "N/A"),
            "Edge color mode": stats.get("edge_color_mode", "N/A"),
            "Q score": stats.get("Q_score", "Not provided"),
            "Z score": stats.get("Z_score", "Not provided"),
        }
        table = create_stats_table(display_stats, title="Modularity Statistics")
        console.print(table)

        # Show module sizes
        if "module_sizes" in stats:
            console.print()
            console.print("[bold]Module Sizes:[/bold]")
            for mod_name, size in stats["module_sizes"].items():
                console.print(f"  {mod_name}: {size} nodes")

        # Optionally open in browser
        if show:
            import webbrowser
            output_path = Path(output).resolve()
            webbrowser.open(f"file://{output_path}")

    except Exception as e:
        print_error(f"Error creating modularity plot: {e}")
        raise click.Abort()
