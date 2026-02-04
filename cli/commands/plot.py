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
@click.option("--edge-threshold", default=0.0, type=float,
              help="Minimum absolute edge weight to display. Default: 0.0 (show all non-zero)")

# === Brain mesh appearance ===
@click.option("--mesh-color", default="lightgray",
              help="Color of the brain mesh surface. Default: lightgray")
@click.option("--mesh-opacity", default=0.15, type=float,
              help="Opacity of brain mesh (0-1). Lower values show more of the network. Default: 0.15")

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

# === Convenience ===
@click.option("--show/--no-show", default=False,
              help="Open the HTML file in browser after creation.")
def plot(mesh, coords, matrix, output, title, node_size, node_color,
         node_border_color, pos_edge_color, neg_edge_color,
         edge_width_min, edge_width_max, edge_width_fixed, edge_threshold,
         mesh_color, mesh_opacity, label_font_size, fast_render,
         camera, enable_camera_controls, show_only_connected,
         hide_nodes_with_hidden_edges, node_metrics,
         export_image, image_format, image_dpi,
         export_show_title, export_show_legend, show):
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
            export_show_legend=export_show_legend
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
