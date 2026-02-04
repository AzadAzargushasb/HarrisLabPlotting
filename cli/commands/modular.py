"""
Modularity visualization commands.
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
def modular(mesh, coords, matrix, modules, output, title, q_score, z_score,
            node_size, node_border_color,
            edge_color_mode, pos_edge_color, neg_edge_color,
            edge_width_min, edge_width_max, edge_width_fixed, edge_threshold,
            mesh_color, mesh_opacity, label_font_size, fast_render,
            camera, enable_camera_controls, show_only_connected,
            hide_nodes_with_hidden_edges, node_metrics,
            export_image, image_format, image_dpi,
            export_show_title, export_show_legend, show):
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
