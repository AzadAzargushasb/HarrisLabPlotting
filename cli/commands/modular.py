"""
Modularity visualization commands.
"""

import click
from pathlib import Path

from ..console import console, print_success, print_error, print_warning, print_info, create_stats_table


# Common options for camera views
CAMERA_VIEWS = ["anterior", "posterior", "lateral-left", "lateral-right",
                "superior", "inferior", "dorsal", "ventral"]


@click.command()
@click.option("--mesh", "-m", required=True, type=click.Path(exists=True),
              help="Path to brain mesh file (.gii format).")
@click.option("--coords", "-c", required=True, type=click.Path(exists=True),
              help="Path to ROI coordinates file (.csv format).")
@click.option("--matrix", "-x", required=True, type=click.Path(exists=True),
              help="Path to connectivity matrix (.npy, .csv, or .txt format).")
@click.option("--modules", "-d", required=True, type=click.Path(exists=True),
              help="Path to module assignments file (.csv, .npy, or .txt format).")
@click.option("--output", "-o", default=None, type=click.Path(),
              help="Output file path. Default: modularity_plot.html")
@click.option("--title", "-t", default="Brain Connectivity with Modularity",
              help="Plot title.")
@click.option("--q-score", default=None, type=float,
              help="Modularity Q score to display in title.")
@click.option("--z-score", default=None, type=float,
              help="Z-rand score to display in title.")
@click.option("--edge-color-mode", default="module",
              type=click.Choice(["module", "sign"], case_sensitive=False),
              help="Edge coloring mode: 'module' (colored by module) or 'sign' (positive/negative).")
@click.option("--node-size", default=10.0, type=float,
              help="Size of ROI nodes. Can also be path to size vector file.")
@click.option("--edge-threshold", default=0.0, type=float,
              help="Minimum absolute edge weight to display.")
@click.option("--edge-width-min", default=1.0, type=float,
              help="Minimum edge line width.")
@click.option("--edge-width-max", default=5.0, type=float,
              help="Maximum edge line width.")
@click.option("--opacity", default=0.3, type=float,
              help="Brain mesh opacity (0-1).")
@click.option("--camera", default="anterior", type=click.Choice(CAMERA_VIEWS, case_sensitive=False),
              help="Camera view preset.")
@click.option("--top-n", default=None, type=int,
              help="Show only top N edges by weight.")
@click.option("--format", "output_format", default="html",
              type=click.Choice(["html", "png", "pdf", "svg", "jpeg", "webp"]),
              help="Output file format.")
@click.option("--width", default=1200, type=int,
              help="Output image width (for static formats).")
@click.option("--height", default=900, type=int,
              help="Output image height (for static formats).")
@click.option("--show/--no-show", default=False,
              help="Open the plot in browser after creation (HTML only).")
def modular(mesh, coords, matrix, modules, output, title, q_score, z_score,
            edge_color_mode, node_size, edge_threshold, edge_width_min,
            edge_width_max, opacity, camera, top_n, output_format, width, height, show):
    """
    Create a brain connectivity plot with modularity visualization.

    This command creates a visualization where nodes are colored by their
    module assignment, and edges can be colored by module or sign.

    \b
    Examples:
      # Basic modularity plot
      hlplot modular -m brain.gii -c rois.csv -x connectivity.npy -d modules.csv

      # With Q and Z scores in title
      hlplot modular -m brain.gii -c rois.csv -x connectivity.npy -d modules.csv \\
        --q-score 0.45 --z-score 3.2

      # Edge coloring by sign (positive/negative)
      hlplot modular -m brain.gii -c rois.csv -x connectivity.npy -d modules.csv \\
        --edge-color-mode sign

      # Export as high-resolution PNG
      hlplot modular -m brain.gii -c rois.csv -x connectivity.npy -d modules.csv \\
        --format png --width 2400 --height 1800
    """
    try:
        # Import here to avoid slow startup
        import numpy as np
        import pandas as pd
        from HarrisLabPlotting import (
            load_mesh_file,
            create_brain_connectivity_plot_with_modularity,
            load_connectivity_input,
            threshold_matrix_top_n
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
            module_data = pd.read_csv(modules, header=None)
            if module_data.shape[1] == 1:
                module_assignments = module_data.iloc[:, 0].values
            else:
                # Assume first column is the assignments
                module_assignments = module_data.iloc[:, 0].values
        else:
            module_assignments = np.loadtxt(modules)

        n_modules = len(np.unique(module_assignments))
        print_success(f"Loaded {len(module_assignments)} assignments across {n_modules} modules")

        # Apply top-N threshold if specified
        if top_n is not None:
            print_info(f"Thresholding to top {top_n} edges...")
            connectivity = threshold_matrix_top_n(connectivity, top_n)

        # Determine output path
        if output is None:
            output = f"modularity_plot.{output_format}"

        print_info("Creating modularity visualization...")

        # Create the plot
        fig, stats = create_brain_connectivity_plot_with_modularity(
            vertices=vertices,
            faces=faces,
            roi_coords_df=roi_df,
            connectivity_matrix=connectivity,
            module_assignments=module_assignments,
            plot_title=title,
            Q_score=q_score,
            Z_score=z_score,
            edge_color_mode=edge_color_mode,
            node_size=node_size,
            edge_threshold=edge_threshold,
            edge_width_range=(edge_width_min, edge_width_max),
            opacity=opacity,
            camera_view=camera
        )

        # Save the figure
        output_path = Path(output)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if output_format == "html":
            fig.write_html(str(output_path))
        else:
            fig.write_image(str(output_path), width=width, height=height)

        print_success(f"Saved plot to {output_path}")

        # Display statistics
        console.print()
        table = create_stats_table(stats, title="Modularity Statistics")
        console.print(table)

        # Optionally open in browser
        if show and output_format == "html":
            import webbrowser
            webbrowser.open(f"file://{output_path.resolve()}")

    except Exception as e:
        print_error(f"Error creating modularity plot: {e}")
        raise click.Abort()
