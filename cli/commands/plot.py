"""
Basic brain connectivity plotting commands.
"""

import click
from pathlib import Path
import os

from ..console import console, print_success, print_error, print_warning, print_info, create_stats_table
from ..config_loader import load_config, validate_config, resolve_paths, ConfigError


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
@click.option("--output", "-o", default=None, type=click.Path(),
              help="Output file path. Default: connectivity_plot.html")
@click.option("--title", "-t", default="Brain Connectivity",
              help="Plot title.")
@click.option("--node-size", default=10.0, type=float,
              help="Size of ROI nodes. Can also be path to size vector file.")
@click.option("--node-color", default="purple",
              help="Node color. Can be: color name, hex code, or path to color/module file.")
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
@click.option("--config", "config_file", default=None, type=click.Path(exists=True),
              help="Path to YAML config file. CLI options override config values.")
@click.option("--show/--no-show", default=False,
              help="Open the plot in browser after creation (HTML only).")
def plot(mesh, coords, matrix, output, title, node_size, node_color,
         edge_threshold, edge_width_min, edge_width_max, opacity,
         camera, top_n, output_format, width, height, config_file, show):
    """
    Create a basic brain connectivity plot.

    \b
    Examples:
      # Basic plot with default settings
      hlplot plot --mesh brain.gii --coords rois.csv --matrix connectivity.npy

      # Customized plot
      hlplot plot -m brain.gii -c rois.csv -x connectivity.npy \\
        --title "My Study" --camera lateral-left --opacity 0.5

      # Export as PNG
      hlplot plot -m brain.gii -c rois.csv -x connectivity.npy \\
        --format png --output my_plot.png --width 1600 --height 1200

      # Use configuration file
      hlplot plot --config my_config.yaml
    """
    try:
        # Import here to avoid slow startup
        import numpy as np
        import pandas as pd
        from HarrisLabPlotting import (
            load_mesh_file,
            create_brain_connectivity_plot,
            load_connectivity_input,
            CameraController,
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

        # Apply top-N threshold if specified
        if top_n is not None:
            print_info(f"Thresholding to top {top_n} edges...")
            connectivity = threshold_matrix_top_n(connectivity, top_n)

        # Determine output path
        if output is None:
            output = f"connectivity_plot.{output_format}"

        print_info("Creating brain connectivity plot...")

        # Create the plot
        fig, stats = create_brain_connectivity_plot(
            vertices=vertices,
            faces=faces,
            roi_coords_df=roi_df,
            connectivity_matrix=connectivity,
            plot_title=title,
            node_size=node_size,
            node_color=node_color,
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
        table = create_stats_table(stats, title="Plot Statistics")
        console.print(table)

        # Optionally open in browser
        if show and output_format == "html":
            import webbrowser
            webbrowser.open(f"file://{output_path.resolve()}")

    except Exception as e:
        print_error(f"Error creating plot: {e}")
        raise click.Abort()
