"""
Batch processing commands for multiple subjects/plots.
"""

import click
from pathlib import Path
import os

from ..console import (
    console, print_success, print_error, print_warning, print_info,
    create_stats_table, create_progress_context
)
from ..config_loader import load_config, validate_config, resolve_paths, ConfigError


@click.command()
@click.option("--config", "-c", required=True, type=click.Path(exists=True),
              help="Path to YAML configuration file with batch settings.")
@click.option("--output-dir", "-o", default=None, type=click.Path(),
              help="Output directory for all plots. Overrides config setting.")
@click.option("--format", "output_format", default=None,
              type=click.Choice(["html", "png", "pdf", "svg", "jpeg", "webp"]),
              help="Output file format for all plots. Overrides config setting.")
@click.option("--parallel/--sequential", default=False,
              help="Process subjects in parallel (experimental).")
@click.option("--dry-run", is_flag=True,
              help="Show what would be processed without creating plots.")
def batch(config, output_dir, output_format, parallel, dry_run):
    """
    Process multiple subjects/plots from a configuration file.

    The configuration file should contain a 'batch' section listing
    each subject with their specific data files.

    \b
    Example config (batch_config.yaml):
      mesh_file: "data/brain.gii"
      roi_coords_file: "data/rois.csv"
      output_dir: "./outputs"

      plot:
        opacity: 0.3
        camera_view: anterior

      batch:
        - name: "subject_01"
          matrix: "data/sub01_connectivity.npy"
          modules: "data/sub01_modules.csv"
        - name: "subject_02"
          matrix: "data/sub02_connectivity.npy"
          modules: "data/sub02_modules.csv"

    \b
    Examples:
      # Basic batch processing
      hlplot batch --config batch_config.yaml

      # Override output directory
      hlplot batch --config batch_config.yaml --output-dir ./results/

      # Dry run to check configuration
      hlplot batch --config batch_config.yaml --dry-run
    """
    try:
        print_info(f"Loading configuration from {config}...")
        cfg = load_config(config)

        # Resolve paths relative to config file location
        config_dir = str(Path(config).parent)
        cfg = resolve_paths(cfg, config_dir)

        # Apply CLI overrides
        if output_dir:
            cfg["output_dir"] = output_dir
        if output_format:
            cfg["output_format"] = output_format

        # Validate configuration
        errors = validate_config(cfg)
        if errors:
            print_error("Configuration errors:")
            for error in errors:
                console.print(f"  [red]•[/red] {error}")
            raise click.Abort()

        # Check for batch section
        if "batch" not in cfg or not cfg["batch"]:
            print_error("No 'batch' section found in configuration file.")
            print_info("Add a batch section with subject definitions.")
            raise click.Abort()

        batch_items = cfg["batch"]
        print_success(f"Found {len(batch_items)} subjects to process")

        # Ensure output directory exists
        out_dir = Path(cfg.get("output_dir", "./outputs"))
        out_dir.mkdir(parents=True, exist_ok=True)

        if dry_run:
            print_warning("DRY RUN - No plots will be created")
            console.print()
            console.print("[bold]Subjects to process:[/bold]")
            for item in batch_items:
                console.print(f"  [cyan]{item['name']}[/cyan]")
                console.print(f"    Matrix: {item['matrix']}")
                if "modules" in item:
                    console.print(f"    Modules: {item['modules']}")
            console.print()
            console.print(f"[bold]Output directory:[/bold] {out_dir}")
            console.print(f"[bold]Output format:[/bold] {cfg.get('output_format', 'html')}")
            return

        # Import plotting functions
        import numpy as np
        import pandas as pd
        from HarrisLabPlotting import (
            load_mesh_file,
            create_brain_connectivity_plot,
            create_brain_connectivity_plot_with_modularity,
            load_connectivity_input
        )

        # Load shared resources
        print_info(f"Loading mesh from {cfg['mesh_file']}...")
        vertices, faces = load_mesh_file(cfg["mesh_file"])

        print_info(f"Loading ROI coordinates from {cfg['roi_coords_file']}...")
        roi_df = pd.read_csv(cfg["roi_coords_file"])

        # Process each subject
        processed = 0
        failed = 0
        results = []

        with create_progress_context("Processing subjects...") as progress:
            task = progress.add_task("Processing...", total=len(batch_items))

            for item in batch_items:
                name = item["name"]
                progress.update(task, description=f"Processing {name}...")

                try:
                    # Load subject-specific matrix
                    matrix = load_connectivity_input(item["matrix"])

                    # Determine if this is a modularity plot
                    has_modules = "modules" in item and item["modules"]

                    # Get plot settings
                    plot_cfg = cfg.get("plot", {})
                    camera_cfg = cfg.get("camera", {})
                    mod_cfg = cfg.get("modularity", {})

                    # Create output filename
                    fmt = cfg.get("output_format", "html")
                    output_file = out_dir / f"{name}.{fmt}"

                    if has_modules:
                        # Load module assignments
                        mod_path = Path(item["modules"])
                        if mod_path.suffix.lower() == '.npy':
                            modules = np.load(item["modules"])
                        else:
                            mod_data = pd.read_csv(item["modules"], header=None)
                            modules = mod_data.iloc[:, 0].values

                        fig, stats = create_brain_connectivity_plot_with_modularity(
                            vertices=vertices,
                            faces=faces,
                            roi_coords_df=roi_df,
                            connectivity_matrix=matrix,
                            module_assignments=modules,
                            plot_title=item.get("title", f"Subject: {name}"),
                            Q_score=item.get("q_score"),
                            Z_score=item.get("z_score"),
                            edge_color_mode=mod_cfg.get("edge_color_mode", "module"),
                            node_size=plot_cfg.get("node_size", 10),
                            edge_threshold=plot_cfg.get("edge_threshold", 0),
                            opacity=plot_cfg.get("opacity", 0.3),
                            camera_view=camera_cfg.get("view", "anterior")
                        )
                    else:
                        fig, stats = create_brain_connectivity_plot(
                            vertices=vertices,
                            faces=faces,
                            roi_coords_df=roi_df,
                            connectivity_matrix=matrix,
                            plot_title=item.get("title", f"Subject: {name}"),
                            node_size=plot_cfg.get("node_size", 10),
                            node_color=plot_cfg.get("node_color", "purple"),
                            edge_threshold=plot_cfg.get("edge_threshold", 0),
                            opacity=plot_cfg.get("opacity", 0.3),
                            camera_view=camera_cfg.get("view", "anterior")
                        )

                    # Save figure
                    if fmt == "html":
                        fig.write_html(str(output_file))
                    else:
                        fig.write_image(str(output_file),
                                       width=cfg.get("width", 1200),
                                       height=cfg.get("height", 900))

                    processed += 1
                    results.append({"name": name, "status": "success", "output": str(output_file)})

                except Exception as e:
                    failed += 1
                    results.append({"name": name, "status": "failed", "error": str(e)})
                    print_warning(f"Failed to process {name}: {e}")

                progress.advance(task)

        # Summary
        console.print()
        print_success(f"Batch processing complete: {processed} succeeded, {failed} failed")

        if results:
            console.print()
            console.print("[bold]Results:[/bold]")
            for r in results:
                if r["status"] == "success":
                    console.print(f"  [green]OK[/green] {r['name']}: {r['output']}")
                else:
                    console.print(f"  [red]X[/red] {r['name']}: {r['error']}")

    except ConfigError as e:
        print_error(f"Configuration error: {e}")
        raise click.Abort()
    except Exception as e:
        print_error(f"Error during batch processing: {e}")
        raise click.Abort()
