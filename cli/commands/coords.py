"""
ROI coordinate utility commands.
"""

import click
from pathlib import Path

from ..console import console, print_success, print_error, print_warning, print_info, create_stats_table


@click.group()
def coords():
    """
    ROI coordinate utilities.

    Commands for loading, validating, and manipulating ROI coordinate files.
    """
    pass


@coords.command("load")
@click.option("--file", "-f", required=True, type=click.Path(exists=True),
              help="Path to ROI coordinates file (.csv format).")
@click.option("--show-stats", is_flag=True, help="Show coordinate statistics.")
@click.option("--show-head", default=0, type=int, help="Show first N rows.")
@click.option("--validate", is_flag=True, help="Validate coordinate format.")
def coords_load(file, show_stats, show_head, validate):
    """
    Load and inspect ROI coordinates file.

    \b
    Expected CSV format:
      - Columns for x, y, z coordinates
      - Optional columns for ROI names/labels
      - One row per ROI

    \b
    Examples:
      # Load and show stats
      hlplot coords load --file rois.csv --show-stats

      # Show first 10 rows
      hlplot coords load --file rois.csv --show-head 10

      # Validate format
      hlplot coords load --file rois.csv --validate
    """
    try:
        import pandas as pd
        import numpy as np

        print_info(f"Loading coordinates from {file}...")
        df = pd.read_csv(file)
        print_success(f"Loaded {len(df)} ROIs with {len(df.columns)} columns")

        # Show column info
        console.print()
        console.print("[bold]Columns:[/bold]")
        for col in df.columns:
            dtype = str(df[col].dtype)
            console.print(f"  [cyan]{col}[/cyan] ({dtype})")

        if validate:
            console.print()
            console.print("[bold]Validation:[/bold]")
            errors = []

            # Check for coordinate columns
            coord_cols = []
            for col_name in ['x', 'y', 'z', 'X', 'Y', 'Z']:
                if col_name in df.columns:
                    coord_cols.append(col_name)

            if len(coord_cols) < 3:
                # Try numeric columns
                numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
                if len(numeric_cols) >= 3:
                    print_warning(f"Using first 3 numeric columns as coordinates: {numeric_cols[:3]}")
                    coord_cols = numeric_cols[:3]
                else:
                    errors.append("Could not identify x, y, z coordinate columns")

            if coord_cols:
                # Check for NaN values
                for col in coord_cols[:3]:
                    nan_count = df[col].isna().sum()
                    if nan_count > 0:
                        errors.append(f"Column '{col}' has {nan_count} NaN values")

                # Check coordinate ranges (typical brain coordinates)
                for col in coord_cols[:3]:
                    col_min, col_max = df[col].min(), df[col].max()
                    if col_max - col_min > 500:  # Very large range
                        print_warning(f"Column '{col}' has large range: [{col_min:.1f}, {col_max:.1f}]")

            if errors:
                for error in errors:
                    console.print(f"  [red]X[/red] {error}")
            else:
                console.print("  [green]OK[/green] Coordinate file appears valid")

        if show_stats:
            console.print()
            # Get coordinate columns
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()[:3]

            stats = {}
            for col in numeric_cols:
                stats[f"{col} (min)"] = df[col].min()
                stats[f"{col} (max)"] = df[col].max()
                stats[f"{col} (mean)"] = df[col].mean()

            stats["Total ROIs"] = len(df)

            table = create_stats_table(stats, title="Coordinate Statistics")
            console.print(table)

        if show_head > 0:
            console.print()
            console.print(f"[bold]First {show_head} rows:[/bold]")
            console.print(df.head(show_head).to_string())

    except Exception as e:
        print_error(f"Error loading coordinates: {e}")
        raise click.Abort()


@coords.command("map")
@click.option("--file", "-f", required=True, type=click.Path(exists=True),
              help="Path to ROI coordinates file.")
@click.option("--output", "-o", required=True, type=click.Path(),
              help="Output file path for mapped coordinates.")
@click.option("--x-col", default=None, help="Column name for X coordinates.")
@click.option("--y-col", default=None, help="Column name for Y coordinates.")
@click.option("--z-col", default=None, help="Column name for Z coordinates.")
@click.option("--scale", default=1.0, type=float, help="Scale factor to apply.")
def coords_map(file, output, x_col, y_col, z_col, scale):
    """
    Map and transform ROI coordinates.

    Useful for standardizing coordinate files or applying transformations.

    \b
    Examples:
      # Map with custom column names
      hlplot coords map --file raw_coords.csv --output mapped.csv \\
        --x-col "coord_x" --y-col "coord_y" --z-col "coord_z"

      # Apply scaling
      hlplot coords map --file coords.csv --output scaled.csv --scale 0.001
    """
    try:
        import pandas as pd
        from HarrisLabPlotting import load_and_clean_coordinates

        print_info(f"Loading coordinates from {file}...")

        # Load with optional column mapping
        df = pd.read_csv(file)

        # Determine coordinate columns
        if x_col and y_col and z_col:
            coords_df = df[[x_col, y_col, z_col]].copy()
            coords_df.columns = ['x', 'y', 'z']
        else:
            # Try to auto-detect
            if all(c in df.columns for c in ['x', 'y', 'z']):
                coords_df = df[['x', 'y', 'z']].copy()
            elif all(c in df.columns for c in ['X', 'Y', 'Z']):
                coords_df = df[['X', 'Y', 'Z']].copy()
                coords_df.columns = ['x', 'y', 'z']
            else:
                # Use first 3 numeric columns
                numeric_cols = df.select_dtypes(include=['number']).columns[:3]
                if len(numeric_cols) < 3:
                    print_error("Could not identify coordinate columns. Use --x-col, --y-col, --z-col.")
                    raise click.Abort()
                coords_df = df[numeric_cols].copy()
                coords_df.columns = ['x', 'y', 'z']

        # Apply scaling
        if scale != 1.0:
            coords_df *= scale
            print_info(f"Applied scale factor: {scale}")

        # Save
        coords_df.to_csv(output, index=False)
        print_success(f"Saved mapped coordinates to {output}")
        print_info(f"Output has {len(coords_df)} ROIs with columns: {list(coords_df.columns)}")

    except Exception as e:
        print_error(f"Error mapping coordinates: {e}")
        raise click.Abort()


@coords.command("extract")
@click.option("--file", "-f", required=True, type=click.Path(exists=True),
              help="Path to ROI file (atlas or parcellation).")
@click.option("--output", "-o", required=True, type=click.Path(),
              help="Output CSV file for extracted coordinates.")
@click.option("--method", default="centroid",
              type=click.Choice(["centroid", "peak"]),
              help="Method for extracting coordinates.")
def coords_extract(file, output, method):
    """
    Extract ROI coordinates from atlas/parcellation files.

    \b
    Examples:
      # Extract centroids from atlas
      hlplot coords extract --file atlas.nii.gz --output rois.csv

      # Extract peak coordinates
      hlplot coords extract --file atlas.nii.gz --output rois.csv --method peak
    """
    try:
        import nibabel as nib
        import numpy as np
        import pandas as pd
        from scipy import ndimage

        print_info(f"Loading {file}...")

        # Load the atlas file
        img = nib.load(file)
        data = img.get_fdata()
        affine = img.affine

        # Get unique labels (excluding 0 which is usually background)
        labels = np.unique(data)
        labels = labels[labels != 0]

        print_info(f"Found {len(labels)} unique ROI labels")

        coordinates = []

        for label in labels:
            mask = data == label

            if method == "centroid":
                # Calculate centroid in voxel space
                center = ndimage.center_of_mass(mask)
            else:  # peak - use center of mass as well for now
                center = ndimage.center_of_mass(mask)

            # Convert to world coordinates
            voxel_coords = np.array([center[0], center[1], center[2], 1])
            world_coords = affine @ voxel_coords

            coordinates.append({
                'label': int(label),
                'x': world_coords[0],
                'y': world_coords[1],
                'z': world_coords[2]
            })

        # Create DataFrame
        coords_df = pd.DataFrame(coordinates)
        coords_df.to_csv(output, index=False)

        print_success(f"Extracted {len(coords_df)} ROI coordinates to {output}")

    except Exception as e:
        print_error(f"Error extracting coordinates: {e}")
        raise click.Abort()


@coords.command("generate")
@click.option("--volume", "-v", required=True, type=click.Path(exists=True),
              help="Path to NIfTI volume file containing ROI labels.")
@click.option("--labels", "-l", required=True, type=click.Path(exists=True),
              help="Path to text file containing ROI labels (tab-delimited: number\\tlabel).")
@click.option("--output-dir", "-o", required=True, type=click.Path(),
              help="Directory where output files will be saved.")
@click.option("--name", "-n", default="roi_coordinates",
              help="Name for the output files (without extension).")
def coords_generate(volume, labels, output_dir, name):
    """
    Generate ROI coordinates from a NIfTI volume file with labels.

    This command extracts center-of-gravity (COG) coordinates for each ROI
    defined in the volume file, using the provided label file for ROI names.

    \b
    Output files:
      - {name}_comma.csv: Comma-delimited CSV
      - {name}_tab.csv: Tab-delimited CSV
      - {name}.pkl: Pickle file for Python

    \b
    Label file format (tab-delimited):
      1\\tROI_Name_1
      2\\tROI_Name_2
      ...

    \b
    Examples:
      # Generate coordinates from a 170-ROI atlas
      hlplot coords generate \\
        --volume brain_atlas.nii.gz \\
        --labels roi_labels.txt \\
        --output-dir ./coordinates \\
        --name atlas_170_coordinates

    \b
    Pipeline Usage:
      This is typically the FIRST step after obtaining a NIfTI volume file.
      Use the generated coordinates with 'hlplot plot' or 'hlplot modular'.
    """
    try:
        from HarrisLabPlotting import coordinate_function

        print_info(f"Generating coordinates from volume: {volume}")
        print_info(f"Using labels from: {labels}")
        print_info(f"Output directory: {output_dir}")

        # Call the coordinate_function
        df = coordinate_function(
            volume_file_location=volume,
            roi_label_file=labels,
            name_of_file=name,
            save_directory=output_dir
        )

        console.print()
        print_success(f"Generated coordinates for {len(df)} ROIs")
        print_info(f"Output files saved to: {output_dir}")

    except Exception as e:
        print_error(f"Error generating coordinates: {e}")
        raise click.Abort()


@coords.command("map-subset")
@click.option("--coords", "-c", required=True, type=click.Path(exists=True),
              help="Path to full ROI coordinates file (CSV or pickle).")
@click.option("--subset", "-s", required=True, type=click.Path(exists=True),
              help="Path to subset ROI list (.node file, .txt, or .csv).")
@click.option("--output-dir", "-o", required=True, type=click.Path(),
              help="Directory where output files will be saved.")
@click.option("--name", "-n", default="mapped_roi_coordinates",
              help="Name for the output files (without extension).")
def coords_map_subset(coords, subset, output_dir, name):
    """
    Map a subset of ROIs to their coordinates from a full coordinate set.

    Use this when your connectivity matrix has fewer ROIs than your full atlas.
    This command extracts only the coordinates that match your subset.

    \b
    Supported subset file formats:
      - .node: BrainNet Viewer node file (uses last column as ROI names)
      - .txt: Text file with ROI names (one per line, or tab-delimited: index\\tname)
      - .csv: CSV file with 'roi_name' column

    \b
    Output files:
      - {name}_comma.csv: Comma-delimited CSV with mapped coordinates
      - {name}_tab.csv: Tab-delimited CSV
      - {name}.pkl: Pickle file for Python

    \b
    Examples:
      # Map 28 ROIs from a 170-ROI atlas using a .node file
      hlplot coords map-subset \\
        --coords atlas_170_coordinates.csv \\
        --subset my_28_rois.node \\
        --output-dir ./mapped \\
        --name atlas_28_mapped

      # Map using a text file with ROI names
      hlplot coords map-subset \\
        --coords atlas_170_coordinates.csv \\
        --subset roi_subset.txt \\
        --output-dir ./mapped

    \b
    Pipeline Usage:
      Use this AFTER 'hlplot coords generate' when you need to work with
      a subset of ROIs (e.g., when your connectivity matrix is smaller
      than your full atlas).
    """
    try:
        from HarrisLabPlotting import map_coordinate

        print_info(f"Mapping coordinates from: {coords}")
        print_info(f"Using subset list from: {subset}")
        print_info(f"Output directory: {output_dir}")

        # Call the map_coordinate function
        mapped_df, unmapped_rois = map_coordinate(
            original_coords_file=coords,
            reduced_roi_file=subset,
            save_directory=output_dir,
            name_of_file=name
        )

        console.print()
        if unmapped_rois:
            print_warning(f"{len(unmapped_rois)} ROI(s) could not be mapped")
        else:
            print_success(f"Successfully mapped {len(mapped_df)} ROIs")
        print_info(f"Output files saved to: {output_dir}")

    except Exception as e:
        print_error(f"Error mapping coordinates: {e}")
        raise click.Abort()
