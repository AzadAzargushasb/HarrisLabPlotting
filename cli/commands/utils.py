"""
Utility commands for matrix processing and data conversion.
"""

import click
from pathlib import Path

from ..console import console, print_success, print_error, print_warning, print_info, create_stats_table


@click.group()
def utils():
    """
    Utility commands for data processing.

    Commands for thresholding matrices, converting file formats,
    and other data manipulation tasks.
    """
    pass


@utils.command("threshold")
@click.option("--matrix", "-m", required=True, type=click.Path(exists=True),
              help="Input connectivity matrix file.")
@click.option("--output", "-o", required=True, type=click.Path(),
              help="Output file for thresholded matrix.")
@click.option("--top-n", default=None, type=int,
              help="Keep only top N edges by absolute weight.")
@click.option("--percentile", default=None, type=float,
              help="Keep edges above this percentile (0-100).")
@click.option("--absolute", default=None, type=float,
              help="Keep edges with absolute weight above this value.")
def threshold_cmd(matrix, output, top_n, percentile, absolute):
    """
    Threshold a connectivity matrix.

    Apply thresholding to keep only the strongest connections.
    Only one threshold method can be used at a time.

    \b
    Examples:
      # Keep top 100 edges
      hlplot utils threshold --matrix conn.npy --output thresh.npy --top-n 100

      # Keep top 10% of edges
      hlplot utils threshold --matrix conn.npy --output thresh.npy --percentile 90

      # Keep edges above absolute value 0.5
      hlplot utils threshold --matrix conn.npy --output thresh.npy --absolute 0.5
    """
    try:
        import numpy as np
        from HarrisLabPlotting import threshold_matrix_top_n, load_connectivity_input

        # Check that exactly one threshold method is specified
        methods = [top_n is not None, percentile is not None, absolute is not None]
        if sum(methods) != 1:
            print_error("Specify exactly one threshold method: --top-n, --percentile, or --absolute")
            raise click.Abort()

        print_info(f"Loading matrix from {matrix}...")
        mat = load_connectivity_input(matrix)
        print_info(f"Matrix shape: {mat.shape}")

        # Count original edges
        original_edges = np.sum(mat != 0)
        print_info(f"Original non-zero edges: {original_edges}")

        if top_n is not None:
            print_info(f"Thresholding to top {top_n} edges...")
            result = threshold_matrix_top_n(mat, top_n)

        elif percentile is not None:
            print_info(f"Thresholding to top {100-percentile:.1f}% of edges...")
            abs_mat = np.abs(mat)
            threshold = np.percentile(abs_mat[abs_mat > 0], percentile)
            result = mat.copy()
            result[abs_mat < threshold] = 0

        else:  # absolute
            print_info(f"Thresholding with absolute value > {absolute}...")
            result = mat.copy()
            result[np.abs(result) < absolute] = 0

        # Count remaining edges
        remaining_edges = np.sum(result != 0)
        print_info(f"Remaining non-zero edges: {remaining_edges}")

        # Save result
        output_path = Path(output)
        if output_path.suffix.lower() == '.csv':
            import pandas as pd
            pd.DataFrame(result).to_csv(output, index=False, header=False)
        else:
            np.save(output, result)

        print_success(f"Saved thresholded matrix to {output}")

    except Exception as e:
        print_error(f"Error thresholding matrix: {e}")
        raise click.Abort()


@utils.command("convert-node-edge")
@click.option("--node", "-n", required=True, type=click.Path(exists=True),
              help="Input node file (.node format).")
@click.option("--edge", "-e", required=True, type=click.Path(exists=True),
              help="Input edge file (.edge format).")
@click.option("--output", "-o", required=True, type=click.Path(),
              help="Output matrix file (.npy or .csv).")
def convert_node_edge(node, edge, output):
    """
    Convert node/edge files to a connectivity matrix.

    Converts BrainNet Viewer format (.node, .edge) to numpy array.

    \b
    Examples:
      hlplot utils convert-node-edge --node data.node --edge data.edge --output matrix.npy
    """
    try:
        from HarrisLabPlotting import load_node_file, load_edge_file, node_edge_to_roi_matrix

        print_info(f"Loading node file from {node}...")
        node_data = load_node_file(node)
        n_nodes = len(node_data)
        print_success(f"Loaded {n_nodes} nodes")

        print_info(f"Loading edge file from {edge}...")
        edge_data = load_edge_file(edge)
        print_success(f"Loaded edge data")

        print_info("Converting to matrix format...")
        matrix = node_edge_to_roi_matrix(node_data, edge_data, n_nodes)

        # Save
        output_path = Path(output)
        if output_path.suffix.lower() == '.csv':
            import pandas as pd
            pd.DataFrame(matrix).to_csv(output, index=False, header=False)
        else:
            import numpy as np
            np.save(output, matrix)

        print_success(f"Saved {matrix.shape} matrix to {output}")

    except Exception as e:
        print_error(f"Error converting files: {e}")
        raise click.Abort()


@utils.command("info")
@click.option("--matrix", "-m", required=True, type=click.Path(exists=True),
              help="Connectivity matrix file.")
def matrix_info(matrix):
    """
    Display information about a connectivity matrix.

    \b
    Examples:
      hlplot utils info --matrix connectivity.npy
    """
    try:
        import numpy as np
        from HarrisLabPlotting import load_connectivity_input

        print_info(f"Loading matrix from {matrix}...")
        mat = load_connectivity_input(matrix)

        # Compute statistics
        stats = {
            "Shape": f"{mat.shape[0]} x {mat.shape[1]}",
            "Data type": str(mat.dtype),
            "Non-zero values": np.sum(mat != 0),
            "Density": f"{100 * np.sum(mat != 0) / mat.size:.2f}%",
            "Min value": mat.min(),
            "Max value": mat.max(),
            "Mean (non-zero)": mat[mat != 0].mean() if np.any(mat != 0) else 0,
            "Std (non-zero)": mat[mat != 0].std() if np.any(mat != 0) else 0,
            "Symmetric": np.allclose(mat, mat.T),
            "Positive edges": np.sum(mat > 0),
            "Negative edges": np.sum(mat < 0),
        }

        # Check for diagonal values
        if mat.shape[0] == mat.shape[1]:
            diag_sum = np.trace(mat)
            stats["Diagonal sum"] = diag_sum

        console.print()
        table = create_stats_table(stats, title="Matrix Information")
        console.print(table)

    except Exception as e:
        print_error(f"Error reading matrix: {e}")
        raise click.Abort()


@utils.command("convert")
@click.option("--input", "-i", "input_file", required=True, type=click.Path(exists=True),
              help="Input file path.")
@click.option("--output", "-o", required=True, type=click.Path(),
              help="Output file path.")
def convert_format(input_file, output):
    """
    Convert between file formats.

    Supports conversion between: .npy, .csv, .txt

    \b
    Examples:
      # Convert numpy to CSV
      hlplot utils convert --input matrix.npy --output matrix.csv

      # Convert CSV to numpy
      hlplot utils convert --input matrix.csv --output matrix.npy
    """
    try:
        import numpy as np
        import pandas as pd

        input_path = Path(input_file)
        output_path = Path(output)

        print_info(f"Loading from {input_file}...")

        # Load based on input format
        if input_path.suffix.lower() == '.npy':
            data = np.load(input_file)
        elif input_path.suffix.lower() == '.csv':
            data = pd.read_csv(input_file, header=None).values
        elif input_path.suffix.lower() == '.txt':
            data = np.loadtxt(input_file)
        else:
            print_error(f"Unsupported input format: {input_path.suffix}")
            raise click.Abort()

        print_info(f"Loaded data with shape: {data.shape}")

        # Save based on output format
        if output_path.suffix.lower() == '.npy':
            np.save(output, data)
        elif output_path.suffix.lower() == '.csv':
            pd.DataFrame(data).to_csv(output, index=False, header=False)
        elif output_path.suffix.lower() == '.txt':
            np.savetxt(output, data)
        else:
            print_error(f"Unsupported output format: {output_path.suffix}")
            raise click.Abort()

        print_success(f"Converted to {output}")

    except Exception as e:
        print_error(f"Error converting file: {e}")
        raise click.Abort()


@utils.command("validate")
@click.option("--mesh", "-m", type=click.Path(exists=True),
              help="Mesh file to validate.")
@click.option("--coords", "-c", type=click.Path(exists=True),
              help="ROI coordinates file to validate.")
@click.option("--matrix", "-x", type=click.Path(exists=True),
              help="Connectivity matrix to validate.")
@click.option("--modules", "-d", type=click.Path(exists=True),
              help="Module assignments file to validate.")
def validate_files(mesh, coords, matrix, modules):
    """
    Validate input files for compatibility.

    Checks that files can be loaded and are compatible with each other.

    \b
    Examples:
      # Validate all files
      hlplot utils validate --mesh brain.gii --coords rois.csv --matrix conn.npy

      # Validate specific file
      hlplot utils validate --matrix conn.npy
    """
    try:
        import numpy as np
        import pandas as pd

        results = []
        n_rois_mesh = None
        n_rois_coords = None
        n_rois_matrix = None

        if mesh:
            print_info(f"Validating mesh: {mesh}...")
            try:
                from HarrisLabPlotting import load_mesh_file
                vertices, faces = load_mesh_file(mesh)
                results.append(("Mesh", "valid", f"{len(vertices)} vertices, {len(faces)} faces"))
            except Exception as e:
                results.append(("Mesh", "invalid", str(e)))

        if coords:
            print_info(f"Validating coordinates: {coords}...")
            try:
                df = pd.read_csv(coords)
                n_rois_coords = len(df)
                results.append(("Coordinates", "valid", f"{n_rois_coords} ROIs"))
            except Exception as e:
                results.append(("Coordinates", "invalid", str(e)))

        if matrix:
            print_info(f"Validating matrix: {matrix}...")
            try:
                from HarrisLabPlotting import load_connectivity_input
                mat = load_connectivity_input(matrix)
                n_rois_matrix = mat.shape[0]
                results.append(("Matrix", "valid", f"{mat.shape} shape"))
            except Exception as e:
                results.append(("Matrix", "invalid", str(e)))

        if modules:
            print_info(f"Validating modules: {modules}...")
            try:
                mod_path = Path(modules)
                if mod_path.suffix.lower() == '.npy':
                    mod_data = np.load(modules)
                else:
                    mod_data = pd.read_csv(modules, header=None).iloc[:, 0].values
                n_modules = len(np.unique(mod_data))
                results.append(("Modules", "valid", f"{len(mod_data)} assignments, {n_modules} modules"))
            except Exception as e:
                results.append(("Modules", "invalid", str(e)))

        # Display results
        console.print()
        console.print("[bold]Validation Results:[/bold]")
        for name, status, details in results:
            if status == "valid":
                console.print(f"  [green]OK[/green] {name}: {details}")
            else:
                console.print(f"  [red]X[/red] {name}: {details}")

        # Check compatibility
        if n_rois_coords is not None and n_rois_matrix is not None:
            console.print()
            if n_rois_coords == n_rois_matrix:
                console.print(f"  [green]OK[/green] Coords and matrix compatible ({n_rois_coords} ROIs)")
            else:
                console.print(f"  [red]X[/red] Coords ({n_rois_coords}) and matrix ({n_rois_matrix}) ROI count mismatch")

    except Exception as e:
        print_error(f"Error during validation: {e}")
        raise click.Abort()
