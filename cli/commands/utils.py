"""
Utility commands for matrix processing and data conversion.
"""

import click
from pathlib import Path

from ..console import console, print_success, print_error, print_warning, print_info, create_stats_table
from ..pager import PagerGroup


@click.group(cls=PagerGroup)
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
@click.option("--keep-sign", default="both",
              type=click.Choice(["both", "positive", "negative"], case_sensitive=False),
              help=(
                  "Restrict the matrix to a single sign before/along with the magnitude "
                  "threshold. 'both' (default) keeps positive and negative edges. "
                  "'positive' zeroes out every entry < 0, keeping only positive edges. "
                  "'negative' zeroes out every entry > 0, keeping only negative edges "
                  "(their original sign is preserved). "
                  "Combine freely with --top-n / --percentile / --absolute: the sign "
                  "filter is applied FIRST, then the magnitude threshold is applied to "
                  "the remaining entries (so e.g. --keep-sign positive --top-n 50 keeps "
                  "the 50 strongest positive edges)."
              ))
def threshold_cmd(matrix, output, top_n, percentile, absolute, keep_sign):
    """
    Threshold a connectivity matrix.

    Apply thresholding to keep only the strongest connections, optionally
    restricted to a single sign (positive-only or negative-only).

    Exactly one magnitude threshold method (--top-n, --percentile, or
    --absolute) must be specified. The --keep-sign flag is independent and
    can be combined with any of them: the sign filter is applied first,
    then the magnitude threshold runs on the remaining entries.

    \b
    Examples:
      # Keep top 100 edges of any sign
      hlplot utils threshold --matrix conn.npy --output thresh.npy --top-n 100

      # Keep top 10% of edges of any sign
      hlplot utils threshold --matrix conn.npy --output thresh.npy --percentile 90

      # Keep edges above absolute value 0.5
      hlplot utils threshold --matrix conn.npy --output thresh.npy --absolute 0.5

      # Keep ONLY positive edges (zero out all negatives)
      hlplot utils threshold --matrix conn.npy --output pos.npy \\
        --keep-sign positive --absolute 0

      # Keep the 50 strongest POSITIVE edges
      hlplot utils threshold --matrix conn.npy --output pos50.npy \\
        --keep-sign positive --top-n 50

      # Keep ONLY negative edges above |0.3|
      hlplot utils threshold --matrix conn.npy --output neg.npy \\
        --keep-sign negative --absolute 0.3
    """
    try:
        import numpy as np
        from HarrisLabPlotting import (
            threshold_matrix_top_n,
            filter_matrix_by_sign,
            load_connectivity_input,
        )

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
        n_pos = int(np.sum(mat > 0))
        n_neg = int(np.sum(mat < 0))
        print_info(f"Original non-zero edges: {original_edges} ({n_pos} positive, {n_neg} negative)")

        # Apply the sign filter FIRST (so subsequent magnitude thresholds
        # operate only on the entries we want to keep).
        keep_sign = keep_sign.lower()
        if keep_sign != "both":
            print_info(f"Applying sign filter: keep_sign='{keep_sign}'")
            mat = filter_matrix_by_sign(mat, keep_sign=keep_sign)
            after_sign = int(np.sum(mat != 0))
            print_info(f"Edges after sign filter: {after_sign}")

        if top_n is not None:
            print_info(f"Thresholding to top {top_n} edges...")
            result = threshold_matrix_top_n(mat, top_n)

        elif percentile is not None:
            print_info(f"Thresholding to top {100-percentile:.1f}% of edges...")
            abs_mat = np.abs(mat)
            nonzero = abs_mat[abs_mat > 0]
            if nonzero.size == 0:
                print_warning("No non-zero entries remain to compute percentile on; output will be all zeros.")
                result = mat.copy()
            else:
                threshold = np.percentile(nonzero, percentile)
                result = mat.copy()
                result[abs_mat < threshold] = 0

        else:  # absolute
            print_info(f"Thresholding with absolute value > {absolute}...")
            result = mat.copy()
            result[np.abs(result) < absolute] = 0

        # Count remaining edges
        remaining_edges = np.sum(result != 0)
        rem_pos = int(np.sum(result > 0))
        rem_neg = int(np.sum(result < 0))
        print_info(f"Remaining non-zero edges: {remaining_edges} ({rem_pos} positive, {rem_neg} negative)")

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
@click.option("--coords", "-c", required=True, type=click.Path(exists=True),
              help="Full ROI coordinates CSV (any atlas size - 114, 170, custom - "
                   "from `hlplot coords generate` or `coords map-subset`). MUST have "
                   "at least as many rows as the .node file (and edge-matrix row "
                   "length), and MUST contain every ROI name from the .node file. "
                   "The output matrix is sized to this CSV's row count; .edge values "
                   "are placed by matching ROI names. Unmatched rows/cols are zero.")
@click.option("--output", "-o", required=True, type=click.Path(),
              help="Output matrix file (.npy or .csv).")
def convert_node_edge(node, edge, coords, output):
    """
    Embed a BrainNet Viewer (.node, .edge) pair into a full ROI atlas matrix.

    The .edge file is an n_nodes x n_nodes connectivity matrix between the
    ROIs listed in the .node file. This command embeds it into the larger
    N x N matrix defined by --coords (matched by ROI name), filling unmapped
    rows/columns with zeros. The resulting matrix lines up row-for-row with
    the coords CSV, so it can be passed straight to `hlplot plot --matrix`.

    \b
    Constraints:
      * len(coords) >= len(node)
      * edge.shape == (len(node), len(node))
      * Every roi_name in the .node file must appear in coords' roi_name column

    \b
    Examples:
      # Embed a 28-ROI subset into a 170-ROI atlas
      hlplot utils convert-node-edge \\
          --node rois_28.node \\
          --edge connectivity_28.edge \\
          --coords atlas_170_coordinates.csv \\
          --output connectivity_28_in_170.csv
    """
    try:
        from HarrisLabPlotting import load_node_file, load_edge_file, node_edge_to_roi_matrix
        import pandas as pd

        print_info(f"Loading node file from {node}...")
        node_data = load_node_file(node)
        n_nodes = len(node_data)
        print_success(f"Loaded {n_nodes} nodes")

        print_info(f"Loading edge file from {edge}...")
        edge_data = load_edge_file(edge)
        print_success(f"Loaded {edge_data.shape} edge matrix")

        # Pre-checks: surface clear errors before delegating to the mapper.
        if edge_data.ndim != 2 or edge_data.shape[0] != edge_data.shape[1]:
            raise ValueError(
                f"Edge matrix is not square: shape={edge_data.shape}."
            )
        if edge_data.shape[0] != n_nodes:
            raise ValueError(
                f"Edge matrix has {edge_data.shape[0]} rows but .node file has "
                f"{n_nodes} entries - they must match."
            )

        print_info(f"Loading coords reference from {coords}...")
        with open(coords, 'r') as f:
            first_line = f.readline()
        delimiter = '\t' if '\t' in first_line else ','
        coords_df = pd.read_csv(coords, sep=delimiter)

        if 'roi_name' not in coords_df.columns:
            raise ValueError(
                f"Coords CSV is missing the required 'roi_name' column. "
                f"Found columns: {list(coords_df.columns)}"
            )
        if len(coords_df) < n_nodes:
            raise ValueError(
                f"Coords CSV has {len(coords_df)} ROIs but .node file has "
                f"{n_nodes}. --coords must reference at least as many ROIs as "
                f"the node file."
            )
        print_success(f"Loaded {len(coords_df)} reference ROIs")

        print_info("Embedding edge matrix into reference atlas...")
        matrix, roi_names, node_indices = node_edge_to_roi_matrix(node, edge, coords)
        print_success(
            f"Mapped {len(node_indices)} nodes into {matrix.shape} matrix"
        )

        # Save
        output_path = Path(output)
        if output_path.suffix.lower() == '.csv':
            pd.DataFrame(matrix).to_csv(output, index=False, header=False)
        else:
            import numpy as np
            np.save(output, matrix)

        print_success(f"Saved {matrix.shape} matrix to {output}")

    except Exception as e:
        print_error(f"Error converting files: {e}")
        raise click.Abort()


@utils.command("info")
@click.option("--matrix", "-m", type=click.Path(exists=True),
              help="Connectivity matrix file.")
@click.option("--volume", "-v", type=click.Path(exists=True),
              help="Label/atlas NIfTI volume. Reports ROI count and whether the "
                   "labels are integer-valued (a non-integer atlas yields NaN COGs).")
def matrix_info(matrix, volume):
    """
    Display information about a connectivity matrix and/or a label volume.

    \b
    Examples:
      hlplot utils info --matrix connectivity.npy
      hlplot utils info --volume atlas.nii.gz
    """
    try:
        import numpy as np

        if not matrix and not volume:
            print_error("Provide --matrix and/or --volume.")
            raise click.Abort()

        if volume:
            from HarrisLabPlotting import inspect_label_volume
            print_info(f"Inspecting label volume: {volume}...")
            vinfo = inspect_label_volume(volume)
            vstats = {
                "Shape": " x ".join(str(s) for s in vinfo["shape"]),
                "Data type": vinfo["dtype"],
                "ROI labels": vinfo["n_labels"],
                "Label range": f"{vinfo['label_min']} to {vinfo['label_max']}",
                "Contiguous": vinfo["contiguous"],
                "Integer-labeled": vinfo["is_integer_labeled"],
                "Max deviation from int": f"{vinfo['max_label_deviation']:.3g}",
            }
            console.print()
            console.print(create_stats_table(vstats, title="Label Volume Information"))
            if not vinfo["is_integer_labeled"]:
                kind = ("stored as floats (not bit-exact integers)"
                        if vinfo["near_integer"] else "not integer-valued")
                print_warning(
                    f"Labels are {kind}; an exact label match would find zero voxels "
                    f"and produce NaN COGs. `hlplot coords generate` handles this "
                    f"automatically (--round-labels, on by default), or pre-fix with "
                    f"`hlplot utils clean-labels`."
                )

        if not matrix:
            return

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

        # ----- direction report ---------------------------------------
        # A bare "Symmetric: False" is not enough to act on: it does not say
        # how asymmetric, how many connections are reciprocal, or which index
        # is the source. All three decide whether the matrix needs transposing.
        from HarrisLabPlotting.directed import (
            check_matrix_symmetry, format_symmetry_report,
        )
        if mat.shape[0] == mat.shape[1]:
            rep = check_matrix_symmetry(mat)
            console.print()
            console.print(format_symmetry_report(rep))

            # Stochastic detection: a transition matrix's row/column sums say
            # which index is the source. Reported ONLY -- never transposed
            # automatically, because a silent flip is the failure this is here
            # to prevent.
            off = mat.astype(float)
            nonneg = bool(np.all(off >= 0))
            row_s, col_s = off.sum(axis=1), off.sum(axis=0)
            row_stoch = nonneg and bool(np.allclose(row_s, 1.0, atol=1e-6))
            col_stoch = nonneg and bool(np.allclose(col_s, 1.0, atol=1e-6))
            if row_stoch or col_stoch:
                console.print()
                if row_stoch and not col_stoch:
                    print_info(
                        "Rows sum to 1: this is a ROW-stochastic transition "
                        "matrix, so row = current state = SOURCE. That already "
                        "matches hlplot's convention -- do NOT transpose."
                    )
                elif col_stoch and not row_stoch:
                    print_warning(
                        "Columns sum to 1: this is a COLUMN-stochastic "
                        "transition matrix, so column = SOURCE. hlplot expects "
                        "row = source -- pass --matrix-orientation col-to-row, "
                        "or run `hlplot utils transpose`."
                    )
                else:
                    print_info(
                        "Both rows and columns sum to 1 (doubly stochastic); "
                        "the sums cannot tell you which index is the source."
                    )
            elif not rep["is_symmetric"]:
                console.print()
                print_info(
                    "Asymmetric and not stochastic. hlplot reads M[i,j] as "
                    "i -> j (row = source). If it came from SPM DCM, that is "
                    "the opposite convention: use --matrix-orientation "
                    "col-to-row."
                )

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


@utils.command("clean-labels")
@click.option("--volume", "-v", required=True, type=click.Path(exists=True),
              help="Input label/atlas NIfTI whose labels may be stored as floats.")
@click.option("--output", "-o", required=True, type=click.Path(),
              help="Output NIfTI path (.nii or .nii.gz).")
@click.option("--dtype", default="int16",
              help="Integer dtype for the cleaned volume (default: int16).")
def clean_labels_cmd(volume, output, dtype):
    """
    Round a float-labeled atlas to clean integer labels.

    Some atlases store integer ROI labels as floats with tiny rounding error
    (e.g. 0.9999999 for label 1). The exact label match in `coords generate`
    then finds ZERO voxels and every COG comes out NaN. This rounds the volume
    to clean integer labels so coordinate extraction works.

    \b
    Examples:
      hlplot utils clean-labels --volume atlas_float.nii --output atlas_int.nii.gz
    """
    try:
        from HarrisLabPlotting import inspect_label_volume, clean_label_volume

        info = inspect_label_volume(volume)
        if info["is_integer_labeled"]:
            print_info("Labels are already bit-exact integers; rounding is a no-op (still written).")
        else:
            print_warning(
                f"Labels are not bit-exact integers (max deviation "
                f"{info['max_label_deviation']:.3g}); rounding to nearest integer."
            )
        clean_label_volume(volume, output_path=output, dtype=dtype)
        print_success(f"Wrote cleaned label volume ({info['n_labels']} ROIs) to {output}")

    except Exception as e:
        print_error(f"Error cleaning labels: {e}")
        raise click.Abort()


@utils.command("check-alignment")
@click.option("--coords", "-c", required=True, type=click.Path(exists=True),
              help="ROI coordinates CSV (cog_x/cog_y/cog_z), e.g. from `coords generate`.")
@click.option("--mesh", "-m", required=True, type=click.Path(exists=True),
              help="Brain mesh file the COGs should land on.")
@click.option("--volume", "-v", type=click.Path(exists=True),
              help="Optional source label atlas, to also check it shares the mesh's space.")
@click.option("--matrix", "-x", type=click.Path(exists=True),
              help="Optional connectivity matrix, to check its size matches the coords.")
def check_alignment_cmd(coords, mesh, volume, matrix):
    """
    Check that an atlas/coords/mesh trio are mutually consistent before plotting.

    Runs a battery of sanity checks and prints a PASS / WARN / FAIL report:

    \b
      * are the ROI COGs inside the brain mesh? (convex-hull test)
      * any NaN COGs, or COGs collapsed onto the midline?
      * (with --volume) are the atlas and mesh in the same template space?
      * (with --matrix) does the matrix size match the coords ROI count?

    \b
    Examples:
      hlplot utils check-alignment --coords rois_comma.csv --mesh brain.obj
      hlplot utils check-alignment -c rois_comma.csv -m brain.obj -v atlas.nii.gz -x conn.csv
    """
    try:
        from HarrisLabPlotting import check_coords_in_mesh, compare_volume_mesh_space

        order = {"PASS": 0, "WARN": 1, "FAIL": 2}
        overall = "PASS"

        def _merge(v):
            return v if order[v] > order[overall] else overall

        # --- COGs inside the mesh ---
        print_info("Checking ROI COGs against the mesh...")
        r = check_coords_in_mesh(coords, mesh)
        nv = r["nearest_vertex_dist_mm"]
        cstats = {
            "ROIs": r["n_rois"],
            "NaN COGs": r["n_nan"],
            "Inside mesh hull": f"{r['n_inside']} / {r['n_inside'] + r['n_outside']}",
            "COG bbox within mesh": r["coords_bbox_within_mesh"],
            "On midline (|x|<2mm)": f"{r['n_on_midline']} ({r['midline_fraction']:.0%})",
            "Nearest-vertex dist (mm)": (
                f"max {nv['max']:.1f}, mean {nv['mean']:.1f}"
                if nv["max"] is not None else "n/a"),
            "Verdict": r["verdict"],
        }
        console.print()
        console.print(create_stats_table(cstats, title="COGs vs Mesh"))
        for msg in r["messages"]:
            console.print(f"  - {msg}")
        overall = _merge(r["verdict"])

        # --- atlas vs mesh template space ---
        if volume:
            print_info("Checking atlas vs mesh template space...")
            s = compare_volume_mesh_space(volume, mesh)
            sstats = {
                "Bbox overlap": f"{s['bbox_overlap_fraction']:.0%}",
                "Centroid offset (mm)": (f"{s['centroid_offset_mm']:.1f}"
                                         if s['centroid_offset_mm'] is not None else "n/a"),
                "Same space": s["same_space"],
                "Verdict": s["verdict"],
            }
            console.print()
            console.print(create_stats_table(sstats, title="Atlas vs Mesh Space"))
            for msg in s["messages"]:
                console.print(f"  - {msg}")
            overall = _merge(s["verdict"])

        # --- matrix size vs coords ---
        if matrix:
            from HarrisLabPlotting import load_connectivity_input
            mat = load_connectivity_input(matrix)
            n_coords = r["n_rois"]
            console.print()
            if mat.shape[0] == mat.shape[1] == n_coords:
                console.print(f"  [green]OK[/green] Matrix {mat.shape} matches {n_coords} ROIs")
            else:
                console.print(f"  [red]X[/red] Matrix {mat.shape} does not match {n_coords} coords ROIs")
                overall = _merge("FAIL")

        console.print()
        color = {"PASS": "green", "WARN": "yellow", "FAIL": "red"}[overall]
        console.print(f"[bold {color}]Overall: {overall}[/bold {color}]")

    except Exception as e:
        print_error(f"Error during alignment check: {e}")
        raise click.Abort()


@utils.command("transpose")
@click.option("--matrix", "-m", required=True, type=click.Path(exists=True),
              help="Matrix file to transpose (.csv, .npy, .txt, .edge, .mat).")
@click.option("--output", "-o", required=True, type=click.Path(),
              help="Where to write the transposed matrix (.csv or .npy).")
def transpose_matrix(matrix, output):
    """
    Transpose a connectivity matrix, flipping its direction convention.

    \b
    hlplot reads M[i, j] as the connection i -> j (row = SOURCE, column =
    TARGET) -- the numpy / networkx convention. Transpose when your matrix
    uses the opposite convention:

    \b
      SPM DCM              A(i,j) is FROM j TO i  -> transpose
      column-stochastic    columns sum to 1       -> transpose
      row-stochastic       rows sum to 1          -> do NOT transpose
      networkx / BCT       already i -> j         -> do NOT transpose

    \b
    The symmetry verdict is printed before and after so you can see the
    transpose actually changed something. `hlplot utils info --matrix ...`
    reports which convention a matrix appears to use.

    \b
    Examples:
      hlplot utils transpose --matrix DCM_A.csv --output DCM_A_rowcol.csv
      hlplot utils transpose -m probs.npy -o probs_T.npy
    """
    try:
        import numpy as np
        from pathlib import Path as _Path
        from HarrisLabPlotting import load_connectivity_input
        from HarrisLabPlotting.directed import (
            check_matrix_symmetry, format_symmetry_report,
        )

        print_info(f"Loading matrix from {matrix}...")
        mat = load_connectivity_input(matrix)
        if mat.shape[0] != mat.shape[1]:
            print_error(f"Matrix must be square to transpose meaningfully, "
                        f"got {mat.shape[0]} x {mat.shape[1]}")
            raise click.Abort()

        console.print()
        console.print("[bold]before[/bold]")
        console.print(format_symmetry_report(check_matrix_symmetry(mat)))

        out = mat.T.copy()

        console.print()
        console.print("[bold]after[/bold]")
        console.print(format_symmetry_report(check_matrix_symmetry(out)))
        if np.allclose(mat, out):
            print_warning(
                "The matrix is symmetric, so transposing changed nothing. "
                "Direction conventions only matter for asymmetric matrices."
            )

        out_path = _Path(output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        if out_path.suffix.lower() == ".npy":
            np.save(out_path, out)
        else:
            np.savetxt(out_path, out, delimiter=",")
        print_success(f"Wrote transposed {out.shape[0]} x {out.shape[1]} "
                      f"matrix to {out_path}")

    except click.Abort:
        raise
    except Exception as e:
        print_error(f"Error transposing matrix: {e}")
        raise click.Abort()
