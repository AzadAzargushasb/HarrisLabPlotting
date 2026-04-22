"""
Combine folders of BrainNet Viewer .edge / .node files from the CLI.
"""

import click
from pathlib import Path

from ..console import console, print_success, print_error, print_info


def _resolve_sort_mode(alphabetical: bool, size: bool) -> str:
    if alphabetical and size:
        raise click.UsageError("Use either -a/--alphabetical or -s/--size, not both.")
    if alphabetical:
        return "alphabetical"
    if size:
        return "size"
    return "directory"


@click.group()
def combine():
    """
    Combine folders of `.edge` and/or `.node` files.

    Subcommands mirror the standalone edge-combine and node-combine CLIs.

    \b
    Examples:
      hlplot combine edges /path/to/folder
      hlplot combine nodes /path/to/folder
      hlplot combine both  /path/to/folder
    """
    pass


@combine.command("edges")
@click.argument("folder", type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.option("-o", "--output", default="total.edge",
              help="Output filename (default: total.edge).")
@click.option("-p", "--precision", type=int, default=5,
              help="Decimal places for float formatter (default: 5).")
@click.option("-a", "--alphabetical", is_flag=True,
              help="Sort files alphabetically.")
@click.option("-s", "--size", is_flag=True,
              help="Sort files by matrix dimension (largest first).")
def edges_cmd(folder, output, precision, alphabetical, size):
    """Combine `.edge` files into a block-diagonal matrix."""
    try:
        from HarrisLabPlotting import combine_edge_folder
        sort_mode = _resolve_sort_mode(alphabetical, size)
        print_info(f"Combining .edge files from {folder} ({sort_mode})...")
        out = combine_edge_folder(
            folder, output_name=output, precision=precision, sort_mode=sort_mode
        )
        print_success(f"Wrote {out}")
    except Exception as e:
        print_error(f"Error combining edges: {e}")
        raise click.Abort()


@combine.command("nodes")
@click.argument("folder", type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.option("-o", "--output", default="total.node",
              help="Output filename (default: total.node).")
@click.option("-p", "--precision", type=int, default=5,
              help="Decimal places for x/y/z (default: 5). size/color kept as integers.")
@click.option("-a", "--alphabetical", is_flag=True,
              help="Sort files alphabetically.")
@click.option("-s", "--size", is_flag=True,
              help="Sort files by row count (most rows first).")
def nodes_cmd(folder, output, precision, alphabetical, size):
    """Concatenate `.node` files in the same order edge-combine uses."""
    try:
        from HarrisLabPlotting import combine_node_folder
        sort_mode = _resolve_sort_mode(alphabetical, size)
        print_info(f"Combining .node files from {folder} ({sort_mode})...")
        out = combine_node_folder(
            folder, output_name=output, precision=precision, sort_mode=sort_mode
        )
        print_success(f"Wrote {out}")
    except Exception as e:
        print_error(f"Error combining nodes: {e}")
        raise click.Abort()


@combine.command("both")
@click.argument("folder", type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.option("--edge-output", default="total.edge",
              help="Output filename for edges (default: total.edge).")
@click.option("--node-output", default="total.node",
              help="Output filename for nodes (default: total.node).")
@click.option("-p", "--precision", type=int, default=5,
              help="Decimal places for floats (default: 5). Integer columns unchanged.")
@click.option("-a", "--alphabetical", is_flag=True,
              help="Sort stems alphabetically.")
@click.option("-s", "--size", is_flag=True,
              help="Sort stems by matrix dimension (largest first).")
def both_cmd(folder, edge_output, node_output, precision, alphabetical, size):
    """Combine paired .node/.edge files with a shared ordering and per-stem dim check."""
    try:
        from HarrisLabPlotting import combine_node_edge_folder
        sort_mode = _resolve_sort_mode(alphabetical, size)
        print_info(f"Combining paired .node / .edge from {folder} ({sort_mode})...")
        edge_path, node_path = combine_node_edge_folder(
            folder,
            edge_output=edge_output,
            node_output=node_output,
            precision=precision,
            sort_mode=sort_mode,
        )
        print_success(f"Wrote {edge_path}")
        print_success(f"Wrote {node_path}")
    except Exception as e:
        print_error(f"Error combining paired files: {e}")
        raise click.Abort()
