"""
Combine folders of BrainNet Viewer .edge and/or .node files.

Mirrors the behavior of the standalone edge-combine and node-combine CLIs but
lives inside HarrisLabPlotting so downstream code can combine files
programmatically or from the `hlplot` CLI.

Alignment contract
------------------
The three helpers below share the same sort semantics so that running
`combine_edge_folder(X)` and `combine_node_folder(X)` on the same folder (with
the same sort_mode) produces `total.edge` (block-diagonal NxN) and
`total.node` (N concatenated rows) where node row k corresponds to edge block
position k. `combine_node_edge_folder` makes this guarantee explicit by sharing
one stem order between the two writes.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.linalg import block_diag


# ---------- I/O helpers ---------------------------------------------------- #

def _read_edge_matrix(path: Path) -> np.ndarray:
    """Load a square whitespace- or tab-delimited .edge file."""
    rows = [
        [float(tok) for tok in ln.strip().split()]
        for ln in path.read_text().splitlines()
        if ln.strip()
    ]
    arr = np.asarray(rows, float)
    if arr.ndim != 2 or arr.shape[0] != arr.shape[1]:
        raise ValueError(f"{path} is not a square matrix (shape {arr.shape})")
    return arr


def _read_node_df(path: Path) -> pd.DataFrame:
    """Load a BrainNet Viewer .node file into a DataFrame.

    Columns: x, y, z (float), size, color (numeric), label (str).
    Tolerates CRLF line endings by stripping the label column.
    """
    df = pd.read_csv(
        path,
        sep="\t",
        header=None,
        names=["x", "y", "z", "size", "color", "label"],
        dtype={"label": str},
    )
    df["label"] = df["label"].astype(str).str.strip()
    return df


def _edge_dim(path: Path) -> int:
    """Matrix dimension = count of non-blank lines."""
    with path.open() as f:
        return sum(1 for ln in f if ln.strip())


def _node_rows(path: Path) -> int:
    """Row count = count of non-blank lines."""
    with path.open() as f:
        return sum(1 for ln in f if ln.strip())


# ---------- sort helpers --------------------------------------------------- #

_SORT_MODES = {"directory", "alphabetical", "size"}


def _validate_sort_mode(sort_mode: str) -> None:
    if sort_mode not in _SORT_MODES:
        raise ValueError(
            f"sort_mode must be one of {_SORT_MODES}, got {sort_mode!r}"
        )


def _sort_paths(paths: Sequence[Path], size_fn, sort_mode: str) -> List[Path]:
    """Sort paths using the same semantics as edge-combine."""
    _validate_sort_mode(sort_mode)
    if sort_mode == "size":
        # Largest first, tie-break on name for determinism.
        return sorted(paths, key=lambda p: (-size_fn(p), p.name))
    # Both "alphabetical" and "directory" sort by name (matches edge-combine).
    return sorted(paths, key=lambda p: p.name)


def _glob_excluding(folder: Path, pattern: str, skip_names: Iterable[str]) -> List[Path]:
    skip = set(skip_names)
    return [p for p in folder.glob(pattern) if p.name not in skip]


# ---------- writers -------------------------------------------------------- #

def _write_edge_matrix(matrix: np.ndarray, path: Path, precision: int) -> None:
    np.savetxt(path, matrix, fmt=f"%.{precision}f", delimiter="\t")


def _format_node_row(row: pd.Series, precision: int) -> str:
    x = f"{float(row['x']):.{precision}f}"
    y = f"{float(row['y']):.{precision}f}"
    z = f"{float(row['z']):.{precision}f}"
    size = str(int(round(float(row["size"]))))
    color = str(int(round(float(row["color"]))))
    label = str(row["label"]).strip()
    return "\t".join([x, y, z, size, color, label])


def _write_node_frames(frames: Sequence[pd.DataFrame], path: Path, precision: int) -> None:
    lines = []
    for df in frames:
        for _, row in df.iterrows():
            lines.append(_format_node_row(row, precision))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# ---------- public API ----------------------------------------------------- #

def combine_edge_folder(
    folder: str | Path,
    output_name: str = "total.edge",
    precision: int = 5,
    sort_mode: str = "directory",
) -> Path:
    """Assemble `.edge` files in `folder` into a block-diagonal `total.edge`.

    Parameters
    ----------
    folder : str | Path
        Directory containing the input `.edge` files.
    output_name : str
        Output filename written into `folder`. Defaults to ``total.edge``.
    precision : int
        Decimal places for the float formatter. Defaults to ``5``.
    sort_mode : str
        One of ``"directory"``, ``"alphabetical"`` (both sort by filename), or
        ``"size"`` (largest matrix first, tie-break on name).

    Returns
    -------
    Path
        Path of the written output file.
    """
    folder = Path(folder)
    edge_files = _glob_excluding(folder, "*.edge", {output_name})
    if not edge_files:
        raise FileNotFoundError(f"No .edge files found in {folder}")

    edge_files = _sort_paths(edge_files, _edge_dim, sort_mode)
    blocks = [_read_edge_matrix(p) for p in edge_files]
    combined = block_diag(*blocks)

    out_path = folder / output_name
    _write_edge_matrix(combined, out_path, precision)
    return out_path


def combine_node_folder(
    folder: str | Path,
    output_name: str = "total.node",
    precision: int = 5,
    sort_mode: str = "directory",
) -> Path:
    """Concatenate `.node` files in `folder` into a single tab-delimited file.

    Parameters mirror :func:`combine_edge_folder`; ``precision`` applies only
    to x/y/z coordinates (size/color are written as integers).
    """
    folder = Path(folder)
    # Exclude our own output and the edge-combine output, in case a user writes
    # both into the same folder repeatedly.
    node_files = _glob_excluding(folder, "*.node", {output_name, "total.edge"})
    if not node_files:
        raise FileNotFoundError(f"No .node files found in {folder}")

    node_files = _sort_paths(node_files, _node_rows, sort_mode)
    frames = [_read_node_df(p) for p in node_files]

    out_path = folder / output_name
    _write_node_frames(frames, out_path, precision)
    return out_path


def combine_node_edge_folder(
    folder: str | Path,
    edge_output: str = "total.edge",
    node_output: str = "total.node",
    precision: int = 5,
    sort_mode: str = "directory",
) -> Tuple[Path, Path]:
    """Combine paired `.node` and `.edge` files using one shared ordering.

    Enumerates stems once, asserts that every `.node` has a matching `.edge`
    and that the node row count equals the edge matrix dimension for each
    stem, then writes both `total.edge` (block-diagonal) and `total.node`
    (row-concatenated) using the same sort order.

    Returns
    -------
    (edge_path, node_path) : tuple[Path, Path]
    """
    folder = Path(folder)
    skip = {edge_output, node_output}
    edge_files = _glob_excluding(folder, "*.edge", skip)
    node_files = _glob_excluding(folder, "*.node", skip)
    if not edge_files and not node_files:
        raise FileNotFoundError(f"No .edge or .node files found in {folder}")

    edge_by_stem = {p.stem: p for p in edge_files}
    node_by_stem = {p.stem: p for p in node_files}
    stems = sorted(set(edge_by_stem) & set(node_by_stem))
    if not stems:
        raise FileNotFoundError(
            f"No stem with both .node and .edge in {folder}"
        )

    missing_edge = sorted(set(node_by_stem) - set(edge_by_stem))
    missing_node = sorted(set(edge_by_stem) - set(node_by_stem))
    if missing_edge or missing_node:
        parts = []
        if missing_edge:
            parts.append(f"missing .edge for: {missing_edge}")
        if missing_node:
            parts.append(f"missing .node for: {missing_node}")
        raise FileNotFoundError("; ".join(parts))

    # Sort by a stem-ordered Path list (we sort one list, apply to both).
    stem_paths = [folder / f"{s}.edge" for s in stems]  # fake list for sort
    stem_paths = _sort_paths(stem_paths, _edge_dim, sort_mode)
    ordered_stems = [p.stem for p in stem_paths]

    # Per-stem sanity check: node rows match edge dimension.
    blocks = []
    frames = []
    for stem in ordered_stems:
        e_path = edge_by_stem[stem]
        n_path = node_by_stem[stem]
        block = _read_edge_matrix(e_path)
        df = _read_node_df(n_path)
        if block.shape[0] != len(df):
            raise ValueError(
                f"Stem {stem!r}: edge dim {block.shape[0]} "
                f"!= node rows {len(df)}"
            )
        blocks.append(block)
        frames.append(df)

    edge_path = folder / edge_output
    node_path = folder / node_output
    _write_edge_matrix(block_diag(*blocks), edge_path, precision)
    _write_node_frames(frames, node_path, precision)
    return edge_path, node_path


__all__ = [
    "combine_edge_folder",
    "combine_node_folder",
    "combine_node_edge_folder",
]
