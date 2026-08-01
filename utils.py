"""
Utility Functions
=================
Helper functions for brain visualization calculations.
"""

import numpy as np
import pandas as pd
import json
import re
from pathlib import Path
from typing import Tuple, Union, List, Optional, Dict
from scipy.io import loadmat


class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder for numpy types."""

    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.bool_):
            return bool(obj)
        return super(NumpyEncoder, self).default(obj)


def classify_node_role(z_score: float, pc: float) -> Tuple[str, str]:
    """
    Classify node role using the Guimerà & Amaral cartographic two-cut.

    Implements the seven-region role classification of Guimerà & Nunes
    Amaral, *Functional cartography of complex metabolic networks*,
    **Nature** 433, 895-900 (2005), https://doi.org/10.1038/nature03288.

    The non-hub vs. hub split is at within-module ``Z = 2.5``. Each
    half is then sub-divided by the participation coefficient ``P``:

    - Non-hubs (``Z < 2.5``):
        - **R1 Ultra-peripheral**: ``P <= 0.05``
        - **R2 Peripheral**: ``0.05 < P <= 0.62``
        - **R3 Non-hub connector**: ``0.62 < P <= 0.80``
        - **R4 Non-hub kinless**: ``P > 0.80``

    - Hubs (``Z >= 2.5``):
        - **R5 Provincial hub**: ``P <= 0.30``
        - **R6 Connector hub**: ``0.30 < P <= 0.75``
        - **R7 Kinless hub**: ``P > 0.75``

    The returned border color is chosen to be visually distinct from
    the default module fill palette (red / green / blue / orange /
    purple / brown), since this color is rendered as a ring around a
    module-colored fill in the modularity plot.

    Parameters
    ----------
    z_score : float
        Within-module Z-score (also called z, within-module degree
        z-score). Hub status is determined by ``z_score >= 2.5``.
    pc : float
        Participation coefficient (``0 <= P <= 1``). Cuts the non-hub
        and hub halves into the role sub-regions above.

    Returns
    -------
    tuple of (str, str)
        ``(role_name, hex_color)`` for the node. ``role_name`` is one
        of: ``"Ultra-peripheral"``, ``"Peripheral"``, ``"Non-hub
        connector"``, ``"Non-hub kinless"``, ``"Provincial hub"``,
        ``"Connector hub"``, ``"Kinless hub"``, or ``"Unclassified"``
        (defensive fallback for non-finite inputs).

    References
    ----------
    Guimerà R, Nunes Amaral LA. Functional cartography of complex
    metabolic networks. *Nature* 433, 895-900 (2005).
    https://doi.org/10.1038/nature03288
    """
    # Defensive: NaN / inf go to "Unclassified" instead of an arbitrary
    # branch (the comparisons below would otherwise all be False).
    if not (np.isfinite(z_score) and np.isfinite(pc)):
        return "Unclassified", "#808080"

    if z_score < 2.5:
        # Non-hub regions (R1 - R4)
        if pc <= 0.05:
            return "Ultra-peripheral", "#FFFFFF"     # white -- barely-there border
        elif pc <= 0.62:
            return "Peripheral", "#CCCCCC"           # light gray -- quiet, non-hub
        elif pc <= 0.80:
            return "Non-hub connector", "#00CED1"    # turquoise (R3)
        else:
            return "Non-hub kinless", "#FF1493"      # deep pink (R4)
    else:
        # Hub regions (R5 - R7)
        if pc <= 0.30:
            return "Provincial hub", "#FFFF00"       # bright yellow (R5)
        elif pc <= 0.75:
            return "Connector hub", "#000000"        # black (R6)
        else:
            return "Kinless hub", "#FF00FF"          # magenta (R7)


_HEMI_SUFFIX_RE = re.compile(r"_(L|R|left|right|lh|rh)$", re.IGNORECASE)
_LEFT_TOKENS = {"l", "left", "lh"}


def short_roi_name(name: str, keep_hemisphere: bool = False) -> str:
    """Shorten an ROI name by dropping or abbreviating its hemisphere suffix.

    Useful when labeling a plot with more than a handful of nodes: a suffix like
    ``_left`` roughly doubles the label length, so shortening it cuts label
    overlap a lot.

    Only a *trailing* hemisphere token is matched, so names that merely contain
    "left"/"right"/"l"/"r" are untouched.

    ``keep_hemisphere=True`` abbreviates the suffix to ``_L`` / ``_R`` instead of
    removing it. Prefer that whenever both hemispheres are visible in the same
    view (e.g. superior/inferior): dropping the suffix entirely makes the left and
    right node of a pair carry the *same* label, which is ambiguous.

    Examples
    --------
    >>> short_roi_name('V1_L')
    'V1'
    >>> short_roi_name('AUD_left')
    'AUD'
    >>> short_roi_name('IFG.cv_left')
    'IFG.cv'
    >>> short_roi_name('Thalamus_P_right')
    'Thalamus_P'
    >>> short_roi_name('Cerebellum')          # nothing to shorten
    'Cerebellum'
    >>> short_roi_name('AUD_left', keep_hemisphere=True)
    'AUD_L'
    >>> short_roi_name('Thalamus_P_right', keep_hemisphere=True)
    'Thalamus_P_R'
    >>> short_roi_name('V1_L', keep_hemisphere=True)
    'V1_L'
    >>> short_roi_name('Cerebellum', keep_hemisphere=True)
    'Cerebellum'

    Parameters
    ----------
    name : str
        The full ROI name (e.g. from a coords file's ``roi_name`` column).
    keep_hemisphere : bool
        When ``True``, abbreviate the suffix to ``_L``/``_R`` rather than drop it,
        so left/right pairs stay distinguishable. Default ``False``.

    Returns
    -------
    str
        The shortened name. Names without a trailing hemisphere token are
        returned unchanged either way.
    """
    text = str(name)
    match = _HEMI_SUFFIX_RE.search(text)
    base = _HEMI_SUFFIX_RE.sub("", text)
    if match is None or not keep_hemisphere:
        return base
    side = "L" if match.group(1).lower() in _LEFT_TOKENS else "R"
    return f"{base}_{side}"


def calculate_node_size(pc: float, z_score: float, mode: str = 'both',
                        base_size: int = 6, max_multiplier: float = 5.0) -> float:
    """
    Calculate dynamic node size with controlled scaling for better visibility.

    Parameters
    ----------
    pc : float
        Participation coefficient
    z_score : float
        Within-module z-score
    mode : str
        Sizing mode: 'pc', 'zscore', or 'both'
    base_size : int
        Base node size
    max_multiplier : float
        Maximum size multiplier

    Returns
    -------
    float
        Calculated node size
    """
    if mode == 'pc':
        multiplier = 1 + (pc ** 0.5) * (max_multiplier - 1)
    elif mode == 'zscore':
        normalized_z = min(abs(z_score) / 2.0, 1.0)
        multiplier = 1 + (normalized_z ** 0.6) * (max_multiplier - 1)
    elif mode == 'both':
        pc_component = (pc ** 0.5) * (max_multiplier - 1) * 0.5
        z_component = (min(abs(z_score) / 2.0, 1.0) ** 0.6) * (max_multiplier - 1) * 0.5
        multiplier = 1 + pc_component + z_component
    else:
        multiplier = 1

    final_size = base_size * multiplier
    return max(base_size * 0.7, final_size)


def calculate_edge_width(weight: float, all_weights: np.ndarray,
                         min_width: float = 0.5, max_width: float = 6.0) -> float:
    """
    Calculate edge width based on coherence strength.

    Parameters
    ----------
    weight : float
        Edge weight value
    all_weights : np.ndarray
        Array of all edge weights for normalization
    min_width : float
        Minimum edge width
    max_width : float
        Maximum edge width

    Returns
    -------
    float
        Calculated edge width
    """
    weight_abs = abs(weight)
    min_weight = np.min(np.abs(all_weights[all_weights != 0]))
    max_weight = np.max(np.abs(all_weights))

    if max_weight > min_weight:
        normalized = (weight_abs - min_weight) / (max_weight - min_weight)
    else:
        normalized = 0.5

    normalized = normalized ** 0.7
    width = min_width + normalized * (max_width - min_width)
    return width


def filter_edges_by_module(connectivity_matrix, module_assignments, module_id, mode='all'):
    """
    Filter edges based on module membership.

    Parameters
    ----------
    connectivity_matrix : np.ndarray
        Connectivity matrix
    module_assignments : np.ndarray
        Module assignment for each node
    module_id : int
        Module ID to filter for
    mode : str
        'intra' for within-module, 'inter' for between-module, 'all' for no filtering

    Returns
    -------
    np.ndarray
        Filtered connectivity matrix
    """
    filtered = connectivity_matrix.copy()
    module_mask = (module_assignments == module_id)

    if mode == 'intra':
        for i in range(len(module_assignments)):
            for j in range(len(module_assignments)):
                if not (module_mask[i] and module_mask[j]):
                    filtered[i, j] = 0
    elif mode == 'inter':
        for i in range(len(module_assignments)):
            for j in range(len(module_assignments)):
                if not ((module_mask[i] and not module_mask[j]) or
                        (not module_mask[i] and module_mask[j])):
                    filtered[i, j] = 0

    return filtered


def threshold_matrix_top_n(matrix, n_edges):
    """
    Keep only top N edges in the matrix.

    Parameters
    ----------
    matrix : np.ndarray
        Input connectivity matrix
    n_edges : int
        Number of top edges to keep

    Returns
    -------
    np.ndarray
        Thresholded matrix
    """
    matrix_copy = matrix.copy()

    upper_tri = np.triu(matrix_copy, k=1)
    flat_values = upper_tri[upper_tri != 0]

    if len(flat_values) > n_edges:
        threshold_value = np.sort(np.abs(flat_values))[-n_edges]
        matrix_copy[np.abs(matrix_copy) < threshold_value] = 0

    return matrix_copy


def filter_matrix_by_sign(matrix, keep_sign='both'):
    """
    Zero out positive or negative entries in a connectivity matrix.

    This is useful when you want to plot or analyze only the positive
    correlations or only the negative (anti-)correlations of a matrix.
    Entries with the unwanted sign are set to 0 (i.e. removed). Diagonal
    elements and existing zero entries are not changed by this operation.

    Parameters
    ----------
    matrix : np.ndarray
        Input connectivity matrix (any shape, typically NxN).
    keep_sign : str, optional
        Which sign to keep. One of:

        - ``'both'`` (default): no filtering, return a copy of the matrix
          unchanged.
        - ``'positive'``: keep only entries where ``value > 0``. All entries
          with ``value < 0`` are set to 0.
        - ``'negative'``: keep only entries where ``value < 0``. All entries
          with ``value > 0`` are set to 0. The remaining negative values
          keep their original sign (they are NOT made positive).

    Returns
    -------
    np.ndarray
        A new matrix of the same shape as ``matrix`` with the unwanted
        sign zeroed out. The original matrix is not modified.

    Raises
    ------
    ValueError
        If ``keep_sign`` is not one of ``'both'``, ``'positive'``,
        ``'negative'``.

    Examples
    --------
    >>> import numpy as np
    >>> m = np.array([[0,  0.5, -0.3],
    ...               [0.5, 0,   0.2],
    ...               [-0.3, 0.2, 0]])
    >>> filter_matrix_by_sign(m, 'positive')
    array([[0. , 0.5, 0. ],
           [0.5, 0. , 0.2],
           [0. , 0.2, 0. ]])
    >>> filter_matrix_by_sign(m, 'negative')
    array([[ 0. ,  0. , -0.3],
           [ 0. ,  0. ,  0. ],
           [-0.3,  0. ,  0. ]])
    """
    if keep_sign not in ('both', 'positive', 'negative'):
        raise ValueError(
            f"keep_sign must be one of 'both', 'positive', 'negative'; "
            f"got {keep_sign!r}"
        )

    result = matrix.copy()
    if keep_sign == 'positive':
        result[result < 0] = 0
    elif keep_sign == 'negative':
        result[result > 0] = 0
    # 'both' -> no change
    return result


def load_node_file(node_file_path: str) -> pd.DataFrame:
    """
    Load a BrainNet Viewer node file.

    Node file format (tab-separated):
    X Y Z size color roi_name

    Parameters
    ----------
    node_file_path : str
        Path to the .node file

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: x, y, z, size, color, roi_name
    """
    df = pd.read_csv(
        node_file_path,
        sep='\t',
        header=None,
        names=['x', 'y', 'z', 'size', 'color', 'roi_name']
    )
    return df


def load_edge_file(edge_file_path: str) -> np.ndarray:
    """
    Load a BrainNet Viewer edge file (connectivity matrix).

    Edge file format: tab-separated matrix of connectivity values.

    Parameters
    ----------
    edge_file_path : str
        Path to the .edge file

    Returns
    -------
    np.ndarray
        Connectivity matrix (n_nodes x n_nodes)
    """
    matrix = np.loadtxt(edge_file_path, delimiter='\t')
    return matrix


def node_edge_to_roi_matrix(
    node_file: str,
    edge_file: str,
    roi_reference: Union[str, List[str], pd.DataFrame],
    roi_name_column: str = 'roi_name'
) -> Tuple[np.ndarray, List[str], List[int]]:
    """
    Convert node/edge files to a full ROI x ROI connectivity matrix.

    This function takes BrainNet Viewer format node and edge files and maps
    them to a full connectivity matrix with dimensions matching the ROI
    reference (e.g., from roi_coordinates.py output).

    Parameters
    ----------
    node_file : str
        Path to the .node file containing ROI coordinates and names
    edge_file : str
        Path to the .edge file containing the connectivity matrix
    roi_reference : str, list, or DataFrame
        Either:
        - Path to a CSV file with ROI names (e.g., from roi_coordinates.py)
        - A list of ROI names
        - A DataFrame with ROI names in roi_name_column
    roi_name_column : str, optional
        Column name containing ROI names if roi_reference is a CSV path or DataFrame.
        Default is 'roi_name'.

    Returns
    -------
    tuple
        - full_matrix : np.ndarray
            Full ROI x ROI connectivity matrix with zeros for unmapped ROIs
        - roi_names : list
            List of all ROI names from the reference
        - node_indices : list
            Indices of the node ROIs in the full ROI list

    Raises
    ------
    ValueError
        If any ROI name in the node file cannot be found in the ROI reference
    FileNotFoundError
        If node_file, edge_file, or roi_reference (if path) doesn't exist

    Examples
    --------
    >>> # Using a CSV file from roi_coordinates.py
    >>> matrix, rois, indices = node_edge_to_roi_matrix(
    ...     'data/subset.node',
    ...     'data/subset.edge',
    ...     'atlas_114_mapped.csv'
    ... )
    >>> matrix.shape
    (114, 114)

    >>> # Using a list of ROI names
    >>> roi_list = ['ROI_A', 'ROI_B', 'ROI_C', ...]
    >>> matrix, rois, indices = node_edge_to_roi_matrix(
    ...     'data/subset.node',
    ...     'data/subset.edge',
    ...     roi_list
    ... )
    """
    # Load node file
    node_df = load_node_file(node_file)
    node_roi_names = node_df['roi_name'].tolist()

    # Load edge file
    edge_matrix = load_edge_file(edge_file)

    # Validate edge matrix dimensions match node count
    n_nodes = len(node_roi_names)
    if edge_matrix.shape[0] != n_nodes or edge_matrix.shape[1] != n_nodes:
        raise ValueError(
            f"Edge matrix dimensions {edge_matrix.shape} do not match "
            f"number of nodes ({n_nodes})"
        )

    # Get full ROI list from reference
    if isinstance(roi_reference, str):
        # It's a file path - load it
        # Auto-detect delimiter
        with open(roi_reference, 'r') as f:
            first_line = f.readline()
        delimiter = '\t' if '\t' in first_line else ','
        roi_df = pd.read_csv(roi_reference, sep=delimiter)

        if roi_name_column not in roi_df.columns:
            raise ValueError(
                f"Column '{roi_name_column}' not found in ROI reference file. "
                f"Available columns: {list(roi_df.columns)}"
            )
        full_roi_names = roi_df[roi_name_column].tolist()

    elif isinstance(roi_reference, pd.DataFrame):
        if roi_name_column not in roi_reference.columns:
            raise ValueError(
                f"Column '{roi_name_column}' not found in DataFrame. "
                f"Available columns: {list(roi_reference.columns)}"
            )
        full_roi_names = roi_reference[roi_name_column].tolist()

    elif isinstance(roi_reference, list):
        full_roi_names = roi_reference

    else:
        raise TypeError(
            f"roi_reference must be a file path (str), list, or DataFrame. "
            f"Got {type(roi_reference)}"
        )

    # Create mapping from node ROI names to indices in full ROI list
    n_full_rois = len(full_roi_names)
    roi_name_to_index = {name: i for i, name in enumerate(full_roi_names)}

    # Validate all node ROIs exist in the reference and get their indices
    node_indices = []
    missing_rois = []

    for roi_name in node_roi_names:
        if roi_name in roi_name_to_index:
            node_indices.append(roi_name_to_index[roi_name])
        else:
            missing_rois.append(roi_name)

    if missing_rois:
        raise ValueError(
            f"The following ROI names from the node file were not found in the "
            f"ROI reference:\n{missing_rois}\n\n"
            f"Available ROI names in reference (first 20): "
            f"{full_roi_names[:20]}..."
        )

    # Create full matrix and fill in values
    full_matrix = np.zeros((n_full_rois, n_full_rois), dtype=edge_matrix.dtype)

    for i, idx_i in enumerate(node_indices):
        for j, idx_j in enumerate(node_indices):
            full_matrix[idx_i, idx_j] = edge_matrix[i, j]

    return full_matrix, full_roi_names, node_indices


def convert_node_size_input(
    node_size_input: Union[int, float, np.ndarray, pd.Series, pd.DataFrame, List, Dict, str],
    n_nodes: int,
    default_size: float = 8.0
) -> np.ndarray:
    """
    Convert various node size input formats to a numpy array.

    Parameters
    ----------
    node_size_input : int, float, np.ndarray, pd.Series, pd.DataFrame, list, dict, or str
        Node size specification. Can be:
        - Scalar (int/float): All nodes get the same size
        - numpy array: Direct array of sizes
        - pandas Series: Sizes indexed by position
        - pandas DataFrame: First numeric column used as sizes
        - list: Converted to numpy array
        - dict: Keys are node indices, values are sizes
        - str: Path to file (.csv, .txt, .npy, .mat)
    n_nodes : int
        Expected number of nodes (for validation)
    default_size : float
        Default size to use for missing values

    Returns
    -------
    np.ndarray
        Array of node sizes with length n_nodes

    Raises
    ------
    ValueError
        If input cannot be converted or doesn't match expected length
    """
    # Scalar input - all nodes same size
    if isinstance(node_size_input, (int, float)):
        return np.full(n_nodes, float(node_size_input))

    # Already a numpy array
    if isinstance(node_size_input, np.ndarray):
        arr = node_size_input.flatten()
        if len(arr) != n_nodes:
            raise ValueError(
                f"Node size array length ({len(arr)}) does not match "
                f"number of nodes ({n_nodes})"
            )
        return arr.astype(float)

    # Pandas Series
    if isinstance(node_size_input, pd.Series):
        arr = node_size_input.values.flatten()
        if len(arr) != n_nodes:
            raise ValueError(
                f"Node size Series length ({len(arr)}) does not match "
                f"number of nodes ({n_nodes})"
            )
        return arr.astype(float)

    # Pandas DataFrame - use first numeric column
    if isinstance(node_size_input, pd.DataFrame):
        numeric_cols = node_size_input.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) == 0:
            raise ValueError("DataFrame has no numeric columns for node sizes")
        arr = node_size_input[numeric_cols[0]].values.flatten()
        if len(arr) != n_nodes:
            raise ValueError(
                f"Node size DataFrame length ({len(arr)}) does not match "
                f"number of nodes ({n_nodes})"
            )
        return arr.astype(float)

    # List
    if isinstance(node_size_input, list):
        arr = np.array(node_size_input, dtype=float).flatten()
        if len(arr) != n_nodes:
            raise ValueError(
                f"Node size list length ({len(arr)}) does not match "
                f"number of nodes ({n_nodes})"
            )
        return arr

    # Dictionary - keys are indices, values are sizes
    if isinstance(node_size_input, dict):
        arr = np.full(n_nodes, default_size)
        for idx, size in node_size_input.items():
            if 0 <= idx < n_nodes:
                arr[idx] = float(size)
        return arr

    # String - file path
    if isinstance(node_size_input, str):
        path = Path(node_size_input)
        if not path.exists():
            raise FileNotFoundError(f"Node size file not found: {node_size_input}")

        suffix = path.suffix.lower()

        if suffix == '.npy':
            arr = np.load(node_size_input).flatten()
        elif suffix == '.mat':
            mat_data = loadmat(node_size_input)
            # Get first non-metadata key
            data_keys = [k for k in mat_data.keys() if not k.startswith('__')]
            if not data_keys:
                raise ValueError(f"No data found in .mat file: {node_size_input}")
            arr = mat_data[data_keys[0]].flatten()
        elif suffix in ['.csv', '.txt']:
            # Try to load as simple array first
            try:
                arr = np.loadtxt(node_size_input, delimiter=',' if suffix == '.csv' else None).flatten()
            except ValueError:
                # Try as DataFrame with header
                df = pd.read_csv(node_size_input)
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                if len(numeric_cols) == 0:
                    raise ValueError(f"No numeric columns in file: {node_size_input}")
                arr = df[numeric_cols[0]].values.flatten()
        else:
            raise ValueError(f"Unsupported file format for node sizes: {suffix}")

        if len(arr) != n_nodes:
            raise ValueError(
                f"Node size file length ({len(arr)}) does not match "
                f"number of nodes ({n_nodes})"
            )
        return arr.astype(float)

    raise TypeError(
        f"Unsupported node_size type: {type(node_size_input)}. "
        f"Expected int, float, np.ndarray, pd.Series, pd.DataFrame, list, dict, or file path."
    )


def load_connectivity_input(
    connectivity_input: Union[np.ndarray, str, pd.DataFrame],
    n_expected_nodes: Optional[int] = None
) -> np.ndarray:
    """
    Load connectivity matrix from various input formats.

    Parameters
    ----------
    connectivity_input : np.ndarray, str, or pd.DataFrame
        Connectivity matrix or path to file. Supports:
        - numpy array: Used directly
        - str: Path to file (.npy, .csv, .txt, .mat, .edge)
        - pd.DataFrame: Converted to numpy array
    n_expected_nodes : int, optional
        Expected number of nodes for validation

    Returns
    -------
    np.ndarray
        Connectivity matrix

    Raises
    ------
    ValueError
        If matrix is not square or doesn't match expected size
    """
    matrix = None

    # Already a numpy array
    if isinstance(connectivity_input, np.ndarray):
        matrix = connectivity_input

    # Pandas DataFrame
    elif isinstance(connectivity_input, pd.DataFrame):
        matrix = connectivity_input.values

    # String - file path
    elif isinstance(connectivity_input, str):
        path = Path(connectivity_input)
        if not path.exists():
            raise FileNotFoundError(f"Connectivity file not found: {connectivity_input}")

        suffix = path.suffix.lower()

        if suffix == '.npy':
            matrix = np.load(connectivity_input)
        elif suffix == '.npz':
            data = np.load(connectivity_input)
            keys = list(data.keys())
            matrix = data[keys[0]]
        elif suffix == '.mat':
            mat_data = loadmat(connectivity_input)
            data_keys = [k for k in mat_data.keys() if not k.startswith('__')]
            if not data_keys:
                raise ValueError(f"No data found in .mat file: {connectivity_input}")
            matrix = mat_data[data_keys[0]]
        elif suffix == '.edge':
            # BrainNet Viewer edge file format
            matrix = load_edge_file(connectivity_input)
        elif suffix in ['.csv', '.txt']:
            # Detect delimiter
            with open(connectivity_input, 'r') as f:
                first_line = f.readline()
            if '\t' in first_line:
                matrix = np.loadtxt(connectivity_input, delimiter='\t')
            else:
                try:
                    matrix = np.loadtxt(connectivity_input, delimiter=',')
                except ValueError:
                    # May have header row
                    df = pd.read_csv(connectivity_input)
                    matrix = df.select_dtypes(include=[np.number]).values
        else:
            raise ValueError(f"Unsupported connectivity file format: {suffix}")
    else:
        raise TypeError(
            f"Unsupported connectivity_input type: {type(connectivity_input)}. "
            f"Expected np.ndarray, pd.DataFrame, or file path string."
        )

    # Validate matrix is square
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError(
            f"Connectivity matrix must be square. Got shape: {matrix.shape}"
        )

    # Validate size if expected
    if n_expected_nodes is not None and matrix.shape[0] != n_expected_nodes:
        raise ValueError(
            f"Connectivity matrix size ({matrix.shape[0]}) does not match "
            f"expected number of nodes ({n_expected_nodes})"
        )

    return matrix


def load_node_metrics(
    metrics_input: Union[str, pd.DataFrame],
    n_expected_nodes: Optional[int] = None
) -> pd.DataFrame:
    """
    Load node metrics from various input formats.

    Parameters
    ----------
    metrics_input : str or pd.DataFrame
        Node metrics data. Supports:
        - str: Path to CSV file
        - pd.DataFrame: Used directly
    n_expected_nodes : int, optional
        Expected number of nodes for validation

    Returns
    -------
    pd.DataFrame
        DataFrame with metrics (rows = nodes, columns = metric names)

    Raises
    ------
    ValueError
        If number of rows doesn't match expected nodes
    """
    if isinstance(metrics_input, pd.DataFrame):
        df = metrics_input.copy()
    elif isinstance(metrics_input, str):
        path = Path(metrics_input)
        if not path.exists():
            raise FileNotFoundError(f"Metrics file not found: {metrics_input}")

        # Detect delimiter
        with open(metrics_input, 'r') as f:
            first_line = f.readline()
        delimiter = '\t' if '\t' in first_line else ','
        df = pd.read_csv(metrics_input, sep=delimiter)
    else:
        raise TypeError(
            f"Unsupported metrics_input type: {type(metrics_input)}. "
            f"Expected pd.DataFrame or file path string."
        )

    if n_expected_nodes is not None and len(df) != n_expected_nodes:
        raise ValueError(
            f"Metrics DataFrame rows ({len(df)}) does not match "
            f"expected number of nodes ({n_expected_nodes})"
        )

    return df


def resolve_show_node_labels(
    show_input: Union[bool, np.ndarray, pd.Series, List, str, None],
    n_nodes: int,
) -> np.ndarray:
    """Resolve the ``show_node_labels`` parameter to a per-node bool mask.

    Used by ``create_brain_connectivity_plot`` and
    ``create_brain_connectivity_plot_with_modularity`` to control whether
    each ROI's persistent text label is rendered next to its node marker.
    Hover tooltips are independent of this mask -- they always show the
    full ROI name + metadata regardless of the value here.

    Parameters
    ----------
    show_input : bool, np.ndarray, list, pd.Series, str, or None
        The user-supplied value. Accepted forms:

        - ``True`` or ``None`` (default): every label is shown
          (returns an all-True mask).
        - ``False``: no labels are shown (returns an all-False mask).
        - 1-D ``np.ndarray`` / ``list`` / ``pd.Series`` of length
          ``n_nodes`` with 0/1 or boolean values: per-node mask, where
          ``1`` (or ``True``) shows the label and ``0`` (or ``False``)
          hides it.
        - ``str``: path to a single-column CSV / TXT / NPY file
          containing the same length-``n_nodes`` 0/1 vector. CSV may
          have a header (e.g. ``show_label``) or be headerless;
          delimiter is auto-detected (tab vs. comma).
    n_nodes : int
        Expected mask length, used both to broadcast scalar ``True`` /
        ``False`` and to validate vector inputs.

    Returns
    -------
    np.ndarray of dtype ``bool``, shape ``(n_nodes,)``.

    Raises
    ------
    ValueError
        If a vector is passed whose length does not equal ``n_nodes``,
        or whose values are not coercible to 0/1 / True/False (e.g.
        contains 0.5 or 2).
    FileNotFoundError
        If ``show_input`` is a string that does not point at an
        existing file.
    """
    # Default / None / explicit True
    if show_input is None or show_input is True:
        return np.ones(n_nodes, dtype=bool)
    # Explicit False
    if show_input is False:
        return np.zeros(n_nodes, dtype=bool)

    # File path
    if isinstance(show_input, str):
        path = Path(show_input)
        if not path.exists():
            raise FileNotFoundError(
                f"show_node_labels file not found: {show_input}"
            )
        suffix = path.suffix.lower()
        if suffix == '.npy':
            arr = np.load(path)
        else:
            # Auto-detect comma vs tab; tolerate optional header.
            with open(path, 'r') as f:
                first_line = f.readline()
            delim = '\t' if '\t' in first_line else ','
            try:
                arr = np.loadtxt(path, delimiter=delim)
            except ValueError:
                # Header present -- fall back to pandas, take first numeric column.
                df = pd.read_csv(path, sep=delim)
                num_df = df.select_dtypes(include=[np.number, bool])
                if num_df.shape[1] == 0:
                    raise ValueError(
                        f"show_node_labels file {show_input!r} contains "
                        f"no numeric / boolean column."
                    )
                arr = num_df.iloc[:, 0].to_numpy()
        return _validate_label_mask(arr, n_nodes, source=show_input)

    # Array / list / Series
    arr = np.asarray(show_input)
    return _validate_label_mask(arr, n_nodes, source='input vector')


def _validate_label_mask(
    arr: np.ndarray,
    n_nodes: int,
    source: str,
) -> np.ndarray:
    """Coerce a numeric/boolean vector into a length-``n_nodes`` bool mask.

    Accepts either booleans or 0/1 integers (or numerically-equal floats).
    Anything else -- 0.5, 2, NaN -- is rejected with a clear ValueError so
    user-supplied non-binary vectors fail fast instead of silently
    rendering wrong labels.
    """
    arr = np.squeeze(np.asarray(arr))
    if arr.ndim != 1:
        raise ValueError(
            f"show_node_labels must be 1-D; got shape {arr.shape} "
            f"from {source}."
        )
    if arr.shape[0] != n_nodes:
        raise ValueError(
            f"show_node_labels length ({arr.shape[0]}) does not match "
            f"the expected number of nodes ({n_nodes}); source: {source}."
        )
    if arr.dtype == bool:
        return arr.copy()
    # Numeric -- must be 0 or 1 exactly.
    if not np.all(np.isfinite(arr)):
        raise ValueError(
            f"show_node_labels contains non-finite values; source: {source}."
        )
    if not np.all((arr == 0) | (arr == 1)):
        bad_idx = int(np.argmax((arr != 0) & (arr != 1)))
        raise ValueError(
            f"show_node_labels must contain only 0/1 or True/False; "
            f"got value {arr[bad_idx]!r} at row {bad_idx} (source: {source})."
        )
    return arr.astype(bool)


def generate_module_colors(n_modules: int) -> List[str]:
    """
    Generate visually distinct colors for module assignments.

    Uses a combination of qualitative colors for small numbers of modules
    and HSV-distributed colors for larger numbers.

    Parameters
    ----------
    n_modules : int
        Number of distinct modules/colors needed

    Returns
    -------
    List[str]
        List of color strings in rgb() format
    """
    import colorsys

    # Predefined visually distinct colors for small numbers of modules
    # These are chosen to be easily distinguishable
    predefined_colors = [
        'rgb(227, 26, 28)',    # Red
        'rgb(51, 160, 44)',    # Green
        'rgb(31, 120, 180)',   # Blue
        'rgb(255, 127, 0)',    # Orange
        'rgb(106, 61, 154)',   # Purple
        'rgb(177, 89, 40)',    # Brown
        'rgb(255, 255, 51)',   # Yellow
        'rgb(166, 206, 227)',  # Light Blue
        'rgb(251, 154, 153)',  # Light Red
        'rgb(178, 223, 138)',  # Light Green
        'rgb(253, 191, 111)',  # Light Orange
        'rgb(202, 178, 214)',  # Light Purple
    ]

    if n_modules <= len(predefined_colors):
        return predefined_colors[:n_modules]
    else:
        # Generate colors using HSV for larger numbers
        colors = []
        for i in range(n_modules):
            # Distribute hues evenly, with good saturation and value
            hue = i / n_modules
            # Offset to avoid starting at red which may conflict
            hue = (hue + 0.05) % 1.0
            rgb = colorsys.hsv_to_rgb(hue, 0.75, 0.9)
            colors.append(f'rgb({int(rgb[0]*255)},{int(rgb[1]*255)},{int(rgb[2]*255)})')
        return colors


def is_valid_color(color_value) -> bool:
    """
    Check if a value is a valid color specification.

    Supports:
    - Named colors (e.g., 'red', 'blue', 'purple')
    - Hex colors (e.g., '#FF0000', '#f00')
    - RGB strings (e.g., 'rgb(255,0,0)')
    - RGBA strings (e.g., 'rgba(255,0,0,0.5)')

    Parameters
    ----------
    color_value : any
        Value to check

    Returns
    -------
    bool
        True if the value appears to be a valid color
    """
    if not isinstance(color_value, str):
        return False

    color_value = color_value.strip().lower()

    # Check for hex colors
    if color_value.startswith('#'):
        hex_part = color_value[1:]
        if len(hex_part) in [3, 6, 8]:
            try:
                int(hex_part, 16)
                return True
            except ValueError:
                return False

    # Check for rgb/rgba format
    if color_value.startswith('rgb'):
        return True

    # Common CSS named colors (not exhaustive, but covers most common ones)
    named_colors = {
        'red', 'green', 'blue', 'yellow', 'orange', 'purple', 'pink', 'brown',
        'black', 'white', 'gray', 'grey', 'cyan', 'magenta', 'lime', 'navy',
        'teal', 'maroon', 'olive', 'silver', 'aqua', 'fuchsia', 'gold',
        'indigo', 'violet', 'coral', 'salmon', 'khaki', 'plum', 'orchid',
        'tan', 'peru', 'sienna', 'chocolate', 'crimson', 'tomato', 'orangered',
        'darkorange', 'lightgray', 'lightgrey', 'darkgray', 'darkgrey',
        'lightblue', 'lightgreen', 'lightyellow', 'lightpink', 'lightcoral',
        'darkblue', 'darkgreen', 'darkred', 'darkcyan', 'darkmagenta',
        'skyblue', 'steelblue', 'royalblue', 'midnightblue', 'forestgreen',
        'seagreen', 'limegreen', 'springgreen', 'mediumblue', 'dodgerblue'
    }

    return color_value in named_colors


def convert_node_color_input(
    node_color_input: Union[str, np.ndarray, pd.Series, pd.DataFrame, List, str],
    n_nodes: int,
    default_color: str = 'purple'
) -> Tuple[Union[str, List[str]], Optional[Dict], Optional[np.ndarray]]:
    """
    Convert various node color input formats to a usable format.

    Parameters
    ----------
    node_color_input : str, np.ndarray, pd.Series, pd.DataFrame, list, or file path
        Node color specification. Can be:
        - Single color string: All nodes get the same color
        - numpy array of integers: Module assignments (1-indexed), auto-generate colors
        - numpy array of color strings: Per-node colors
        - pandas Series: Colors or module assignments indexed by position
        - pandas DataFrame: First column used for colors/assignments
        - list: Converted appropriately based on content
        - str (file path): Path to file (.csv, .npy) containing assignments or colors
    n_nodes : int
        Expected number of nodes (for validation)
    default_color : str
        Default color if input is invalid

    Returns
    -------
    Tuple containing:
        - colors: Either a single color string or list of colors (one per node)
        - module_color_map: Dict mapping module IDs to colors (None if single color)
        - module_assignments: Array of module assignments (None if using direct colors)

    Raises
    ------
    ValueError
        If input cannot be converted or doesn't match expected length
    """
    # Case 1: Single color string
    if isinstance(node_color_input, str) and not Path(node_color_input).exists():
        # Check if it's a valid color (not a file path)
        if is_valid_color(node_color_input):
            return node_color_input, None, None

    # Convert input to array
    arr = None

    # Case 2: File path
    if isinstance(node_color_input, str):
        path = Path(node_color_input)
        if not path.exists():
            raise FileNotFoundError(f"Node color file not found: {node_color_input}")

        suffix = path.suffix.lower()

        if suffix == '.npy':
            arr = np.load(node_color_input).flatten()
        elif suffix == '.csv':
            df = pd.read_csv(node_color_input)
            # Check for common column names
            if 'module' in df.columns:
                arr = df['module'].values
            elif 'color' in df.columns:
                arr = df['color'].values
            else:
                # Use first column
                arr = df.iloc[:, 0].values if len(df.columns) == 1 else df.iloc[:, -1].values
        elif suffix == '.txt':
            try:
                arr = np.loadtxt(node_color_input).flatten()
            except ValueError:
                # May be color strings
                with open(node_color_input, 'r') as f:
                    arr = np.array([line.strip() for line in f.readlines()])
        else:
            raise ValueError(f"Unsupported file format for node colors: {suffix}")

    # Case 3: numpy array
    elif isinstance(node_color_input, np.ndarray):
        arr = node_color_input.flatten()

    # Case 4: pandas Series
    elif isinstance(node_color_input, pd.Series):
        arr = node_color_input.values

    # Case 5: pandas DataFrame
    elif isinstance(node_color_input, pd.DataFrame):
        # Check for common column names
        if 'module' in node_color_input.columns:
            arr = node_color_input['module'].values
        elif 'color' in node_color_input.columns:
            arr = node_color_input['color'].values
        else:
            # Use first numeric column or first column
            numeric_cols = node_color_input.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                arr = node_color_input[numeric_cols[0]].values
            else:
                arr = node_color_input.iloc[:, 0].values

    # Case 6: list
    elif isinstance(node_color_input, list):
        arr = np.array(node_color_input)

    else:
        raise TypeError(
            f"Unsupported node_color type: {type(node_color_input)}. "
            f"Expected str, np.ndarray, pd.Series, pd.DataFrame, list, or file path."
        )

    # Validate length
    if len(arr) != n_nodes:
        raise ValueError(
            f"Node color array length ({len(arr)}) does not match "
            f"number of nodes ({n_nodes})"
        )

    # Determine if we have module assignments (integers) or direct colors
    # Check if all values are numeric integers
    is_integer_assignments = False
    try:
        # Try to convert to int and check if they're the same
        int_arr = arr.astype(float).astype(int)
        if np.allclose(arr.astype(float), int_arr):
            is_integer_assignments = True
            arr = int_arr
    except (ValueError, TypeError):
        # Not numeric, must be color strings
        pass

    if is_integer_assignments:
        # These are module assignments - generate colors
        unique_modules = np.unique(arr)
        n_modules = len(unique_modules)

        # Generate colors for modules
        module_colors = generate_module_colors(n_modules)

        # Create module to color mapping (1-indexed expected)
        module_color_map = {module: module_colors[i] for i, module in enumerate(sorted(unique_modules))}

        # Create per-node color list
        node_colors = [module_color_map[m] for m in arr]

        return node_colors, module_color_map, arr

    else:
        # These should be color values - validate them
        invalid_colors = []
        for i, c in enumerate(arr):
            if not is_valid_color(str(c)):
                invalid_colors.append((i, c))

        if invalid_colors:
            # Show first few invalid colors
            examples = invalid_colors[:5]
            raise ValueError(
                f"Invalid color values found at indices: {examples}. "
                f"Colors must be valid CSS color names, hex codes (#RRGGBB), "
                f"or rgb() format strings."
            )

        return list(arr), None, None


def load_edge_color_matrix(
    color_input: Union[str, np.ndarray, pd.DataFrame],
    n_expected_nodes: Optional[int] = None
) -> Tuple[np.ndarray, Optional[Dict]]:
    """
    Load a per-edge color matrix used to color edges in a brain connectivity plot.

    The color matrix has the SAME shape as the connectivity matrix
    (``n_nodes x n_nodes``). Each cell ``[i, j]`` specifies the color to use
    when drawing the edge between ROI i and ROI j. The cell can hold either:

    1. **A color string** -- a CSS named color (``"red"``), a hex code
       (``"#FF0000"``, ``"#f00"``), or an ``"rgb(R,G,B)"`` /
       ``"rgba(R,G,B,A)"`` string. The string is used as-is for the edge.
    2. **An integer label** -- a categorical class id (e.g. ``1``, ``2``,
       ``3``). All edges sharing the same integer get the same auto-generated
       color from a distinct palette (the same palette as
       :func:`generate_module_colors`). This lets you label edges with
       integer "edge groups" without having to pick colors yourself.

    Empty cells, ``NaN`` and the integer ``0`` are treated as "no color"
    and the corresponding edge is **skipped** when drawing -- exactly as if
    that cell were missing from the connectivity matrix.

    Parameters
    ----------
    color_input : str, np.ndarray, or pd.DataFrame
        Color matrix or path to a file containing one. Supported file
        formats are ``.csv``, ``.txt`` (delimiter auto-detected) and
        ``.npy``. CSVs are loaded as strings; integer columns are auto-
        promoted to a categorical palette.
    n_expected_nodes : int, optional
        Expected dimensionality. If provided, the matrix must be square
        with this size; otherwise a :class:`ValueError` is raised.

    Returns
    -------
    color_matrix : np.ndarray of dtype object
        ``n_nodes x n_nodes`` array of color strings (or empty string ``""``
        for cells that should be skipped). All non-empty cells are
        guaranteed to be valid color strings ready to hand to plotly.
    label_to_color : dict or None
        When the input was integer-categorical, a mapping
        ``{int_label: color_string}`` describing how labels were assigned
        to colors. ``None`` when the input was already raw color strings.

    Raises
    ------
    ValueError
        If the loaded matrix is not square, doesn't match
        ``n_expected_nodes``, or contains values that are neither valid
        colors nor integers.
    """
    # ---- 1. Load raw matrix ----
    if isinstance(color_input, np.ndarray):
        raw = color_input
    elif isinstance(color_input, pd.DataFrame):
        raw = color_input.values
    elif isinstance(color_input, str):
        path = Path(color_input)
        if not path.exists():
            raise FileNotFoundError(f"Edge color matrix file not found: {color_input}")
        suffix = path.suffix.lower()
        if suffix == '.npy':
            raw = np.load(color_input, allow_pickle=True)
        elif suffix in ('.csv', '.txt'):
            with open(color_input, 'r') as f:
                first_line = f.readline()
            delimiter = '\t' if '\t' in first_line else ','
            # Always read as strings so '#FF0000' / 'red' / '1' all parse.
            df = pd.read_csv(color_input, sep=delimiter, header=None, dtype=str)
            raw = df.values
        else:
            raise ValueError(
                f"Unsupported edge color matrix format: {suffix}. "
                f"Expected .csv, .txt, or .npy."
            )
    else:
        raise TypeError(
            f"Unsupported color_input type: {type(color_input)}. "
            f"Expected np.ndarray, pd.DataFrame, or file path string."
        )

    if raw.ndim != 2 or raw.shape[0] != raw.shape[1]:
        raise ValueError(
            f"Edge color matrix must be square. Got shape: {raw.shape}"
        )

    if n_expected_nodes is not None and raw.shape[0] != n_expected_nodes:
        raise ValueError(
            f"Edge color matrix size ({raw.shape[0]}) does not match "
            f"expected number of nodes ({n_expected_nodes})"
        )

    n = raw.shape[0]
    out = np.full((n, n), "", dtype=object)

    # ---- 2. Try to interpret as integer categorical labels ----
    # If every non-empty cell can be parsed as an integer, treat as
    # categorical and assign colors from generate_module_colors.
    def _is_empty(v):
        if v is None:
            return True
        if isinstance(v, float) and np.isnan(v):
            return True
        if isinstance(v, str) and v.strip() == "":
            return True
        return False

    flat = []
    all_int = True
    for i in range(n):
        for j in range(n):
            v = raw[i, j]
            if _is_empty(v):
                continue
            try:
                iv = int(float(v))
                # Treat 0 as "no color"
                if iv == 0:
                    continue
                flat.append(iv)
            except (ValueError, TypeError):
                all_int = False
                break
        if not all_int:
            break

    label_to_color: Optional[Dict] = None

    if all_int and flat:
        unique_labels = sorted(set(flat))
        palette = generate_module_colors(len(unique_labels))
        label_to_color = {lab: palette[idx] for idx, lab in enumerate(unique_labels)}
        for i in range(n):
            for j in range(n):
                v = raw[i, j]
                if _is_empty(v):
                    continue
                try:
                    iv = int(float(v))
                except (ValueError, TypeError):
                    continue
                if iv == 0:
                    continue
                out[i, j] = label_to_color[iv]
        return out, label_to_color

    # ---- 3. Otherwise interpret as raw color strings ----
    bad: List[Tuple[int, int, object]] = []
    for i in range(n):
        for j in range(n):
            v = raw[i, j]
            if _is_empty(v):
                continue
            # Numeric 0 (e.g. from a .npy of zeros) -> skip
            if isinstance(v, (int, float)) and v == 0:
                continue
            sv = str(v).strip()
            if sv == "" or sv == "0":
                continue
            if not is_valid_color(sv):
                bad.append((i, j, v))
                if len(bad) >= 5:
                    break
            else:
                out[i, j] = sv
        if len(bad) >= 5:
            break

    if bad:
        raise ValueError(
            f"Edge color matrix contains values that are neither valid colors "
            f"nor integer labels. First offending cells (i, j, value): {bad}. "
            f"Use CSS color names ('red'), hex codes ('#FF0000'), 'rgb(...)' "
            f"strings, or integer labels (1, 2, 3, ...)."
        )

    return out, None


def transform_pvalue_matrix(
    pvalue_matrix: np.ndarray,
    pvalue_threshold: float = 0.05,
    sign_matrix: Optional[np.ndarray] = None,
    epsilon: float = 1e-300,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert a matrix of p-values into a "weight" matrix suitable for plotting.

    The transform is ``-log10(p)``. This is the standard way to visualise
    significance and is naturally bounded for the p-values that come out of
    typical experiments:

    ===========  =============
    p-value      -log10(p)
    ===========  =============
    0.05          1.30
    0.01          2.00
    0.001         3.00
    0.0001        4.00
    0.00001       5.00
    ===========  =============

    Compared with ``1/p``, this transform avoids exploding to huge values
    (``1/0.002 = 500``) and instead grows logarithmically, which makes the
    edge widths in the brain plot stay readable.

    Parameters
    ----------
    pvalue_matrix : np.ndarray
        Square matrix of p-values. Cells should normally be in ``(0, 1]``.
        Cells with ``NaN``, values ``<= 0`` or values ``> 1`` are treated
        as "no edge" and zeroed out in the result.
    pvalue_threshold : float, optional
        Cells with ``p > pvalue_threshold`` are zeroed out (i.e. not drawn).
        Default ``0.05``. Set to ``1.0`` to keep every p-value.
    sign_matrix : np.ndarray, optional
        Optional matrix of the same shape as ``pvalue_matrix`` containing the
        sign of the underlying effect (typically ``+1`` for positive,
        ``-1`` for negative, ``0`` for unsigned). When provided, the
        returned weight matrix is multiplied by ``sign(sign_matrix)`` so
        that positive effects come out as positive ``-log10(p)`` values
        and negative effects come out as negative ``-log10(p)`` values.
        This is what makes the downstream pos/neg edge coloring work for
        signed p-values.
    epsilon : float, optional
        Small floor used to avoid ``-log10(0) = inf`` for cells with
        exactly ``p == 0``. Default ``1e-300``.

    Returns
    -------
    weight_matrix : np.ndarray
        ``-log10(p)`` (signed if ``sign_matrix`` was provided), with cells
        above ``pvalue_threshold`` or otherwise invalid set to 0.
    pvalue_clean : np.ndarray
        The original p-value matrix with the same cells zeroed out, so the
        caller can still display the raw p-value in hover text.
    """
    p = np.array(pvalue_matrix, dtype=float, copy=True)
    pclean = np.array(pvalue_matrix, dtype=float, copy=True)

    invalid = (
        np.isnan(p)
        | (p <= 0)
        | (p > 1)
        | (p > pvalue_threshold)
    )

    p_safe = np.where(invalid, 1.0, np.maximum(p, epsilon))
    weights = -np.log10(p_safe)
    weights[invalid] = 0.0
    pclean[invalid] = 0.0

    if sign_matrix is not None:
        s = np.array(sign_matrix, dtype=float, copy=False)
        if s.shape != weights.shape:
            raise ValueError(
                f"sign_matrix shape {s.shape} does not match pvalue_matrix "
                f"shape {weights.shape}"
            )
        sign = np.sign(s)
        # Where the sign is 0 we leave the magnitude alone (treated as
        # positive so it still shows up).
        sign = np.where(sign == 0, 1.0, sign)
        weights = weights * sign

    return weights, pclean


def get_node_edge_connectivity(
    G,
    node_idx: int,
    edge_type: str = 'both'
) -> bool:
    """
    Check if a node has edges of a specific type.

    Parameters
    ----------
    G : networkx.Graph
        Graph containing edges with 'weight' attribute
    node_idx : int
        Node index to check
    edge_type : str
        'positive', 'negative', or 'both'

    Returns
    -------
    bool
        True if node has edges of specified type
    """
    if node_idx not in G.nodes():
        return False

    for neighbor in G.neighbors(node_idx):
        weight = G[node_idx][neighbor].get('weight', 0)
        if edge_type == 'positive' and weight > 0:
            return True
        elif edge_type == 'negative' and weight < 0:
            return True
        elif edge_type == 'both':
            return True

    return False


# ---------------------------------------------------------------------------
# Label-volume hygiene + atlas/mesh alignment sanity checks
# ---------------------------------------------------------------------------
# These helpers answer the "is my data actually consistent before I plot?"
# questions that bite people working with a new atlas + mesh pair:
#   - is the atlas integer-labeled, or stored as floats (-> NaN COGs)?
#   - do the ROI centre-of-gravity coordinates land *inside* the brain mesh,
#     or is the atlas in a different template space (-> nodes float off the brain)?
#   - did a bilateral/merged parcellation collapse every COG onto the midline?
# See the "Checking atlas/mesh alignment" tutorial for worked examples.


def _load_label_image(volume):
    """Accept a NIfTI path or a nibabel image; return (nibabel image, float data)."""
    import nibabel as nib
    if isinstance(volume, (str, Path)):
        img = nib.load(str(volume))
    elif hasattr(volume, "affine") and hasattr(volume, "dataobj"):
        img = volume
    else:
        raise TypeError(
            "volume must be a path to a NIfTI file or a nibabel image, "
            f"got {type(volume)!r}"
        )
    data = np.asanyarray(img.dataobj).astype(np.float64)
    return img, data


def inspect_label_volume(volume) -> Dict:
    """
    Report whether a label volume is cleanly integer-labeled.

    This is the "how do I check?" companion to :func:`clean_label_volume`. Some
    atlases store integer ROI labels as floats with tiny rounding error (e.g.
    ``0.9999999`` for label 1), which breaks an exact ``volume == label`` test
    and yields all-NaN COGs.

    Parameters
    ----------
    volume : str | pathlib.Path | nibabel image
        The label volume to inspect.

    Returns
    -------
    dict
        ``shape``, ``dtype``, ``is_integer_labeled`` (True only if values are
        *bit-exact* integers — what an exact ``volume == label`` match needs),
        ``near_integer`` (within 0.5 of an integer, i.e. it really is a label map),
        ``max_label_deviation`` (largest distance of any voxel from the nearest
        integer), ``n_labels``, ``label_min``, ``label_max``, ``contiguous``.
    """
    img, data = _load_label_image(volume)
    rounded = np.rint(data)
    max_dev = float(np.max(np.abs(data - rounded))) if data.size else 0.0
    uniq = np.unique(rounded[rounded > 0]).astype(int)
    # The bug that produces NaN COGs is an *exact* `volume == label` match
    # failing, so what matters is bit-exact integers, not "close to integer":
    # a 0.9999999997 value (deviation ~1e-8) still breaks exact matching.
    return {
        "shape": tuple(int(s) for s in data.shape),
        "dtype": str(np.asanyarray(img.dataobj).dtype),
        "is_integer_labeled": bool(max_dev == 0.0),
        "near_integer": bool(max_dev < 0.5),
        "max_label_deviation": max_dev,
        "n_labels": int(uniq.size),
        "label_min": int(uniq.min()) if uniq.size else None,
        "label_max": int(uniq.max()) if uniq.size else None,
        "contiguous": bool(
            uniq.size and np.array_equal(uniq, np.arange(uniq.min(), uniq.max() + 1))
        ),
    }


def clean_label_volume(volume, output_path=None, dtype="int16"):
    """
    Round a label volume's values to the nearest integer (and optionally save it).

    Fixes atlases whose integer ROI labels are stored as floats with rounding
    error, which otherwise make :func:`coordinate_function`'s exact label match
    return zero voxels (all-NaN COGs). A no-op in value for a clean integer atlas.

    Parameters
    ----------
    volume : str | pathlib.Path | nibabel image
        The label volume to clean.
    output_path : str | pathlib.Path, optional
        If given, save the cleaned volume here (``.nii`` or ``.nii.gz``).
    dtype : str, optional
        Integer dtype for the output (default ``'int16'``).

    Returns
    -------
    nibabel.Nifti1Image
        The cleaned, integer-typed image (a fresh header derived from the affine,
        so any inherited ``scl_slope``/``scl_inter`` scaling is dropped).
    """
    import nibabel as nib
    img, data = _load_label_image(volume)
    rounded = np.rint(data).astype(dtype)
    out_img = nib.Nifti1Image(rounded, img.affine)
    out_img.header.set_data_dtype(dtype)
    if output_path is not None:
        out_path = Path(output_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        nib.save(out_img, str(out_path))
    return out_img


def _labeled_voxel_world_bbox(img, data):
    """World-coordinate bounding box of the labeled (nonzero) voxels."""
    lab = np.rint(data)
    ijk = np.argwhere(lab > 0)
    if ijk.size == 0:
        return None, None
    mn = ijk.min(0)
    mx = ijk.max(0)
    corners = np.array(
        [[a, b, c, 1]
         for a in (mn[0], mx[0]) for b in (mn[1], mx[1]) for c in (mn[2], mx[2])]
    ).T
    world = (img.affine @ corners)[:3].T
    return world.min(0), world.max(0)


def _load_mesh_vertices(mesh):
    """Accept a mesh path, an (vertices, faces) tuple, or a vertices array."""
    if isinstance(mesh, (str, Path)):
        from .mesh import load_mesh_file
        vertices, _ = load_mesh_file(str(mesh))
        return np.asarray(vertices, dtype=float)
    if isinstance(mesh, tuple) and len(mesh) == 2:
        return np.asarray(mesh[0], dtype=float)
    arr = np.asarray(mesh, dtype=float)
    if arr.ndim == 2 and arr.shape[1] == 3:
        return arr
    raise TypeError("mesh must be a path, (vertices, faces) tuple, or (N, 3) array")


def _read_coords_table(coords):
    """Load a coords CSV (comma or tab) or accept a DataFrame; need cog_x/y/z."""
    if isinstance(coords, pd.DataFrame):
        df = coords
    else:
        with open(coords, "r") as f:
            first = f.readline()
        sep = "\t" if "\t" in first else ","
        df = pd.read_csv(coords, sep=sep)
    missing = [c for c in ("cog_x", "cog_y", "cog_z") if c not in df.columns]
    if missing:
        raise ValueError(
            f"Coords table is missing required columns {missing}. "
            f"Found: {list(df.columns)}"
        )
    return df


def check_coords_in_mesh(coords, mesh, midline_eps=2.0, surface_pad_mm=2.0):
    """
    Check that ROI centre-of-gravity coordinates fall inside a brain mesh.

    Catches the most common "my nodes float off the brain" failures: an atlas in
    a different template space than the mesh, NaN COGs from a broken extraction,
    or a bilateral/merged parcellation whose COGs all collapse onto the midline.

    The inside test uses the mesh's **convex hull** (half-space inequalities via
    :class:`scipy.spatial.ConvexHull`) — fast even for a 200k-vertex mesh, and
    decisive for space mismatches. A handful of genuinely-interior COGs can sit
    just outside the hull near deep sulci, so a small number of "outside" points
    is reported as a warning rather than a hard failure.

    Parameters
    ----------
    coords : str | pathlib.Path | pandas.DataFrame
        Coordinates with ``cog_x``, ``cog_y``, ``cog_z`` columns (e.g. the
        ``*_comma.csv`` from ``hlplot coords generate``).
    mesh : str | pathlib.Path | tuple | numpy.ndarray
        Mesh path, an ``(vertices, faces)`` tuple, or an ``(N, 3)`` vertex array.
    midline_eps : float, optional
        ``|x| < midline_eps`` mm counts a COG as "on the midline" (default 2.0).
    surface_pad_mm : float, optional
        Tolerance (mm) added to the convex-hull half-space test (default 2.0).

    Returns
    -------
    dict
        Keys include ``n_rois``, ``n_nan``, ``coords_bbox``, ``mesh_bbox``,
        ``coords_bbox_within_mesh``, ``n_inside``, ``n_outside``,
        ``outside_names``, ``nearest_vertex_dist_mm`` (max/mean), ``n_on_midline``,
        ``midline_fraction``, ``verdict`` (``'PASS'`` | ``'WARN'`` | ``'FAIL'``)
        and ``messages`` (list of human-readable reasons).
    """
    from scipy.spatial import ConvexHull, cKDTree

    df = _read_coords_table(coords)
    names = (df["roi_name"].astype(str).tolist()
             if "roi_name" in df.columns else [str(i) for i in range(len(df))])
    xyz = df[["cog_x", "cog_y", "cog_z"]].to_numpy(dtype=float)

    nan_mask = ~np.isfinite(xyz).all(axis=1)
    n_nan = int(nan_mask.sum())
    valid = xyz[~nan_mask]

    verts = _load_mesh_vertices(mesh)
    mesh_min, mesh_max = verts.min(0), verts.max(0)

    messages = []
    result = {
        "n_rois": int(len(df)),
        "n_nan": n_nan,
        "mesh_bbox": (mesh_min.tolist(), mesh_max.tolist()),
    }

    if valid.size == 0:
        result.update({
            "coords_bbox": None, "coords_bbox_within_mesh": False,
            "n_inside": 0, "n_outside": 0, "outside_names": [],
            "nearest_vertex_dist_mm": {"max": None, "mean": None},
            "n_on_midline": 0, "midline_fraction": 0.0,
            "verdict": "FAIL",
            "messages": ["All COG coordinates are NaN/missing."],
        })
        return result

    c_min, c_max = valid.min(0), valid.max(0)
    within = bool(np.all(c_min >= mesh_min - surface_pad_mm) and
                  np.all(c_max <= mesh_max + surface_pad_mm))

    # Inside-the-hull test via half-space inequalities (a*x+b*y+c*z+off <= 0).
    hull = ConvexHull(verts)
    eqs = hull.equations
    inside_mask = np.all(valid @ eqs[:, :3].T + eqs[:, 3] <= surface_pad_mm, axis=1)
    n_inside = int(inside_mask.sum())
    n_outside = int((~inside_mask).sum())
    valid_names = [n for n, bad in zip(np.array(names)[~nan_mask], ~inside_mask) if bad]

    dist, _ = cKDTree(verts).query(valid)
    n_mid = int((np.abs(valid[:, 0]) < midline_eps).sum())
    mid_frac = n_mid / len(valid)

    result.update({
        "coords_bbox": (c_min.tolist(), c_max.tolist()),
        "coords_bbox_within_mesh": within,
        "n_inside": n_inside,
        "n_outside": n_outside,
        "outside_names": valid_names[:50],
        "nearest_vertex_dist_mm": {"max": float(dist.max()), "mean": float(dist.mean())},
        "n_on_midline": n_mid,
        "midline_fraction": float(mid_frac),
    })

    # Verdict.
    verdict = "PASS"
    if n_nan:
        verdict = "FAIL"
        messages.append(f"{n_nan} ROI(s) have NaN COGs (broken extraction or "
                        f"labels not found in the volume).")
    if not within:
        verdict = "FAIL"
        messages.append("COG bounding box is not inside the mesh bounding box — "
                        "atlas and mesh are likely in different spaces.")
    frac_outside = n_outside / len(valid)
    if frac_outside > 0.20:
        verdict = "FAIL"
        messages.append(f"{n_outside}/{len(valid)} COGs fall outside the mesh hull "
                        f"({frac_outside:.0%}) — probable space mismatch.")
    elif n_outside > 0 and verdict != "FAIL":
        verdict = "WARN"
        messages.append(f"{n_outside}/{len(valid)} COGs sit just outside the convex "
                        f"hull (often fine for deep/sulcal regions).")
    if mid_frac > 0.5:
        verdict = "FAIL" if verdict != "FAIL" else verdict
        messages.append(f"{n_mid}/{len(valid)} COGs lie on the midline "
                        f"(|x|<{midline_eps}mm) — looks like a bilateral/merged "
                        f"parcellation collapsing both hemispheres into one label.")
    if not messages:
        messages.append("All COGs are finite and inside the mesh.")
    result["verdict"] = verdict
    result["messages"] = messages
    return result


def compare_volume_mesh_space(volume, mesh):
    """
    Compare an atlas volume's labeled extent to a mesh's extent ("same space?").

    A coarse but decisive check for template-space mismatches (e.g. an NMT-space
    atlas paired with a native-space mesh): if the labeled-voxel world bounding
    box barely overlaps the mesh bounding box, they are not in the same space.

    Parameters
    ----------
    volume : str | pathlib.Path | nibabel image
        The label volume.
    mesh : str | pathlib.Path | tuple | numpy.ndarray
        Mesh path, ``(vertices, faces)`` tuple, or ``(N, 3)`` vertex array.

    Returns
    -------
    dict
        ``volume_bbox``, ``mesh_bbox``, ``bbox_overlap_fraction`` (intersection /
        smaller box volume), ``centroid_offset_mm``, ``same_space`` (bool),
        ``verdict`` and ``messages``.
    """
    img, data = _load_label_image(volume)
    v_min, v_max = _labeled_voxel_world_bbox(img, data)
    verts = _load_mesh_vertices(mesh)
    m_min, m_max = verts.min(0), verts.max(0)

    if v_min is None:
        return {
            "volume_bbox": None, "mesh_bbox": (m_min.tolist(), m_max.tolist()),
            "bbox_overlap_fraction": 0.0, "centroid_offset_mm": None,
            "same_space": False, "verdict": "FAIL",
            "messages": ["Volume has no labeled voxels."],
        }

    inter_min = np.maximum(v_min, m_min)
    inter_max = np.minimum(v_max, m_max)
    inter = np.clip(inter_max - inter_min, 0, None)
    inter_vol = float(np.prod(inter))
    v_vol = float(np.prod(v_max - v_min))
    m_vol = float(np.prod(m_max - m_min))
    overlap = inter_vol / max(min(v_vol, m_vol), 1e-9)
    centroid_off = float(np.linalg.norm((v_min + v_max) / 2 - (m_min + m_max) / 2))

    same = overlap > 0.5
    verdict = "PASS" if same else "FAIL"
    messages = (["Atlas and mesh extents overlap — consistent with the same space."]
                if same else
                [f"Atlas/mesh bounding boxes overlap only {overlap:.0%} "
                 f"(centroid offset {centroid_off:.1f} mm) — likely different "
                 f"template spaces."])
    return {
        "volume_bbox": (v_min.tolist(), v_max.tolist()),
        "mesh_bbox": (m_min.tolist(), m_max.tolist()),
        "bbox_overlap_fraction": overlap,
        "centroid_offset_mm": centroid_off,
        "same_space": same,
        "verdict": verdict,
        "messages": messages,
    }
