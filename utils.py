"""
Utility Functions
=================
Helper functions for brain visualization calculations.
"""

import numpy as np
import pandas as pd
import json
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
    Classify node role based on within-module z-score and participation coefficient.

    Parameters
    ----------
    z_score : float
        Within-module z-score
    pc : float
        Participation coefficient

    Returns
    -------
    tuple
        (role_name, color) for the node
    """
    if z_score < 0.05 and pc < 0.05:
        return "Ultra-peripheral", "#E8E8E8"  # Very light gray
    elif z_score < 2.5 and pc < 0.62:
        if abs(pc - 0.5) < 0.1:  # Kinless nodes
            return "Kinless", "#FFB6C1"  # Light pink
        else:
            return "Peripheral", "#B0B0B0"  # Light gray
    elif z_score < 2.5 and pc >= 0.62:
        return "Satellite Connector", "#87CEEB"  # Sky blue
    elif z_score >= 2.5 and pc < 0.3:
        return "Provincial Hub", "#FFD700"  # Gold
    elif z_score >= 2.5 and pc >= 0.3:
        return "Connector Hub", "#FF4500"  # Red-orange
    else:
        return "Unclassified", "#808080"  # Gray


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
