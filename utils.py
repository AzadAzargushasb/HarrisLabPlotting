"""
Utility Functions
=================
Helper functions for brain visualization calculations.
"""

import numpy as np
import json
from typing import Tuple


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
