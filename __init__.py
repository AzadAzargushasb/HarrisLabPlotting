"""
HarrisLabPlotting
=================
A Python package for brain connectivity and modularity visualization.

Modules
-------
- mesh: Brain mesh loading utilities
- camera: Camera control for 3D visualizations
- connectivity: Basic brain connectivity plotting
- modularity: Enhanced modularity visualization with PC classification
- roi_coordinates: ROI coordinate extraction and mapping
- loaders: Data loading utilities for modularity results
- utils: Helper functions for visualization calculations

Example Usage
-------------
Basic connectivity plot:
    >>> from HarrisLabPlotting import load_mesh_file, create_brain_connectivity_plot
    >>> vertices, faces = load_mesh_file("brain.gii")
    >>> fig, stats = create_brain_connectivity_plot(vertices, faces, roi_df, matrix)

Enhanced modularity visualization:
    >>> from HarrisLabPlotting import create_enhanced_modularity_visualization
    >>> from HarrisLabPlotting import CameraController
    >>> fig, stats = create_enhanced_modularity_visualization(
    ...     vertices, faces, roi_df, matrix, module_data,
    ...     camera_view='anterior'
    ... )

Run full pipeline:
    >>> from HarrisLabPlotting import run_enhanced_visualization_pipeline
    >>> results = run_enhanced_visualization_pipeline(
    ...     matrix_path="matrices.npy",
    ...     netneurotools_results_dir="results/",
    ...     mesh_file="brain.gii",
    ...     roi_coords_file="rois.csv",
    ...     output_dir="output/",
    ...     k_value=9
    ... )

Cross-module imports within the package:
    # From any module in this package, you can import from other modules:
    from HarrisLabPlotting.mesh import load_mesh_file
    from HarrisLabPlotting.camera import CameraController
    from HarrisLabPlotting.utils import classify_node_role
"""

__version__ = "1.0.0"
__author__ = "Harris Lab"

# Mesh utilities
from .mesh import load_mesh_file

# Camera control
from .camera import CameraController

# Utility functions
from .utils import (
    NumpyEncoder,
    classify_node_role,
    calculate_node_size,
    calculate_edge_width,
    filter_edges_by_module,
    threshold_matrix_top_n,
    load_node_file,
    load_edge_file,
    node_edge_to_roi_matrix,
    convert_node_size_input,
    convert_node_color_input,
    generate_module_colors,
    load_connectivity_input,
    load_node_metrics
)

# Data loaders
from .loaders import NetNeurotoolsModularityLoader

# Basic connectivity visualization
from .connectivity import (
    create_brain_connectivity_plot,
    quick_brain_plot,
    create_brain_connectivity_plot_with_modularity
)

# Enhanced modularity visualization
from .modularity import (
    create_enhanced_modularity_visualization,
    create_interactive_camera_control_panel,
    run_enhanced_visualization_pipeline
)

# ROI coordinate tools
from .roi_coordinates import (
    coordinate_function,
    map_coordinate,
    load_and_clean_coordinates,
    load_matrix_replace_nan
)

# Define what gets exported with "from HarrisLabPlotting import *"
__all__ = [
    # Version info
    "__version__",
    "__author__",

    # Mesh
    "load_mesh_file",

    # Camera
    "CameraController",

    # Utils
    "NumpyEncoder",
    "classify_node_role",
    "calculate_node_size",
    "calculate_edge_width",
    "filter_edges_by_module",
    "threshold_matrix_top_n",
    "load_node_file",
    "load_edge_file",
    "node_edge_to_roi_matrix",
    "convert_node_size_input",
    "convert_node_color_input",
    "generate_module_colors",
    "load_connectivity_input",
    "load_node_metrics",

    # Loaders
    "NetNeurotoolsModularityLoader",

    # Connectivity
    "create_brain_connectivity_plot",
    "quick_brain_plot",
    "create_brain_connectivity_plot_with_modularity",

    # Modularity
    "create_enhanced_modularity_visualization",
    "create_interactive_camera_control_panel",
    "run_enhanced_visualization_pipeline",

    # ROI Coordinates
    "coordinate_function",
    "map_coordinate",
    "load_and_clean_coordinates",
    "load_matrix_replace_nan",
]
