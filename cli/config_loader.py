"""
Configuration file loader for HarrisLabPlotting CLI.
Supports YAML configuration files.
"""

import os
from pathlib import Path
from typing import Dict, Any, Optional, List
import yaml


class ConfigError(Exception):
    """Custom exception for configuration errors."""
    pass


# Default configuration values
DEFAULT_CONFIG = {
    "output_format": "html",
    "plot": {
        "node_size": 10,
        "node_color": "purple",
        "edge_threshold": 0.0,
        "edge_width_range": [1, 5],
        "opacity": 0.3,
        "title": "Brain Connectivity"
    },
    "camera": {
        "view": "anterior"
    },
    "modularity": {
        "enabled": False,
        "edge_color_mode": "module"
    }
}


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from a YAML file.

    Parameters
    ----------
    config_path : str
        Path to the YAML configuration file.

    Returns
    -------
    Dict[str, Any]
        Configuration dictionary.

    Raises
    ------
    ConfigError
        If the file cannot be loaded or is invalid.
    """
    config_path = Path(config_path)

    if not config_path.exists():
        raise ConfigError(f"Configuration file not found: {config_path}")

    if not config_path.suffix.lower() in [".yaml", ".yml"]:
        raise ConfigError(f"Configuration file must be YAML format (.yaml or .yml): {config_path}")

    try:
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
    except yaml.YAMLError as e:
        raise ConfigError(f"Invalid YAML in configuration file: {e}")
    except Exception as e:
        raise ConfigError(f"Error reading configuration file: {e}")

    if config is None:
        config = {}

    # Merge with defaults
    merged_config = _merge_configs(DEFAULT_CONFIG.copy(), config)

    return merged_config


def _merge_configs(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merge two configuration dictionaries."""
    result = base.copy()

    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _merge_configs(result[key], value)
        else:
            result[key] = value

    return result


def validate_config(config: Dict[str, Any]) -> List[str]:
    """
    Validate a configuration dictionary.

    Parameters
    ----------
    config : Dict[str, Any]
        Configuration dictionary to validate.

    Returns
    -------
    List[str]
        List of validation errors (empty if valid).
    """
    errors = []

    # Check required file paths exist (if specified)
    file_fields = ["mesh_file", "roi_coords_file", "connectivity_matrix"]
    for field in file_fields:
        if field in config:
            file_path = Path(config[field])
            if not file_path.exists():
                errors.append(f"File not found: {field} = {config[field]}")

    # Validate output format
    valid_formats = ["html", "png", "pdf", "svg", "jpeg", "webp"]
    output_format = config.get("output_format", "html").lower()
    if output_format not in valid_formats:
        errors.append(f"Invalid output_format: {output_format}. Must be one of: {', '.join(valid_formats)}")

    # Validate plot settings
    if "plot" in config:
        plot_config = config["plot"]

        # Validate node_size
        if "node_size" in plot_config:
            node_size = plot_config["node_size"]
            if not isinstance(node_size, (int, float)) or node_size <= 0:
                errors.append(f"Invalid node_size: {node_size}. Must be a positive number.")

        # Validate edge_threshold
        if "edge_threshold" in plot_config:
            threshold = plot_config["edge_threshold"]
            if not isinstance(threshold, (int, float)):
                errors.append(f"Invalid edge_threshold: {threshold}. Must be a number.")

        # Validate opacity
        if "opacity" in plot_config:
            opacity = plot_config["opacity"]
            if not isinstance(opacity, (int, float)) or not (0 <= opacity <= 1):
                errors.append(f"Invalid opacity: {opacity}. Must be between 0 and 1.")

    # Validate camera settings
    if "camera" in config:
        camera_config = config["camera"]
        valid_views = ["anterior", "posterior", "lateral-left", "lateral-right",
                       "superior", "inferior", "dorsal", "ventral"]

        if "view" in camera_config:
            view = camera_config["view"].lower()
            if view not in valid_views and not all(k in camera_config for k in ["eye", "center", "up"]):
                errors.append(f"Invalid camera view: {view}. Must be one of: {', '.join(valid_views)} or specify custom eye/center/up.")

    # Validate modularity settings
    if "modularity" in config:
        mod_config = config["modularity"]

        if mod_config.get("enabled", False):
            if "module_file" not in mod_config:
                errors.append("Modularity enabled but no module_file specified.")
            elif not Path(mod_config["module_file"]).exists():
                errors.append(f"Module file not found: {mod_config['module_file']}")

        if "edge_color_mode" in mod_config:
            mode = mod_config["edge_color_mode"].lower()
            if mode not in ["module", "sign"]:
                errors.append(f"Invalid edge_color_mode: {mode}. Must be 'module' or 'sign'.")

    # Validate batch settings
    if "batch" in config:
        batch_config = config["batch"]
        if not isinstance(batch_config, list):
            errors.append("Batch configuration must be a list.")
        else:
            for i, item in enumerate(batch_config):
                if not isinstance(item, dict):
                    errors.append(f"Batch item {i} must be a dictionary.")
                elif "name" not in item:
                    errors.append(f"Batch item {i} missing 'name' field.")
                elif "matrix" not in item:
                    errors.append(f"Batch item {i} missing 'matrix' field.")

    return errors


def create_example_config(output_path: str = "hlplot_config.yaml") -> str:
    """
    Create an example configuration file.

    Parameters
    ----------
    output_path : str
        Path to write the example configuration file.

    Returns
    -------
    str
        Path to the created file.
    """
    example_config = """# HarrisLabPlotting Configuration File
# =====================================
# This file configures the hlplot command-line tool.
# All paths can be absolute or relative to the config file location.

# Input files (required for plotting)
mesh_file: "data/brain.gii"
roi_coords_file: "data/roi_coordinates.csv"
connectivity_matrix: "data/connectivity_matrix.npy"

# Output settings
output_dir: "./outputs"
output_format: "html"  # Options: html, png, pdf, svg, jpeg, webp

# Plot settings
plot:
  title: "Brain Connectivity Visualization"
  node_size: 10                    # Size of ROI nodes
  node_color: "purple"             # Color or path to color vector file
  edge_threshold: 0.1              # Minimum edge weight to display
  edge_width_range: [1, 5]         # Min and max edge widths
  opacity: 0.3                     # Brain mesh opacity (0-1)

# Camera settings
camera:
  view: "anterior"  # Options: anterior, posterior, lateral-left, lateral-right, superior, inferior
  # Or specify custom camera position:
  # eye: [0, 2, 0]
  # center: [0, 0, 0]
  # up: [0, 0, 1]

# Modularity settings (optional)
modularity:
  enabled: false
  module_file: "data/module_assignments.csv"
  edge_color_mode: "module"        # Options: module, sign
  q_score: null                    # Modularity Q score (for title)
  z_score: null                    # Z-rand score (for title)

# Batch processing (optional)
# Uncomment to process multiple subjects:
# batch:
#   - name: "subject_01"
#     matrix: "data/sub01_connectivity.npy"
#     modules: "data/sub01_modules.csv"
#   - name: "subject_02"
#     matrix: "data/sub02_connectivity.npy"
#     modules: "data/sub02_modules.csv"
"""

    output_path = Path(output_path)
    with open(output_path, "w") as f:
        f.write(example_config)

    return str(output_path)


def resolve_paths(config: Dict[str, Any], base_dir: Optional[str] = None) -> Dict[str, Any]:
    """
    Resolve relative paths in config to absolute paths.

    Parameters
    ----------
    config : Dict[str, Any]
        Configuration dictionary.
    base_dir : str, optional
        Base directory for resolving relative paths.
        If None, uses current working directory.

    Returns
    -------
    Dict[str, Any]
        Configuration with resolved paths.
    """
    if base_dir is None:
        base_dir = os.getcwd()

    base_path = Path(base_dir)
    result = config.copy()

    # Resolve top-level file paths
    path_fields = ["mesh_file", "roi_coords_file", "connectivity_matrix", "output_dir"]
    for field in path_fields:
        if field in result and result[field]:
            field_path = Path(result[field])
            if not field_path.is_absolute():
                result[field] = str(base_path / field_path)

    # Resolve modularity paths
    if "modularity" in result and "module_file" in result["modularity"]:
        mod_path = Path(result["modularity"]["module_file"])
        if not mod_path.is_absolute():
            result["modularity"]["module_file"] = str(base_path / mod_path)

    # Resolve batch paths
    if "batch" in result:
        for item in result["batch"]:
            for field in ["matrix", "modules"]:
                if field in item:
                    item_path = Path(item[field])
                    if not item_path.is_absolute():
                        item[field] = str(base_path / item_path)

    return result
