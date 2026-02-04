"""
HarrisLabPlotting CLI
=====================
Command-line interface for brain connectivity and modularity visualization.

Main entry point: hlplot
"""

from .main import cli
from .console import console, print_success, print_error, print_warning, print_info
from .config_loader import load_config, validate_config, ConfigError

__all__ = [
    "cli",
    "console",
    "print_success",
    "print_error",
    "print_warning",
    "print_info",
    "load_config",
    "validate_config",
    "ConfigError",
]
