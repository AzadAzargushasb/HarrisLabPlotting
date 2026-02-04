"""
Console utilities for HarrisLabPlotting CLI.
Uses Rich library for formatted terminal output.
"""

import sys
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
from rich.syntax import Syntax
from rich import box
from typing import Optional, List, Dict, Any

# Global console instance with Windows-safe settings
# Use legacy_windows=False and safe_box for better compatibility
console = Console(legacy_windows=False)


def print_success(message: str) -> None:
    """Print a success message in green."""
    console.print(f"[bold green][OK][/bold green] {message}")


def print_error(message: str) -> None:
    """Print an error message in red."""
    console.print(f"[bold red][ERROR][/bold red] {message}")


def print_warning(message: str) -> None:
    """Print a warning message in yellow."""
    console.print(f"[bold yellow][WARN][/bold yellow] {message}")


def print_info(message: str) -> None:
    """Print an info message in blue."""
    console.print(f"[bold blue][INFO][/bold blue] {message}")


def print_header(title: str) -> None:
    """Print a styled header."""
    console.print()
    console.print(Panel(title, style="bold cyan", box=box.DOUBLE))
    console.print()


def print_file_info(file_path: str, file_type: str) -> None:
    """Print information about a file."""
    console.print(f"  [dim]{file_type}:[/dim] [cyan]{file_path}[/cyan]")


def create_stats_table(stats: Dict[str, Any], title: str = "Statistics") -> Table:
    """Create a formatted table for displaying statistics."""
    table = Table(title=title, box=box.ROUNDED)
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")

    for key, value in stats.items():
        if isinstance(value, float):
            table.add_row(key, f"{value:.4f}")
        else:
            table.add_row(key, str(value))

    return table


def create_file_table(files: List[Dict[str, str]], title: str = "Files") -> Table:
    """Create a formatted table for displaying file information."""
    table = Table(title=title, box=box.ROUNDED)
    table.add_column("Type", style="cyan")
    table.add_column("Path", style="green")
    table.add_column("Status", style="yellow")

    for file_info in files:
        table.add_row(
            file_info.get("type", ""),
            file_info.get("path", ""),
            file_info.get("status", "")
        )

    return table


def create_progress_context(description: str = "Processing..."):
    """Create a progress context for long-running operations."""
    return Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        console=console,
    )


def print_config_summary(config: Dict[str, Any]) -> None:
    """Print a summary of the configuration."""
    console.print()
    console.print("[bold]Configuration Summary:[/bold]")
    console.print()

    # Input files
    if any(key in config for key in ["mesh_file", "roi_coords_file", "connectivity_matrix"]):
        console.print("[bold cyan]Input Files:[/bold cyan]")
        if "mesh_file" in config:
            print_file_info(config["mesh_file"], "Mesh")
        if "roi_coords_file" in config:
            print_file_info(config["roi_coords_file"], "ROI Coords")
        if "connectivity_matrix" in config:
            print_file_info(config["connectivity_matrix"], "Matrix")
        console.print()

    # Output settings
    if any(key in config for key in ["output_dir", "output_format"]):
        console.print("[bold cyan]Output Settings:[/bold cyan]")
        if "output_dir" in config:
            print_file_info(config["output_dir"], "Output Dir")
        if "output_format" in config:
            console.print(f"  [dim]Format:[/dim] [cyan]{config['output_format']}[/cyan]")
        console.print()

    # Plot settings
    if "plot" in config:
        console.print("[bold cyan]Plot Settings:[/bold cyan]")
        plot_config = config["plot"]
        for key, value in plot_config.items():
            console.print(f"  [dim]{key}:[/dim] [cyan]{value}[/cyan]")
        console.print()


def print_help_panel(command: str, description: str, options: List[Dict[str, str]]) -> None:
    """Print a help panel for a command."""
    content = f"[bold]{description}[/bold]\n\n"
    content += "[bold cyan]Options:[/bold cyan]\n"

    for opt in options:
        content += f"  [green]{opt['name']}[/green]: {opt['description']}\n"

    console.print(Panel(content, title=f"hlplot {command}", box=box.ROUNDED))


def print_version_info(version: str) -> None:
    """Print version information."""
    console.print()
    console.print(Panel(
        f"[bold cyan]HarrisLabPlotting[/bold cyan] v{version}\n"
        "[dim]Brain connectivity and modularity visualization tools[/dim]",
        box=box.DOUBLE
    ))
    console.print()
