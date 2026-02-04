"""
Main CLI entry point for HarrisLabPlotting.
"""

import click
from pathlib import Path

from .console import console, print_version_info, print_header
from .commands import plot, modular, batch, coords, utils, config


@click.group(invoke_without_command=True)
@click.option("--version", "-v", is_flag=True, help="Show version information.")
@click.pass_context
def cli(ctx, version):
    """
    HarrisLabPlotting - Brain connectivity and modularity visualization.

    A command-line tool for creating brain connectivity plots,
    modularity visualizations, and processing neuroimaging data.

    \b
    Quick Start:
      hlplot plot --mesh brain.gii --coords rois.csv --matrix data.npy
      hlplot modular --mesh brain.gii --coords rois.csv --matrix data.npy --modules mod.csv

    \b
    Use 'hlplot <command> --help' for more information on each command.
    """
    if version:
        from HarrisLabPlotting import __version__
        print_version_info(__version__)
        ctx.exit(0)

    if ctx.invoked_subcommand is None:
        click.echo(ctx.get_help())


# Register subcommands
cli.add_command(plot.plot)
cli.add_command(modular.modular)
cli.add_command(batch.batch)
cli.add_command(coords.coords)
cli.add_command(utils.utils)
cli.add_command(config.config)


if __name__ == "__main__":
    cli()
