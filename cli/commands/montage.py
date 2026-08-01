"""
Compose several already-rendered PNGs into one labeled grid image.

`hlplot montage` is the CLI face of ``HarrisLabPlotting.compose_image_grid``. It
does NOT render brains itself — you render each panel first (with `hlplot plot` /
`hlplot modular`, or any other tool), then stitch the panels into a grid with
optional column headers, row labels, per-cell labels, and a title.

This is the tool for a **multi-mesh / cross-species** figure: `--multi-view` and
`--multi-view-grid` on `plot`/`modular` show ONE mesh from several cameras, but a
grid whose columns are DIFFERENT meshes (human / rat / macaque) needs each panel
rendered separately and then composed here.

Example (2x3, columns = species, cells = views):

\b
  hlplot montage \\
    --images "h_left.png,r_sup.png,m_right.png,h_ant.png,r_inf.png,m_post.png" \\
    --grid "2,3" \\
    --col-labels "Human,Rat,Macaque" \\
    --panel-labels "Left,Superior,Right,Anterior,Inferior,Posterior" \\
    --output species_grid.png
"""

import click
from pathlib import Path

from ..console import print_success, print_error, print_info
from ..pager import PagerCommand


def _split_csv(value):
    """Split a comma-separated CLI string into a stripped list (or None)."""
    if value is None:
        return None
    return [part.strip() for part in value.split(",")]


@click.command(cls=PagerCommand)
@click.option("--images", "-i", required=True, type=str,
              help="Comma-separated list of panel PNG paths, in ROW-MAJOR order "
                   "(left-to-right, then top-to-bottom).")
@click.option("--output", "-o", required=True, type=click.Path(),
              help="Output PNG path for the composed grid.")
@click.option("--grid", "grid_spec", default=None, type=str,
              help="Grid shape as 'rows,cols' (e.g. '2,3'). Omit for a single row "
                   "(1 x N). rows*cols must be >= the number of images.")
@click.option("--col-labels", default=None, type=str,
              help="Comma-separated column headers, one per column, drawn once "
                   "along the top (e.g. 'Human,Rat,Macaque').")
@click.option("--row-labels", default=None, type=str,
              help="Comma-separated row labels, one per row, drawn in a left gutter.")
@click.option("--panel-labels", default=None, type=str,
              help="Comma-separated per-cell labels, one per image, drawn below "
                   "each panel (e.g. 'Left,Superior,Right,...').")
@click.option("--title", default="", type=str,
              help="Combined title drawn above the whole grid.")
@click.option("--background-color", default="white", type=str,
              help="Canvas background: a named color, a hex code like '#1e1e1e', or "
                   "'transparent' for an RGBA output. Default: white.")
@click.option("--label-font-size", default=18, type=int,
              help="Base font size (px) for the per-cell labels. Default: 18.")
@click.option("--title-font-size", default=22, type=int,
              help="Base font size (px) for the combined title. Default: 22.")
@click.option("--header-font-size", default=24, type=int,
              help="Base font size (px) for column / row headers. Default: 24.")
@click.option("--no-autocrop", is_flag=True, default=False,
              help="Do not trim each panel's background border before compositing.")
@click.option("--autocrop-padding", default=8, type=int,
              help="Pixels of padding left around each cropped panel. Default: 8.")
def montage(images, output, grid_spec, col_labels, row_labels, panel_labels,
            title, background_color, label_font_size, title_font_size,
            header_font_size, no_autocrop, autocrop_padding):
    """Compose pre-rendered PNGs into one labeled grid (e.g. a species montage)."""
    try:
        from HarrisLabPlotting import compose_image_grid

        image_paths = _split_csv(images)
        if not image_paths:
            raise click.UsageError("--images must list at least one PNG path.")
        for p in image_paths:
            if not Path(p).exists():
                raise click.BadParameter(f"--images: file not found: {p}")

        grid = None
        if grid_spec:
            parts = [p.strip() for p in grid_spec.split(",")]
            if len(parts) != 2:
                raise click.UsageError("--grid must be 'rows,cols' (e.g. '2,3').")
            try:
                grid = (int(parts[0]), int(parts[1]))
            except ValueError:
                raise click.UsageError("--grid rows and cols must be integers.")
            if grid[0] < 1 or grid[1] < 1:
                raise click.UsageError("--grid rows and cols must both be >= 1.")
            if grid[0] * grid[1] < len(image_paths):
                raise click.UsageError(
                    f"--grid {grid_spec!r} has only {grid[0] * grid[1]} cells but "
                    f"{len(image_paths)} images were given."
                )

        cols = _split_csv(col_labels)
        rows = _split_csv(row_labels)
        panels = _split_csv(panel_labels)

        print_info(f"Composing {len(image_paths)} panel(s) into {output} ...")
        out = compose_image_grid(
            image_paths,
            output,
            grid=grid,
            col_labels=cols,
            row_labels=rows,
            panel_labels=panels,
            title=title,
            background_color=background_color,
            label_font_size=label_font_size,
            title_font_size=title_font_size,
            header_font_size=header_font_size,
            autocrop=not no_autocrop,
            autocrop_padding_px=autocrop_padding,
        )
        print_success(f"Wrote {out}")
    except click.ClickException:
        raise
    except Exception as e:
        print_error(f"Error composing montage: {e}")
        raise click.Abort()
