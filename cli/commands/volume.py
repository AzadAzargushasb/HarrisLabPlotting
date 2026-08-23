"""
hlplot volume -- render statistical volumes (voxel maps) on a glass brain.
"""
import click
from pathlib import Path

from ..console import (console, print_success, print_error, print_warning,
                       print_info)
from ..pager import PagerCommand


def _parse_pair(value, name):
    """'a,b' -> (float, float)."""
    if value is None:
        return None
    parts = [p.strip() for p in str(value).replace(",", " ").split()]
    if len(parts) != 2:
        raise click.BadParameter(f"{name} must be 'LOW,HIGH', got {value!r}")
    return (float(parts[0]), float(parts[1]))


def _parse_size(value, name, default):
    if value is None:
        return default
    parts = [p.strip() for p in str(value).replace(",", " ").split()]
    if len(parts) != 2:
        raise click.BadParameter(f"{name} must be 'WIDTH,HEIGHT', got {value!r}")
    return (int(parts[0]), int(parts[1]))


def _per_map(values, n, name):
    """Spread a repeatable option over n maps.

    Given once, it applies to every map; given n times, positionally. Anything
    else is an error rather than a silent mismatch.
    """
    if not values:
        return [None] * n
    if len(values) == 1:
        return [values[0]] * n
    if len(values) != n:
        raise click.BadParameter(
            f"{name} was given {len(values)} times but there are {n} "
            f"--volume entries. Give it once (applies to all) or once per map."
        )
    return list(values)


@click.command(cls=PagerCommand)
# === Required inputs ===
@click.option("--mesh", "-m", required=True, type=click.Path(exists=True),
              help="Brain mesh file (.gii, .obj, .mz3, .ply). Its vertices must "
                   "be in the SAME world space as the volume(s).")
@click.option("--volume", "-v", "volume_paths", multiple=True,
              type=click.Path(exists=True),
              help="Statistical volume to render (.nii/.nii.gz). Repeat for "
                   "several maps: --volume pos.nii.gz --volume neg.nii.gz")
@click.option("--volume-spec", type=click.Path(exists=True),
              help="YAML file describing the maps (a top-level 'volumes:' "
                   "list). PRECEDENCE: a CLI flag overrides the same key for "
                   "EVERY map in the file, which overrides the built-in "
                   "default.")
@click.option("--output", "-o", default="brain_volume.html",
              help="Output HTML path. Default: brain_volume.html")

# === Per-map appearance (repeatable, matched by position) ===
@click.option("--volume-name", "volume_names", multiple=True,
              help="Label for the map, used on its colorbar. Repeat per map. "
                   "Default: the filename stem.")
@click.option("--volume-cmap", "volume_cmaps", multiple=True,
              help="Colorscale. Built in: 'hot32' (matplotlib hot truncated at "
                   "0.32, the activation default) and 'ice28' (the custom ice "
                   "at 0.28, for deactivation), plus their '_light' variants; "
                   "any plotly scale name also works ('Viridis', 'Hot'). "
                   "On a light background hot32/ice28 auto-switch to the "
                   "_light variant, whose top is truncated so the peak does "
                   "not vanish into the page.")
@click.option("--volume-cmap-no-adapt", is_flag=True, default=False,
              help="Never auto-switch the colorscale for a light background.")

# === Thresholds (pick ONE per map) ===
@click.option("--volume-threshold", "volume_thresholds", multiple=True, type=float,
              help="Threshold as an ABSOLUTE value in the map's own units "
                   "(e.g. 3.1 for a z-map). Units: same as the data.")
@click.option("--volume-top-percent", "volume_top_percents", multiple=True, type=float,
              help="Keep only the strongest N%% of suprathreshold voxels "
                   "(quantile(v, 1-N/100)). Units: percent, 0-100.")
@click.option("--volume-percentile", "volume_percentiles", multiple=True, type=float,
              help="Threshold at the Nth percentile of nonzero magnitudes. "
                   "Units: percent, 0-100. Useful across maps in different "
                   "units.")
@click.option("--volume-range", "volume_ranges", multiple=True,
              help="Explicit COLOUR range as 'LOW,HIGH' in the map's units, "
                   "when the automatic range (threshold to the 99.5th "
                   "percentile) does not suit your data.")

# === Data preparation ===
@click.option("--volume-smooth-fwhm", "volume_smooths", multiple=True,
              help="Gaussian blur width as FWHM in MILLIMETRES: one number for "
                   "all axes, or 'X,Y,Z' per axis. RECOMMENDED: pass the voxel "
                   "size of the ORIGINAL, pre-warp volume (e.g. "
                   "'0.54,0.11,0.11'), so each axis is blurred by about one "
                   "original voxel -- this is what removes the stair-steps left "
                   "by resampling a few thick slices onto a fine grid. "
                   "'auto' probes the data but under-estimates. Default: none.")
@click.option("--volume-level", "volume_levels", multiple=True,
              type=click.Choice(["preserve", "fixed"], case_sensitive=False),
              help="Where the cloud's outer boundary sits after smoothing. "
                   "'preserve' (default) picks the level enclosing the same "
                   "voxel count as the unsmoothed suprathreshold region, "
                   "because blurring lowers the peak and a fixed level eats "
                   "the cluster (up to 40%% of it). 'fixed' draws at the "
                   "literal threshold.")
@click.option("--volume-crop/--no-volume-crop", default=True,
              help="Crop to the suprathreshold bounding box (+6 voxels) before "
                   "rendering. Discards empty space only -- the picture is "
                   "identical, but far fewer voxels reach the browser. "
                   "Default: on.")
@click.option("--volume-clamp-negative", is_flag=True, default=False,
              help="Set negative values to 0. Use on a map that is "
                   "non-negative by construction but was resampled with "
                   "'-interp spline', which overshoots.")

# === The look ===
@click.option("--volume-opacity", "volume_opacities", multiple=True, type=float,
              help="Opacity CEILING of the VOXEL MAP at its peak, 0-1. This is "
                   "the map, NOT the brain (that is --ghost-opacity). Higher = "
                   "more solid voxels. Default 1.0.")
@click.option("--volume-opacity-floor", "volume_floors", multiple=True, type=float,
              help="Opacity at the THRESHOLD, 0-1. Without a floor the ramp "
                   "starts at 0 and voxels at the threshold are invisible "
                   "(~14%% of a typical map). Default 0.15. Set 0 for the "
                   "un-floored ramp.")
@click.option("--volume-gamma", "volume_gammas", multiple=True, type=float,
              help="Shape of the opacity ramp between floor and ceiling. "
                   "LOW (<1) lights up the whole cluster; HIGH (>1) leaves "
                   "only the core. Unitless exponent. Default 1.0 (linear).")
@click.option("--volume-surfaces", "volume_surfaces", multiple=True, type=int,
              help="Number of internal shells the ray-cast steps through. More "
                   "= smoother cloud and SLOWER. Default 200, which is also the "
                   "ceiling. If a render is taking too long, LOWER THIS FIRST "
                   "(100 is about twice as fast and hard to tell apart).")
@click.option("--glass/--no-glass", default=True,
              help="Draw the brain as a translucent shell. Default: on.")
@click.option("--ghost-opacity", default=0.04, type=float,
              help="Opacity of the BRAIN shell, 0-1. Default 0.04. Not the "
                   "voxel map's opacity -- that is --volume-opacity.")
@click.option("--mesh-color", default="#8e8e9a",
              help="Colour of the brain shell. Default: #8e8e9a")

# === Grid size / cost ===
@click.option("--volume-step", "volume_steps", multiple=True, type=int,
              help="Take every Nth voxel along each axis. NOTHING is "
                   "downsampled by default; the projected HTML size and render "
                   "time are printed so a large grid is a choice. With a "
                   "0.54 mm smoothing kernel a step of 5-7 is visually "
                   "identical to full resolution and vastly cheaper.")
@click.option("--volume-max-voxels", "volume_max_voxels", multiple=True, type=int,
              help="Pick the smallest step whose grid fits in this many "
                   "voxels. An alternative to --volume-step when you care "
                   "about the budget rather than the factor.")

# === Camera / export (mirrors hlplot plot) ===
@click.option("--camera", default="oblique",
              type=click.Choice(["oblique", "anterior", "posterior", "left",
                                 "right", "superior", "inferior",
                                 "anterolateral_left", "anterolateral_right",
                                 "posterolateral_left", "posterolateral_right"],
                                case_sensitive=False),
              help="Camera preset. Default: oblique")
@click.option("--zoom", default=1.0, type=float,
              help="Camera zoom multiplier. >1 brings the camera closer. "
                   "Default 1.0")
@click.option("--title", "-t", default="", help="Plot title. Default: none.")
@click.option("--background-color", default="white",
              help="Figure background: a colour name, hex, or 'transparent'. "
                   "Default: white. On a light background hot32/ice28 "
                   "auto-truncate their top so the peak stays visible.")
@click.option("--export-image", type=click.Path(),
              help="Also write a static image (.png/.svg/.pdf).")
@click.option("--image-dpi", default=300, type=int,
              help="DPI for the static export. Default: 300")
@click.option("--export-size", default="1200,1200",
              help="Export canvas as 'width,height'. Default: '1200,1200'")
@click.option("--export-no-title", is_flag=True, default=False,
              help="Exclude the title from the exported image.")
@click.option("--export-no-legend", is_flag=True, default=False,
              help="Exclude the colorbars from the exported image.")
@click.option("--multi-view", default=None,
              help="Comma-separated camera views to render and stitch into one "
                   "PNG, e.g. 'left,superior,posterior'. --export-image then "
                   "names the stitched strip.")
@click.option("--multi-view-panel-size", default="800,800",
              help="Per-panel pixel size for --multi-view. Default '800,800'")
@click.option("--multi-view-grid", default=None,
              help="Lay the panels out as 'ROWS,COLS' instead of one row.")
@click.option("--no-html", is_flag=True, default=False,
              help="Skip the interactive HTML and write only --export-image. "
                   "RECOMMENDED for voxel figures: go.Volume ships the whole "
                   "grid to the browser, so the HTML is far larger than the "
                   "PNG.")
@click.option("--no-space-check", is_flag=True, default=False,
              help="Skip the map-vs-mesh space check. The check is on by "
                   "default because an overlay silently landing off the brain "
                   "is the most common volume failure.")
def volume(mesh, volume_paths, volume_spec, output, volume_names, volume_cmaps,
           volume_cmap_no_adapt, volume_thresholds, volume_top_percents,
           volume_percentiles, volume_ranges, volume_smooths, volume_levels,
           volume_crop, volume_clamp_negative, volume_opacities, volume_floors,
           volume_gammas, volume_surfaces, glass, ghost_opacity, mesh_color,
           volume_steps, volume_max_voxels, camera, zoom, title,
           background_color, export_image, image_dpi, export_size,
           export_no_title, export_no_legend, multi_view,
           multi_view_panel_size, multi_view_grid, no_html, no_space_check):
    """
    Render statistical volumes (voxel maps) inside a glass brain.

    \b
    The map and the mesh must be in the SAME world space. Check with
      hlplot utils check-alignment --volume map.nii.gz --mesh brain.obj
    and see tutorial/VOXEL_PLOTTING.md for the FLIRT commands.

    \b
    Examples:
      # one map, defaults
      hlplot volume --mesh brain.obj --volume zmap.nii.gz -o out.html

      # activation + deactivation, the paper's hot/ice convention
      hlplot volume --mesh brain.obj \\
        --volume pos_z.nii.gz --volume-cmap hot32 --volume-name Activation \\
        --volume neg_z.nii.gz --volume-cmap ice28 --volume-name Deactivation \\
        --volume-threshold 3.1 --volume-smooth-fwhm "0.54,0.11,0.11" \\
        --volume-step 7 --multi-view "left,superior,posterior" \\
        --no-html --export-image fig.png

      # from a spec file, with one flag overriding every map in it
      hlplot volume --mesh brain.obj --volume-spec maps.yaml \\
        --volume-threshold 4.0
    """
    try:
        from HarrisLabPlotting.volume import (
            create_brain_volume_plot, load_volume_spec, normalize_volume_specs,
        )

        # ---- assemble the per-map specs ------------------------------
        if volume_spec:
            specs = load_volume_spec(volume_spec)
            if volume_paths:
                specs += [{"path": p} for p in volume_paths]
        elif volume_paths:
            specs = [{"path": p} for p in volume_paths]
        else:
            print_error("No volumes given. Use --volume FILE (repeatable) "
                        "and/or --volume-spec FILE.yaml")
            raise click.Abort()

        n = len(specs)
        per = {
            "name": _per_map(volume_names, n, "--volume-name"),
            "cmap": _per_map(volume_cmaps, n, "--volume-cmap"),
            "threshold": _per_map(volume_thresholds, n, "--volume-threshold"),
            "top_percent": _per_map(volume_top_percents, n, "--volume-top-percent"),
            "percentile": _per_map(volume_percentiles, n, "--volume-percentile"),
            "range": _per_map(volume_ranges, n, "--volume-range"),
            "smooth_fwhm": _per_map(volume_smooths, n, "--volume-smooth-fwhm"),
            "level": _per_map(volume_levels, n, "--volume-level"),
            "opacity": _per_map(volume_opacities, n, "--volume-opacity"),
            "opacity_floor": _per_map(volume_floors, n, "--volume-opacity-floor"),
            "gamma": _per_map(volume_gammas, n, "--volume-gamma"),
            "surfaces": _per_map(volume_surfaces, n, "--volume-surfaces"),
            "step": _per_map(volume_steps, n, "--volume-step"),
            "max_voxels": _per_map(volume_max_voxels, n, "--volume-max-voxels"),
        }
        for i, s in enumerate(specs):
            for key, vals in per.items():
                if vals[i] is not None:          # CLI flag beats the spec file
                    s[key] = vals[i]
            s.setdefault("crop", volume_crop)
            if volume_clamp_negative:
                s["clamp_negative"] = True

        for s in specs:
            given = [k for k in ("threshold", "top_percent", "percentile")
                     if s.get(k) is not None]
            if len(given) > 1:
                print_warning(
                    f"{Path(s['path']).name}: {len(given)} threshold modes "
                    f"given ({', '.join(given)}). Use exactly one."
                )

        mv = [v.strip() for v in multi_view.split(",")] if multi_view else None
        mvg = _parse_size(multi_view_grid, "--multi-view-grid", None) if multi_view_grid else None

        print_info(f"Rendering {n} volume(s) on {Path(mesh).name}...")
        fig, info = create_brain_volume_plot(
            mesh=mesh, volumes=specs,
            plot_title=title, save_path=output,
            background_color=background_color,
            glass=glass, ghost_opacity=ghost_opacity, mesh_color=mesh_color,
            camera_view=camera, zoom=zoom,
            adapt_cmap=not volume_cmap_no_adapt,
            export_image=export_image, image_dpi=image_dpi,
            export_size=_parse_size(export_size, "--export-size", (1200, 1200)),
            export_show_title=not export_no_title,
            export_show_legend=not export_no_legend,
            multi_view=mv,
            multi_view_panel_size=_parse_size(
                multi_view_panel_size, "--multi-view-panel-size", (800, 800)),
            multi_view_grid=mvg,
            no_html=no_html, check_space=not no_space_check,
        )

        if not no_html:
            print_success(f"Saved interactive visualization to {output}")
        if export_image:
            print_success(f"Exported static image to {export_image}")

        from ..console import create_stats_table
        stats = {}
        for v in info["volumes"]:
            stats[v["name"]] = (f"level {v['level']:.2f}, "
                                f"{v['n_voxels']:,} voxels rendered")
        console.print()
        console.print(create_stats_table(stats, title="Volumes"))

    except click.Abort:
        raise
    except Exception as e:
        print_error(f"Error creating volume plot: {e}")
        raise click.Abort()
