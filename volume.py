"""
Volumetric (voxel-map) plotting
===============================

Renders a statistical volume -- a z-map, t-map, beta map -- as a soft
ray-cast cloud inside a translucent "glass" brain, using plotly's
``go.Volume``.

Why ``go.Volume`` and not isosurfaces
-------------------------------------
An isosurface is a hard shell at one value; stacking several reads as nested
plastic. ``go.Volume`` steps a ray through ``surface_count`` internal shells and
applies an ``opacityscale``, so opacity ramps with intensity and the cluster
fringe fades instead of ending in an edge. That is what makes it look like
MRIcroGL / Surfice rather than a stack of blobs.

Coordinates
-----------
Everything ends up in **world millimetres**, the same space mesh vertices live
in, so an overlay lands on the brain with no extra registration -- provided the
map and the mesh really are in the same space. Use
``hlplot utils check-alignment`` to confirm, and see the voxel-plotting tutorial
for the FLIRT commands.

Cost
----
``go.Volume`` ships x, y, z **and** value as four full arrays into the browser,
so memory and file size scale with the voxel count, not with the number of
shells. Measured: 56k voxels -> ~18 MB HTML, ~27 s per panel; 127k -> ~22 MB,
~42 s; 439k -> ~42 MB, ~107 s. Nothing is downsampled unless you ask, but the
projected cost is always printed first.
"""
from __future__ import annotations

import sys
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

__all__ = [
    "COLORSCALES",
    "get_colorscale",
    "signed_colorscale",
    "load_volume_map",
    "crop_to_support",
    "resolve_threshold",
    "resolve_smoothing_fwhm",
    "smooth_volume",
    "volume_preserving_level",
    "axis_aligned_world_grid",
    "choose_grid_step",
    "project_render_cost",
    "build_volume_traces",
    "load_volume_spec",
    "normalize_volume_specs",
    "create_brain_volume_plot",
    "VOLUME_DEFAULTS",
]

FWHM_TO_SIGMA = 1.0 / (2.0 * np.sqrt(2.0 * np.log(2.0)))   # 0.42466

VOLUME_DEFAULTS = dict(
    opacity=1.0,          # ceiling of the VOXEL MAP's opacity ramp (not the brain)
    opacity_floor=0.15,   # opacity at the threshold, so the fringe stays visible
    gamma=1.0,            # shape of the ramp between floor and ceiling
    surfaces=200,         # shells the ray-cast steps through (ceiling)
    ghost_opacity=0.04,   # the brain shell
    crop_margin=6,        # voxels of padding around the suprathreshold bbox
)
MAX_SURFACES = 200


# --------------------------------------------------------------------------
# colorscales
# --------------------------------------------------------------------------
# Sampled from the study's own colormaps: matplotlib `hot` truncated at 0.32
# (CMAP_LO_POS) and a custom `ice` truncated at 0.28 (CMAP_LO_NEG), so the 3-D
# figures match the 2-D coronal montages.
#
# The *_light variants also truncate the TOP at 0.85. `hot` runs all the way to
# pure #ffffff, so on a white page the highest-z core would be invisible.
COLORSCALES: Dict[str, List] = {
    "hot32": [
        [0.0, '#df0000'], [0.0769, '#ff0500'], [0.1538, '#ff2700'],
        [0.2308, '#ff4c00'], [0.3077, '#ff6e00'], [0.3846, '#ff9000'],
        [0.4615, '#ffb500'], [0.5385, '#ffd700'], [0.6154, '#fffc00'],
        [0.6923, '#ffff2e'], [0.7692, '#ffff61'], [0.8462, '#ffff99'],
        [0.9231, '#ffffcc'], [1.0, '#ffffff'],
    ],
    "hot32_light": [
        [0.0, '#df0000'], [0.0769, '#fc0000'], [0.1538, '#ff1700'],
        [0.2308, '#ff3400'], [0.3077, '#ff4f00'], [0.3846, '#ff6b00'],
        [0.4615, '#ff8600'], [0.5385, '#ffa000'], [0.6154, '#ffbd00'],
        [0.6923, '#ffd700'], [0.7692, '#fff400'], [0.8462, '#ffff17'],
        [0.9231, '#ffff42'], [1.0, '#ffff69'],
    ],
    "ice28": [
        [0.0, '#2579ba'], [0.0769, '#2c88c5'], [0.1538, '#3398d0'],
        [0.2308, '#3aa7da'], [0.3077, '#42b6e4'], [0.3846, '#58c0ea'],
        [0.4615, '#6ec9f0'], [0.5385, '#84d3f6'], [0.6154, '#9cddfc'],
        [0.6923, '#afe4ff'], [0.7692, '#bfeaff'], [0.8462, '#cfefff'],
        [0.9231, '#e0f5ff'], [1.0, '#f0faff'],
    ],
    "ice28_light": [
        [0.0, '#2579ba'], [0.0769, '#2a85c2'], [0.1538, '#3092cb'],
        [0.2308, '#369ed3'], [0.3077, '#3baadc'], [0.3846, '#41b5e4'],
        [0.4615, '#53bee9'], [0.5385, '#65c5ee'], [0.6154, '#76cdf2'],
        [0.6923, '#88d4f7'], [0.7692, '#99dbfc'], [0.8462, '#aae2ff'],
        [0.9231, '#b7e7ff'], [1.0, '#c4ebff'],
    ],
}


def _is_light(background: str) -> bool:
    """True when a background colour is light enough to swallow a white peak."""
    bg = str(background or "white").strip().lower()
    if bg in ("transparent", "none"):
        return True                       # composited onto paper, assume light
    named_dark = {"black", "#000", "#000000"}
    if bg in named_dark:
        return False
    if bg.startswith("#") and len(bg) in (4, 7):
        h = bg[1:]
        if len(h) == 3:
            h = "".join(c * 2 for c in h)
        r, g, b = (int(h[i:i + 2], 16) for i in (0, 2, 4))
        return (0.299 * r + 0.587 * g + 0.114 * b) > 128
    return bg not in ("black",)


def get_colorscale(name: str, background: str = "white", adapt: bool = True):
    """Resolve a colorscale name, adapting the top end to the background.

    Parameters
    ----------
    name : str
        ``'hot32'`` / ``'ice28'`` (the defaults), one of their ``_light``
        variants, or any plotly colorscale name (``'Viridis'``, ``'Hot'``, ...).
    background : str
        Figure background. When it is light and ``adapt`` is True, ``hot32`` and
        ``ice28`` switch to their ``_light`` variants, whose top is truncated so
        the peak does not disappear into the page.
    adapt : bool
        Set False to use the named scale exactly as given.
    """
    key = str(name)
    if adapt and _is_light(background) and key in ("hot32", "ice28"):
        return COLORSCALES[key + "_light"], key + "_light"
    if key in COLORSCALES:
        return COLORSCALES[key], key
    return key, key            # a plotly built-in name; plotly validates it


def signed_colorscale(background: str = "white", adapt: bool = True):
    """Diverging scale for a single signed file.

    Brightest ice at the most negative, brightest hot at the most positive,
    meeting in the middle. The middle is never drawn -- values below threshold
    are NaN -- so the discontinuity at 0.5 is invisible and each half keeps its
    own full ramp.
    """
    ice, _ = get_colorscale("ice28", background, adapt)
    hot, _ = get_colorscale("hot32", background, adapt)
    out = [[0.5 * (1.0 - p), c] for p, c in reversed(ice)]
    out += [[0.5 + 0.5 * p, c] for p, c in hot]
    fixed, last = [], -1.0
    for p, c in out:
        p = max(float(p), last)
        fixed.append([round(min(p, 1.0), 6), c])
        last = p
    fixed[0][0], fixed[-1][0] = 0.0, 1.0
    return fixed


# --------------------------------------------------------------------------
# loading, cropping, thresholds
# --------------------------------------------------------------------------
def load_volume_map(path, clamp_negative: bool = False):
    """Load a NIfTI statistical map.

    Parameters
    ----------
    path : str or Path or nibabel image
    clamp_negative : bool
        Set negatives to 0. Use it on a map that is non-negative by
        construction but was resampled with ``-interp spline``, which
        overshoots: on the Fig 1 positive map spline produced values down to
        -5.87 and 22 M spurious negative voxels.

    Returns
    -------
    (data, affine, zooms) : (ndarray float32, ndarray 4x4, ndarray (3,))
    """
    import nibabel as nib
    img = path if hasattr(path, "affine") else nib.load(str(path))
    data = np.asarray(img.dataobj, dtype=np.float32)
    if data.ndim == 4 and data.shape[3] == 1:
        data = data[..., 0]
    if data.ndim != 3:
        raise ValueError(f"expected a 3-D volume, got shape {data.shape}")
    if clamp_negative:
        data = np.maximum(data, 0.0)
    return data, np.asarray(img.affine, float), np.array(
        img.header.get_zooms()[:3], float)


def crop_to_support(data, threshold, margin: int = VOLUME_DEFAULTS["crop_margin"]):
    """Crop to the bounding box of ``|data| >= threshold``, plus a margin.

    Discards empty space only -- the rendered picture is identical, but far
    fewer voxels reach the browser. On the Fig 1 map this alone is 77.0M ->
    25.4M voxels. The margin matters: without a zero border the cloud is cut
    flat where a cluster touches the crop edge.

    Returns ``(sub, offset)`` where offset is the voxel index of the crop
    origin, needed to rebuild the affine.
    """
    nz = np.argwhere(np.abs(data) >= threshold)
    if not len(nz):
        return data, np.zeros(3, int)
    lo = np.maximum(nz.min(0) - int(margin), 0)
    hi = np.minimum(nz.max(0) + 1 + int(margin), data.shape)
    return data[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]], lo


def resolve_threshold(data, *, absolute=None, top_percent=None,
                      percentile=None, sign="both", name=""):
    """Decide the display threshold, and report the map's value range.

    Exactly one mode may be given:

    absolute : float
        The value itself, in the map's own units -- ``3.1`` for a z-map.
    top_percent : float
        Keep only the strongest N % of suprathreshold voxels (the
        ``top_decile_mask`` recipe: ``quantile(v, 1 - N/100)``).
    percentile : float
        Threshold at the Nth percentile of all nonzero magnitudes. Useful for
        comparing maps in different units.
    (none)
        Auto: the smallest nonzero magnitude in the file, i.e. "show what is
        already in it" -- right for an FSL map that is already cluster-
        thresholded.

    Returns ``(threshold, report)``; the report is printed to **stderr** so it
    never pollutes a piped figure.
    """
    vals = data[data != 0]
    mag = np.abs(vals)
    if sign == "positive":
        pool = vals[vals > 0]
    elif sign == "negative":
        pool = -vals[vals < 0]
    else:
        pool = mag

    given = [(m, v) for m, v in (("--volume-threshold", absolute),
                                 ("--volume-top-percent", top_percent),
                                 ("--volume-percentile", percentile))
             if v is not None]
    if len(given) > 1:
        raise ValueError(
            "more than one threshold mode given ("
            + ", ".join(m for m, _ in given)
            + "). Pick one: an absolute value, a top-percent, or a percentile."
        )

    if absolute is not None:
        thr, how = float(absolute), f"absolute {absolute:g}"
    elif top_percent is not None:
        thr = float(np.quantile(pool, 1.0 - float(top_percent) / 100.0)) if len(pool) else 0.0
        how = f"top {top_percent:g}% of suprathreshold voxels"
    elif percentile is not None:
        thr = float(np.percentile(mag, float(percentile))) if len(mag) else 0.0
        how = f"{percentile:g}th percentile of nonzero magnitudes"
    else:
        thr = float(mag.min()) if len(mag) else 0.0
        how = "auto (smallest nonzero magnitude in the file)"

    kept = int((pool >= thr).sum())
    report = dict(
        threshold=thr, how=how, n_nonzero=int(len(vals)), n_kept=kept,
        vmin=float(vals.min()) if len(vals) else 0.0,
        vmax=float(vals.max()) if len(vals) else 0.0,
        percentiles={p: float(np.percentile(pool, p)) for p in (50, 75, 90, 95, 99)}
        if len(pool) else {},
        name=name,
    )
    return thr, report


def format_threshold_report(rep) -> str:
    """The per-map block printed to stderr, so an unusual range is obvious."""
    pcts = "  ".join(f"p{p}={v:.2f}" for p, v in rep["percentiles"].items())
    return "\n".join([
        f"  volume{' ' + rep['name'] if rep['name'] else ''}:",
        f"    value range   : {rep['vmin']:.3f} .. {rep['vmax']:.3f} "
        f"({rep['n_nonzero']:,} nonzero voxels)",
        f"    distribution  : {pcts}" if pcts else "    distribution  : (empty)",
        f"    threshold     : {rep['threshold']:.3f}  [{rep['how']}]",
        f"    voxels kept   : {rep['n_kept']:,}",
        "    -> set an explicit range with --volume-range LOW,HIGH if these "
        "values do not follow the usual conventions",
    ])


# --------------------------------------------------------------------------
# smoothing and level
# --------------------------------------------------------------------------
def probe_effective_step(data, threshold, factors=(1, 2, 3, 4, 6, 8, 11, 16, 22, 32),
                         tol=0.01, erode=10):
    """Per-axis "how far can this axis be coarsened before anything changes".

    Decimate one axis, interpolate straight back, measure the error on the
    cluster INTERIOR (the hard zero outside a thresholded cluster is a real
    grid-resolution discontinuity whose error would otherwise swamp the smooth
    interior).

    CAVEAT: this reliably identifies WHICH axis is coarse but under-reports by
    how much -- on the Fig 1 map it returns [4, 2, 2] voxels where the truth is
    about [22, 4, 4], because the data is piecewise-LINEAR along the coarse
    axis and decimation cuts the corner at every knot. Prefer passing the source
    voxel size explicitly.
    """
    from scipy.ndimage import binary_erosion
    interior = binary_erosion(np.abs(data) >= threshold, iterations=int(erode))
    if interior.sum() < 100:
        interior = np.abs(data) >= threshold
    if not interior.any():
        return np.ones(3)
    steps = []
    for axis in range(3):
        n = data.shape[axis]
        best = 1
        for f in factors:
            idx = np.arange(0, n, f)
            if len(idx) < 2:
                break
            coarse = np.take(data, idx, axis=axis)
            full = np.arange(n)
            lo = np.clip(np.searchsorted(idx, full, side="right") - 1, 0, len(idx) - 2)
            w = (full - idx[lo]) / (idx[lo + 1] - idx[lo])
            shape = [1, 1, 1]
            shape[axis] = n
            w = w.reshape(shape)
            rec = (np.take(coarse, lo, axis=axis) * (1 - w)
                   + np.take(coarse, lo + 1, axis=axis) * w)
            den = np.sqrt((data[interior] ** 2).mean())
            err = (np.sqrt(((rec[interior] - data[interior]) ** 2).mean()) / den
                   if den else np.inf)
            if err <= tol:
                best = f
            else:
                break
        steps.append(best)
    return np.array(steps, float)


def resolve_smoothing_fwhm(spec, data, threshold, zooms):
    """Turn a smoothing spec into a per-axis FWHM in millimetres.

    spec may be:

    None
        No smoothing (the default). What is in the file is what is drawn.
    'auto'
        Probe the effective sampling per axis. Finds the coarse axis but
        under-smooths -- see :func:`probe_effective_step`.
    a number
        One FWHM in mm, applied to all three axes.
    (fx, fy, fz)
        A different FWHM per axis, in mm. This is the recommended form: pass
        the voxel size of the ORIGINAL, pre-warp volume, so each axis is
        blurred by about one original voxel.

    Returns ``(fwhm_mm, description)``.
    """
    if spec is None:
        return np.zeros(3), "none"
    if isinstance(spec, str):
        if spec.lower() == "none":
            return np.zeros(3), "none"
        if spec.lower() == "auto":
            steps = probe_effective_step(data, threshold)
            fwhm = steps * zooms
            return fwhm, (f"auto (effective steps {steps.astype(int).tolist()} "
                          f"vox -> {np.round(fwhm, 3).tolist()} mm; auto "
                          f"under-estimates anisotropy -- pass the source "
                          f"voxel size for a better result)")
        spec = [float(x) for x in spec.replace(",", " ").split()]
    arr = np.asarray(spec, float)
    if arr.ndim == 0:
        arr = np.repeat(arr, 3)
    if arr.shape != (3,):
        raise ValueError("smoothing FWHM must be a number or three numbers "
                         f"(mm per axis), got {spec!r}")
    return arr, f"explicit {np.round(arr, 3).tolist()} mm"


def smooth_volume(data, fwhm_mm, zooms):
    """Gaussian blur with the width given as FWHM in mm per axis.

    Converted internally to a per-axis sigma in voxels:
    ``sigma = fwhm * 0.4247 / voxel_size``.
    """
    fwhm_mm = np.asarray(fwhm_mm, float)
    if not np.any(fwhm_mm > 0):
        return data
    from scipy.ndimage import gaussian_filter
    sigma_vox = (fwhm_mm * FWHM_TO_SIGMA) / np.asarray(zooms, float)
    return gaussian_filter(data, sigma=sigma_vox, mode="constant", cval=0.0)


def volume_preserving_level(original, smoothed, threshold):
    """The level on the SMOOTHED volume enclosing the same voxel count as
    ``|original| >= threshold``.

    Blurring lowers the peak (14.51 -> 9.88 for an isotropic 0.54 mm kernel on
    the Fig 1 map), so drawing at the original threshold pulls the boundary
    inward and eats the cluster -- 856,435 voxels shown instead of 1,412,841, a
    39 % loss. Matching the volume removes the resampling creases without
    shrinking what the statistics actually found.
    """
    target = int((np.abs(original) >= threshold).sum())
    if target == 0:
        return float(threshold)
    flat = np.abs(smoothed).ravel()
    if target >= flat.size:
        return float(flat.min())
    return float(np.partition(flat, flat.size - target)[flat.size - target])


# --------------------------------------------------------------------------
# the render grid
# --------------------------------------------------------------------------
def axis_aligned_world_grid(data, affine, step: int = 1):
    """Reorder a volume onto an axis-aligned world grid in millimetres.

    ``go.Volume`` needs a regular grid, so each world axis must be driven by
    exactly one voxel axis. Permutations and flips are fine (the Allen affine is
    one); a real rotation is not, and raises -- resample first in that case.

    Returns ``(x, y, z, vol)`` with strictly increasing 1-D coordinate vectors
    in mm and ``vol`` indexed ``[ix, iy, iz]``.
    """
    R = np.asarray(affine)[:3, :3]
    order, signs = [], []
    for w in range(3):
        row = R[w]
        k = int(np.argmax(np.abs(row)))
        off_axis = np.abs(row).sum() - abs(row[k])
        if off_axis > 1e-6 * max(abs(row[k]), 1e-9):
            raise ValueError(
                "this volume's affine contains a rotation, so it has no regular "
                "world grid. Resample it onto an axis-aligned grid first "
                "(e.g. flirt -applyxfm with an axis-aligned -ref)."
            )
        order.append(k)
        signs.append(np.sign(row[k]))
    vol = np.transpose(data, order)
    if step > 1:
        vol = vol[::step, ::step, ::step]
    coords = []
    for w in range(3):
        idx = np.arange(vol.shape[w]) * step
        c = affine[w, order[w]] * idx + affine[w, 3]
        if signs[w] < 0:                       # make it increasing
            vol = np.flip(vol, axis=w)
            c = c[::-1]
        coords.append(c)
    return coords[0], coords[1], coords[2], vol


def choose_grid_step(shape, max_voxels: Optional[int]) -> int:
    """Smallest integer step whose grid fits in ``max_voxels``."""
    if not max_voxels:
        return 1
    n = int(np.prod(shape))
    step = 1
    while n > max_voxels and step < 64:
        step += 1
        n = int(np.prod([int(np.ceil(s / step)) for s in shape]))
    return step


def project_render_cost(n_voxels: int) -> Tuple[float, float]:
    """Projected (HTML megabytes, seconds per panel) for a voxel count.

    Fitted to measurements on this pipeline: 56k voxels -> 17.9 MB / 26.7 s;
    127k -> 22.3 MB / 41.6 s; 439k -> 41.9 MB / 106.5 s. Roughly linear at
    ~63 MB and ~209 s per million voxels, plus a fixed overhead. It is an
    estimate, printed so a very large render is a choice rather than a surprise.
    """
    m = n_voxels / 1e6
    return 14.0 + 63.0 * m, 12.0 + 209.0 * m


# --------------------------------------------------------------------------
# spec files
# --------------------------------------------------------------------------
SPEC_KEYS = (
    "path", "name", "cmap", "threshold", "top_percent", "percentile", "range",
    "smooth_fwhm", "source_space", "level", "sign", "opacity", "opacity_floor",
    "gamma", "surfaces", "step", "max_voxels", "crop", "clamp_negative",
)


def load_volume_spec(path) -> List[Dict]:
    """Read a YAML spec file describing one or more maps.

    ::

        volumes:
          - path: pos_z.nii.gz
            name: Activation
            cmap: hot32
            threshold: 3.1
            range: [3.1, 14.5]
            smooth_fwhm: [0.54, 0.11, 0.11]
          - path: neg_z.nii.gz
            name: Deactivation
            cmap: ice28

    **Precedence: a CLI flag beats a spec entry, which beats the default.**
    A flag given once applies to every map in the file.
    """
    import yaml
    with open(path) as fh:
        doc = yaml.safe_load(fh) or {}
    vols = doc.get("volumes", doc if isinstance(doc, list) else None)
    if not vols:
        raise ValueError(f"{path}: expected a top-level 'volumes:' list")
    out = []
    for i, entry in enumerate(vols):
        if isinstance(entry, str):
            entry = {"path": entry}
        unknown = set(entry) - set(SPEC_KEYS)
        if unknown:
            raise ValueError(
                f"{path}: volume #{i + 1} has unknown key(s) "
                f"{sorted(unknown)}. Valid keys: {', '.join(SPEC_KEYS)}"
            )
        if "path" not in entry:
            raise ValueError(f"{path}: volume #{i + 1} has no 'path'")
        out.append(dict(entry))
    return out


def normalize_volume_specs(volumes, overrides: Optional[Dict] = None) -> List[Dict]:
    """Accept any supported form and return a list of per-map dicts.

    ``volumes`` may be a path, a list of paths, a list of dicts, or the path of
    a YAML spec file. ``overrides`` (the CLI flags) win over spec entries.
    """
    if volumes is None:
        return []
    if isinstance(volumes, (str, bytes)) or hasattr(volumes, "__fspath__"):
        p = str(volumes)
        specs = (load_volume_spec(p)
                 if p.lower().endswith((".yaml", ".yml"))
                 else [{"path": p}])
    elif isinstance(volumes, dict):
        specs = [dict(volumes)]
    else:
        specs = []
        for v in volumes:
            specs.append({"path": str(v)} if isinstance(v, (str, bytes)) else dict(v))
    for s in specs:
        for k, v in (overrides or {}).items():
            if v is not None:
                s[k] = v
    return specs


# --------------------------------------------------------------------------
# traces
# --------------------------------------------------------------------------
def _opacity_ramp(floor, ceiling, gamma, signed=False, n=9):
    """plotly ``opacityscale``: floor at the threshold, ceiling at the peak.

    Starting at zero is what makes threshold-edge voxels vanish -- measured,
    14.2 % of visible positive voxels rendered below 5 % opacity. The floor
    keeps the whole map visible. For a signed map the ramp is two-sided: opaque
    at both extremes, transparent through the middle (which is NaN anyway).
    """
    ts = np.linspace(0.0, 1.0, n)
    out = []
    for t in ts:
        v = abs(2.0 * t - 1.0) if signed else t
        out.append([float(t), float(floor + (ceiling - floor) * (v ** gamma))])
    return out


def build_volume_traces(spec_data, *, background="white", adapt_cmap=True,
                        show_colorbar=True):
    """Build one ``go.Volume`` trace per prepared map.

    ``spec_data`` is a list of dicts from :func:`prepare_volume`, each carrying
    x/y/z/vol, the level, the colour range and the look parameters.
    """
    import plotly.graph_objects as go
    traces = []
    n = len(spec_data)
    for k, d in enumerate(spec_data):
        vol = d["vol"]
        level, vmax = d["level"], d["vmax"]
        signed = d["signed"]
        if signed:
            cs = signed_colorscale(background, adapt_cmap)
            isomin, isomax = -vmax, vmax
            value = np.where(np.abs(vol) >= level, vol, np.nan)
        else:
            cs, _ = get_colorscale(d["cmap"], background, adapt_cmap)
            isomin, isomax = level, vmax
            value = np.where(np.abs(vol) >= level, np.abs(vol), np.nan)
        if d.get("vmin") is not None:
            isomin = float(d["vmin"])
        if d.get("vmax_user") is not None:
            isomax = float(d["vmax_user"])

        X, Y, Z = np.meshgrid(d["x"], d["y"], d["z"], indexing="ij")
        opac = _opacity_ramp(d["opacity_floor"], d["opacity"], d["gamma"],
                             signed=signed)
        # A colorbar is a layout-level artefact in plotly; naming it legend_*
        # is what lets --export-no-legend strip it, matching the size/width keys.
        cb = dict(
            x=1.02 + 0.13 * k, thickness=14, len=0.55,
            title=dict(text=d["name"], font=dict(size=12)),
            outlinewidth=0,
        ) if show_colorbar else None
        traces.append(go.Volume(
            x=X.ravel(), y=Y.ravel(), z=Z.ravel(), value=value.ravel(),
            isomin=isomin, isomax=isomax, cmin=isomin, cmax=isomax,
            colorscale=cs, opacity=1.0, opacityscale=opac,
            surface_count=int(d["surfaces"]),
            showscale=bool(show_colorbar), colorbar=cb,
            caps=dict(x_show=False, y_show=False, z_show=False),
            lighting=dict(ambient=0.9, diffuse=0.2, specular=0.0),
            hoverinfo="skip",
            name=f"Volume: {d['name']}",
            legendgroup=f"volume_{k}", showlegend=False,
        ))
    return traces


def prepare_volume(spec, *, quiet=False):
    """Load, threshold, smooth and grid one map, reporting as it goes.

    Returns the dict consumed by :func:`build_volume_traces`.
    """
    path = spec["path"]
    name = spec.get("name") or str(path).split("/")[-1].split(".")[0]
    data, affine, zooms = load_volume_map(
        path, clamp_negative=bool(spec.get("clamp_negative", False)))

    sign = spec.get("sign", "both")
    thr, rep = resolve_threshold(
        data, absolute=spec.get("threshold"),
        top_percent=spec.get("top_percent"),
        percentile=spec.get("percentile"), sign=sign, name=name)
    if not quiet:
        print(format_threshold_report(rep), file=sys.stderr)

    # a map holding meaningful values of BOTH signs is drawn on one diverging
    # scale; two separate files (hot32 + ice28) are the recommended alternative
    has_pos = bool(np.any(data > thr))
    has_neg = bool(np.any(data < -thr))
    signed = has_pos and has_neg
    if signed and not quiet:
        print("    both signs present -> ONE diverging colorscale. For the "
              "hot/ice convention supply the positive and negative maps as "
              "SEPARATE files.", file=sys.stderr)

    if spec.get("crop", True):
        sub, off = crop_to_support(data, thr)
        if not quiet and sub.shape != data.shape:
            wlo = affine[:3, :3] @ off + affine[:3, 3]
            whi = affine[:3, :3] @ (off + np.array(sub.shape) - 1) + affine[:3, 3]
            print(f"    cropped to suprathreshold bbox "
                  f"+{VOLUME_DEFAULTS['crop_margin']} vox: "
                  f"{'x'.join(map(str, sub.shape))} (was "
                  f"{'x'.join(map(str, data.shape))})", file=sys.stderr)
            print(f"      world mm  x {min(wlo[0], whi[0]):.2f}..{max(wlo[0], whi[0]):.2f}"
                  f"  y {min(wlo[1], whi[1]):.2f}..{max(wlo[1], whi[1]):.2f}"
                  f"  z {min(wlo[2], whi[2]):.2f}..{max(wlo[2], whi[2]):.2f}",
                  file=sys.stderr)
    else:
        sub, off = data, np.zeros(3, int)

    fwhm, how = resolve_smoothing_fwhm(
        spec.get("smooth_fwhm", spec.get("source_space")), sub, thr, zooms)
    sm = smooth_volume(sub, fwhm, zooms)
    if not quiet and np.any(fwhm > 0):
        print(f"    smoothing     : {how}", file=sys.stderr)

    level_mode = str(spec.get("level", "preserve")).lower()
    if np.any(fwhm > 0) and level_mode == "preserve":
        level = volume_preserving_level(sub, sm, thr)
        if not quiet:
            print(f"    level         : {thr:.3f} -> {level:.3f} "
                  f"(volume-preserving; --volume-level fixed to keep {thr:.3f})",
                  file=sys.stderr)
    else:
        level = thr

    aff2 = affine.copy()
    aff2[:3, 3] = affine[:3, :3] @ off + affine[:3, 3]

    step = int(spec.get("step") or 0) or choose_grid_step(
        sm.shape, spec.get("max_voxels"))
    x, y, z, vol = axis_aligned_world_grid(sm, aff2, step=step)
    n_vox = int(vol.size)
    mb, secs = project_render_cost(n_vox)
    if not quiet:
        print(f"    render grid   : {'x'.join(map(str, vol.shape))} "
              f"= {n_vox:,} voxels (step {step})", file=sys.stderr)
        print(f"    projected cost: ~{mb:.0f} MB HTML, ~{secs:.0f} s per panel",
              file=sys.stderr)
        if n_vox > 1_000_000:
            print(f"    WARNING: {n_vox:,} voxels is large. go.Volume ships "
                  f"x/y/z/value as four full arrays to the browser. Reduce "
                  f"with --volume-step N or --volume-max-voxels N; with a "
                  f"0.54 mm smoothing kernel a coarser grid is visually "
                  f"identical.", file=sys.stderr)

    mask = np.abs(vol) >= level
    vmax = float(np.percentile(np.abs(vol)[mask], 99.5)) if mask.any() else level + 1.0
    rng = spec.get("range")
    vmin_user = vmax_user = None
    if rng is not None:
        if isinstance(rng, str):
            rng = [float(v) for v in rng.replace(",", " ").split()]
        vmin_user, vmax_user = float(rng[0]), float(rng[1])

    return dict(
        x=x, y=y, z=z, vol=vol, level=level, vmax=vmax, signed=signed,
        name=name, cmap=spec.get("cmap") or ("ice28" if sign == "negative"
                                             else "hot32"),
        opacity=float(spec.get("opacity", VOLUME_DEFAULTS["opacity"])),
        opacity_floor=float(spec.get("opacity_floor",
                                     VOLUME_DEFAULTS["opacity_floor"])),
        gamma=float(spec.get("gamma", VOLUME_DEFAULTS["gamma"])),
        surfaces=min(int(spec.get("surfaces", VOLUME_DEFAULTS["surfaces"])),
                     MAX_SURFACES),
        vmin=vmin_user, vmax_user=vmax_user, n_voxels=n_vox,
    )


def ghost_mesh_trace(vertices, faces, *, opacity=VOLUME_DEFAULTS["ghost_opacity"],
                     color="#8e8e9a", lighting=None):
    """The translucent brain shell the voxel cloud sits inside.

    This is the BRAIN's opacity -- distinct from ``--volume-opacity``, which is
    the voxel map's.
    """
    import plotly.graph_objects as go
    return go.Mesh3d(
        x=vertices[:, 0], y=vertices[:, 1], z=vertices[:, 2],
        i=faces[:, 0], j=faces[:, 1], k=faces[:, 2],
        color=color, opacity=float(opacity), name="Brain Surface",
        lighting=lighting or dict(ambient=0.55, diffuse=0.45, specular=0.30,
                                  roughness=0.55, fresnel=1.40),
        hoverinfo="skip", showlegend=False,
    )


def check_volume_space(volume_path, vertices, quiet=False) -> Dict:
    """Report whether a map and a mesh occupy the same world space.

    A voxel overlay silently landing off the brain is the most common volume
    failure, so this runs on every volume render. It reuses
    :func:`HarrisLabPlotting.utils.compare_volume_mesh_space`.
    """
    from .utils import compare_volume_mesh_space
    try:
        rep = compare_volume_mesh_space(str(volume_path), vertices)
    except Exception as e:                     # never block a render on this
        if not quiet:
            print(f"    space check   : skipped ({type(e).__name__}: {e})",
                  file=sys.stderr)
        return {}
    if not quiet:
        print(f"    space check   : bbox overlap "
              f"{rep.get('bbox_overlap_fraction', float('nan')):.0%}, "
              f"centroid offset {rep.get('centroid_offset_mm', float('nan')):.2f} mm "
              f"-> {rep.get('verdict', '?')}", file=sys.stderr)
        if str(rep.get("verdict", "")).upper().startswith("FAIL"):
            print("    WARNING: the map and the mesh are in DIFFERENT spaces. "
                  "The overlay will not sit on the brain. See the voxel "
                  "tutorial's 'Getting your data into the same space' section, "
                  "or run `hlplot utils check-alignment`.", file=sys.stderr)
    return rep


# --------------------------------------------------------------------------
# the public plotting entry point
# --------------------------------------------------------------------------
def create_brain_volume_plot(
    mesh=None,
    volumes=None,
    *,
    vertices=None,
    faces=None,
    plot_title: str = "",
    save_path: str = "brain_volume.html",
    background_color: str = "white",
    glass: bool = True,
    mesh_color: str = "#8e8e9a",
    ghost_opacity: float = VOLUME_DEFAULTS["ghost_opacity"],
    camera_view: str = "oblique",
    custom_camera: Optional[Dict] = None,
    zoom: float = 1.0,
    adapt_cmap: bool = True,
    show_colorbar: bool = True,
    export_image: Optional[str] = None,
    image_dpi: int = 300,
    export_size: Tuple[int, int] = (1200, 1200),
    export_show_title: bool = True,
    export_show_legend: bool = True,
    export_autocrop: bool = False,
    multi_view: Optional[List[Union[str, Dict]]] = None,
    multi_view_panel_size: Tuple[int, int] = (800, 800),
    multi_view_panel_labels: Optional[List[str]] = None,
    multi_view_grid: Optional[Tuple[int, int]] = None,
    multi_view_keep_first_legend: bool = True,
    no_html: bool = False,
    check_space: bool = True,
    quiet: bool = False,
    **volume_overrides,
):
    """Render one or more statistical volumes inside a glass brain.

    Parameters
    ----------
    mesh : str or Path, optional
        Brain mesh file (.gii/.obj/.mz3/.ply). Alternatively pass
        ``vertices``/``faces`` directly.
    volumes : str, list of str, list of dict, or str path to a YAML spec
        The map(s) to render. A bare path uses defaults; a dict gives full
        per-map control with the same keys as the YAML spec (``path``, ``name``,
        ``cmap``, ``threshold``, ``range``, ``smooth_fwhm``, ``level``,
        ``opacity``, ``gamma``, ``surfaces``, ``step`` ...).
    glass : bool
        Draw the brain as a translucent shell. Default True.
    ghost_opacity : float
        Opacity of the BRAIN shell, 0-1. Default 0.04. This is not the voxel
        map's opacity -- that is the per-map ``opacity``.
    adapt_cmap : bool
        On a light background, truncate the top of ``hot32``/``ice28`` so the
        peak stays visible. Default True.
    **volume_overrides
        Any per-map key (``threshold``, ``cmap``, ``opacity``, ``gamma``,
        ``surfaces``, ``step``, ``max_voxels``, ``smooth_fwhm``, ``level``, ...)
        given here overrides the same key for EVERY map, including entries from
        a spec file. Precedence: override > spec entry > default.

    Returns
    -------
    (fig, info) : (plotly.graph_objects.Figure, dict)
    """
    import plotly.graph_objects as go
    from .mesh import load_mesh_file
    from .camera import CameraController
    from .connectivity import _export_figure_static

    if vertices is None or faces is None:
        if mesh is None:
            raise ValueError("pass either mesh=... or vertices=/faces=")
        vertices, faces = load_mesh_file(mesh)
    vertices = np.asarray(vertices, float)
    faces = np.asarray(faces, int)

    specs = normalize_volume_specs(volumes, volume_overrides)
    if not specs:
        raise ValueError("no volumes given -- pass volumes='map.nii.gz' or a list")

    prepared, infos = [], []
    for spec in specs:
        if check_space:
            check_volume_space(spec["path"], vertices, quiet=quiet)
        d = prepare_volume(spec, quiet=quiet)
        prepared.append(d)
        infos.append(dict(name=d["name"], level=d["level"],
                          n_voxels=d["n_voxels"], vmax=d["vmax"]))

    fig = go.Figure()
    for tr in build_volume_traces(prepared, background=background_color,
                                  adapt_cmap=adapt_cmap,
                                  show_colorbar=show_colorbar):
        fig.add_trace(tr)
    if glass:
        fig.add_trace(ghost_mesh_trace(vertices, faces, opacity=ghost_opacity,
                                       color=mesh_color))

    if custom_camera is not None:
        camera = dict(custom_camera)
    else:
        camera = CameraController.get_camera_position(camera_view)
    if zoom and zoom != 1.0:
        camera["eye"] = {k: float(v) / float(zoom)
                         for k, v in camera["eye"].items()}

    fig.update_layout(
        scene=dict(xaxis=dict(visible=False), yaxis=dict(visible=False),
                   zaxis=dict(visible=False), bgcolor=background_color,
                   aspectmode="data",
                   camera=dict(eye=camera["eye"], center=camera["center"],
                               up=camera["up"])),
        width=export_size[0], height=export_size[1],
        paper_bgcolor=background_color, plot_bgcolor=background_color,
        margin=dict(l=0, r=0, t=48 if plot_title else 0, b=0),
        title=dict(text=plot_title, x=0.5, xanchor="center",
                   font=dict(size=20)) if plot_title else None,
        showlegend=False,
    )

    from pathlib import Path as _P
    save_path = _P(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    if no_html:
        if not quiet:
            print("no_html=True: skipping the interactive HTML")
    else:
        fig.write_html(str(save_path))
        if not quiet:
            print(f"Saved interactive visualization to: {save_path}")

    if export_image:
        # _export_figure_static dispatches to the multi-view stitcher itself
        # when multi_view is set, so single-view and multi-view share one path
        # (and one set of title/legend/background rules).
        _export_figure_static(
            fig, export_image=export_image,
            multi_view=multi_view,
            multi_view_panel_size=multi_view_panel_size,
            multi_view_panel_labels=multi_view_panel_labels,
            multi_view_keep_first_legend=multi_view_keep_first_legend,
            multi_view_grid=multi_view_grid,
            zoom=zoom, image_dpi=image_dpi, image_format="png",
            plot_title=plot_title, export_show_title=export_show_title,
            export_show_legend=export_show_legend,
            background_color=background_color, export_size=export_size,
            export_autocrop=export_autocrop,
        )

    return fig, dict(volumes=infos, background=background_color)
