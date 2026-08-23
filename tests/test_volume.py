"""Volume preparation: thresholds, smoothing, level correction, the grid."""
import numpy as np
import pytest

from HarrisLabPlotting.volume import (
    axis_aligned_world_grid, choose_grid_step, crop_to_support,
    get_colorscale, project_render_cost, resolve_smoothing_fwhm,
    resolve_threshold, signed_colorscale, smooth_volume,
    volume_preserving_level, normalize_volume_specs,
)


@pytest.fixture
def blob():
    """A smooth 3-D gaussian blob peaking at 10, on a 1 mm isotropic grid."""
    g = np.mgrid[-20:20, -20:20, -20:20].astype(float)
    r2 = (g ** 2).sum(axis=0)
    v = 10.0 * np.exp(-r2 / (2 * 6.0 ** 2))
    v[v < 0.5] = 0.0
    affine = np.diag([1.0, 1.0, 1.0, 1.0])
    affine[:3, 3] = [-20.0, -20.0, -20.0]
    return v.astype(np.float32), affine, np.ones(3)


# ----- thresholds --------------------------------------------------------
def test_absolute_threshold(blob):
    v, _, _ = blob
    thr, rep = resolve_threshold(v, absolute=3.1)
    assert thr == 3.1
    assert rep["n_kept"] == int((np.abs(v[v != 0]) >= 3.1).sum())
    assert "absolute" in rep["how"]


def test_top_percent_keeps_roughly_that_fraction(blob):
    v, _, _ = blob
    thr, rep = resolve_threshold(v, top_percent=10)
    pool = np.abs(v[v != 0])
    assert rep["n_kept"] == pytest.approx(0.10 * pool.size, rel=0.15)
    assert thr > np.median(pool)


def test_percentile_mode(blob):
    v, _, _ = blob
    thr, _ = resolve_threshold(v, percentile=99)
    assert thr == pytest.approx(np.percentile(np.abs(v[v != 0]), 99))


def test_auto_threshold_shows_everything_in_the_file(blob):
    v, _, _ = blob
    thr, rep = resolve_threshold(v)
    assert thr == pytest.approx(np.abs(v[v != 0]).min())
    assert rep["n_kept"] == int((v != 0).sum())


def test_two_threshold_modes_is_an_error(blob):
    v, _, _ = blob
    with pytest.raises(ValueError, match="more than one threshold mode"):
        resolve_threshold(v, absolute=3.1, top_percent=10)


def test_report_carries_the_range_and_distribution(blob):
    v, _, _ = blob
    _, rep = resolve_threshold(v, absolute=3.1)
    assert rep["vmax"] == pytest.approx(10.0, rel=1e-3)
    assert set(rep["percentiles"]) == {50, 75, 90, 95, 99}


# ----- crop --------------------------------------------------------------
def test_crop_keeps_every_suprathreshold_voxel(blob):
    v, _, _ = blob
    sub, off = crop_to_support(v, 3.1, margin=2)
    assert sub.shape < v.shape
    assert int((np.abs(sub) >= 3.1).sum()) == int((np.abs(v) >= 3.1).sum())


def test_crop_offset_locates_the_subvolume(blob):
    v, _, _ = blob
    sub, off = crop_to_support(v, 3.1, margin=0)
    assert np.allclose(v[off[0]:off[0] + sub.shape[0],
                         off[1]:off[1] + sub.shape[1],
                         off[2]:off[2] + sub.shape[2]], sub)


# ----- smoothing and level ----------------------------------------------
def test_fwhm_forms(blob):
    v, _, zooms = blob
    assert np.allclose(resolve_smoothing_fwhm(None, v, 3.1, zooms)[0], 0)
    assert np.allclose(resolve_smoothing_fwhm(2.0, v, 3.1, zooms)[0], [2, 2, 2])
    assert np.allclose(
        resolve_smoothing_fwhm((0.54, 0.11, 0.11), v, 3.1, zooms)[0],
        [0.54, 0.11, 0.11])
    assert np.allclose(
        resolve_smoothing_fwhm("0.54,0.11,0.11", v, 3.1, zooms)[0],
        [0.54, 0.11, 0.11])


def test_smoothing_lowers_the_peak(blob):
    v, _, zooms = blob
    sm = smooth_volume(v, [4.0, 4.0, 4.0], zooms)
    assert sm.max() < v.max()


def test_volume_preserving_level_keeps_the_cluster_size(blob):
    """A fixed level after blurring eats the cluster; the corrected level
    must restore the original voxel count."""
    v, _, zooms = blob
    sm = smooth_volume(v, [4.0, 4.0, 4.0], zooms)
    target = int((np.abs(v) >= 3.1).sum())
    at_fixed = int((np.abs(sm) >= 3.1).sum())
    lvl = volume_preserving_level(v, sm, 3.1)
    at_corrected = int((np.abs(sm) >= lvl).sum())
    assert at_fixed < target                       # the loss this exists to fix
    assert at_corrected == pytest.approx(target, rel=1e-3)
    assert lvl < 3.1


# ----- grid --------------------------------------------------------------
def test_world_grid_is_increasing_and_in_mm(blob):
    v, affine, _ = blob
    x, y, z, vol = axis_aligned_world_grid(v, affine)
    assert vol.shape == v.shape
    for c in (x, y, z):
        assert np.all(np.diff(c) > 0)
    assert x[0] == pytest.approx(-20.0)


def test_world_grid_handles_flips_and_permutations():
    """The Allen affine is a permutation with sign flips; the grid must come
    out axis-aligned and increasing anyway."""
    v = np.arange(2 * 3 * 4, dtype=float).reshape(2, 3, 4)
    affine = np.zeros((4, 4))
    affine[0, 2] = 0.025
    affine[1, 0] = -0.025
    affine[2, 1] = -0.025
    affine[:3, 3] = [-5.695, 5.35, 5.22]
    affine[3, 3] = 1.0
    x, y, z, vol = axis_aligned_world_grid(v, affine)
    assert vol.shape == (4, 2, 3)                  # permuted
    for c in (x, y, z):
        assert np.all(np.diff(c) > 0)


def test_rotated_affine_is_rejected():
    v = np.zeros((3, 3, 3))
    affine = np.eye(4)
    affine[:3, :3] = [[0.7, 0.7, 0], [-0.7, 0.7, 0], [0, 0, 1]]
    with pytest.raises(ValueError, match="rotation"):
        axis_aligned_world_grid(v, affine)


def test_step_subsamples(blob):
    v, affine, _ = blob
    _, _, _, vol = axis_aligned_world_grid(v, affine, step=4)
    assert vol.size < v.size / 40


def test_choose_grid_step_meets_the_budget():
    assert choose_grid_step((285, 300, 297), None) == 1
    step = choose_grid_step((285, 300, 297), 120_000)
    n = np.prod([int(np.ceil(s / step)) for s in (285, 300, 297)])
    assert n <= 120_000


def test_cost_projection_is_monotonic():
    mb1, s1 = project_render_cost(56_000)
    mb2, s2 = project_render_cost(439_000)
    assert mb2 > mb1 and s2 > s1


# ----- colorscales -------------------------------------------------------
def test_light_background_truncates_the_top():
    """hot32 runs to pure white, invisible on a white page."""
    dark, name_dark = get_colorscale("hot32", "black")
    light, name_light = get_colorscale("hot32", "white")
    assert name_dark == "hot32" and name_light == "hot32_light"
    assert dark[-1][1].lower() == "#ffffff"
    assert light[-1][1].lower() != "#ffffff"


def test_adapt_can_be_disabled():
    scale, name = get_colorscale("hot32", "white", adapt=False)
    assert name == "hot32" and scale[-1][1].lower() == "#ffffff"


def test_plotly_names_pass_through():
    scale, name = get_colorscale("Viridis", "white")
    assert scale == "Viridis" and name == "Viridis"


def test_signed_scale_is_monotonic_and_spans_zero_to_one():
    cs = signed_colorscale("black")
    pos = [p for p, _ in cs]
    assert pos[0] == 0.0 and pos[-1] == 1.0
    assert all(b >= a for a, b in zip(pos, pos[1:]))


# ----- specs -------------------------------------------------------------
def test_spec_forms_all_normalize():
    assert normalize_volume_specs("a.nii.gz") == [{"path": "a.nii.gz"}]
    assert normalize_volume_specs(["a.nii.gz", "b.nii.gz"]) == [
        {"path": "a.nii.gz"}, {"path": "b.nii.gz"}]
    got = normalize_volume_specs([dict(path="a.nii.gz", cmap="ice28")])
    assert got[0]["cmap"] == "ice28"


def test_cli_override_beats_the_spec_entry():
    got = normalize_volume_specs([dict(path="a.nii.gz", threshold=3.1)],
                                 {"threshold": 4.0})
    assert got[0]["threshold"] == 4.0


def test_none_overrides_are_ignored():
    got = normalize_volume_specs([dict(path="a.nii.gz", threshold=3.1)],
                                 {"threshold": None, "cmap": "hot32"})
    assert got[0]["threshold"] == 3.1 and got[0]["cmap"] == "hot32"
