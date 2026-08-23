"""Arrow geometry: direction, per-edge sizing, and the two caps."""
import numpy as np
import pytest

from HarrisLabPlotting.directed import (
    ARROW_DEFAULTS, arrowhead_mesh, arrowhead_size, build_directed_edge_traces,
    edge_polyline, extract_directed_edges,
)


def test_one_way_edge_yields_exactly_one_arrow():
    m = np.zeros((3, 3))
    m[2, 0] = 0.5
    edges = extract_directed_edges(m)
    assert len(edges) == 1 and not edges[0]["reciprocal"]


def test_reciprocal_pair_is_flagged_both_ways():
    m = np.zeros((3, 3))
    m[0, 1], m[1, 0] = 0.8, 0.3
    edges = extract_directed_edges(m)
    assert len(edges) == 2
    assert all(e["reciprocal"] for e in edges)
    assert {e["weight"] for e in edges} == {0.8, 0.3}


def test_diagonal_never_becomes_an_edge():
    m = np.zeros((3, 3))
    np.fill_diagonal(m, 1.0)
    assert extract_directed_edges(m) == []


def test_edge_threshold_filters():
    m = np.zeros((3, 3))
    m[0, 1], m[1, 2] = 0.5, 0.05
    assert len(extract_directed_edges(m, edge_threshold=0.1)) == 1


def test_arrowhead_points_from_source_to_target():
    """The cone's tip must be further along source->target than its base."""
    tip = np.array([10.0, 0.0, 0.0])
    direction = np.array([1.0, 0.0, 0.0])
    verts, faces = arrowhead_mesh(tip, direction, length=2.0, radius=0.5)
    assert np.allclose(verts[0], tip)             # vertex 0 is the tip
    assert verts[1][0] == pytest.approx(8.0)      # base is 2.0 behind it
    assert faces.shape[1] == 3


def test_cone_never_narrower_than_its_own_line():
    """The rule that makes an arrowhead visible on the edge it caps."""
    upp = 0.25          # data units per pixel
    for width_px in (0.5, 1.0, 4.0, 9.0):
        radius, _ = arrowhead_size(width_px, upp, edge_length=100.0)
        half_line = 0.5 * width_px * upp
        floor = ARROW_DEFAULTS["min_radius_px"] * upp
        assert radius >= min(half_line, floor) - 1e-12
        assert radius == pytest.approx(
            max(ARROW_DEFAULTS["k_width"] * half_line, floor))


def test_pixel_floor_keeps_thin_edges_visible():
    upp = 0.25
    r_thin, _ = arrowhead_size(0.2, upp, 100.0)
    assert r_thin == pytest.approx(ARROW_DEFAULTS["min_radius_px"] * upp)


def test_short_edge_cap():
    """A 9 px line on a 9.9-unit edge must not get an 11.7-unit arrowhead."""
    upp = 0.3753
    _, long_edge = arrowhead_size(9.0, upp, edge_length=80.0)
    _, short_edge = arrowhead_size(9.0, upp, edge_length=9.9)
    assert short_edge == pytest.approx(ARROW_DEFAULTS["max_edge_frac"] * 9.9)
    assert short_edge < long_edge


def test_one_way_edges_are_straight_reciprocal_are_bowed():
    centroid = np.zeros(3)
    p0, p1 = np.array([-10.0, 0, 0]), np.array([10.0, 0, 0])
    straight = edge_polyline(p0, p1, reciprocal=False, flip=False,
                             centroid=centroid, diag=100.0)
    # every sample must lie ON the chord (test collinearity, not a specific
    # index -- with an even sample count no point sits exactly at the midpoint)
    chord = p1 - p0
    dev = np.linalg.norm(np.cross(straight - p0, chord), axis=1) / np.linalg.norm(chord)
    assert dev.max() < 1e-6

    bowed = edge_polyline(p0, p1, reciprocal=True, flip=False,
                          centroid=centroid, diag=100.0)
    dev_b = np.linalg.norm(np.cross(bowed - p0, chord), axis=1) / np.linalg.norm(chord)
    assert dev_b.max() > 1.0


def test_reciprocal_pair_bows_to_opposite_sides():
    centroid = np.zeros(3)
    p0, p1 = np.array([-10.0, 0, 0]), np.array([10.0, 0, 0])
    a = edge_polyline(p0, p1, reciprocal=True, flip=False,
                      centroid=centroid, diag=100.0)
    b = edge_polyline(p1, p0, reciprocal=True, flip=True,
                      centroid=centroid, diag=100.0)
    chord = p1 - p0
    def perp(pts, origin):
        d = pts - origin
        return d - np.outer(d @ chord / (chord @ chord), chord)
    off_a = perp(a, p0)[len(a) // 2]
    off_b = perp(b, p1)[len(b) // 2]
    assert float(np.dot(off_a, off_b)) < 0      # opposite sides


def test_short_pair_still_separates_via_the_bow_floor():
    """Without a floor, a very short reciprocal pair collapses onto itself."""
    centroid = np.zeros(3)
    p0, p1 = np.array([0.0, 0, 0]), np.array([2.0, 0, 0])   # 2 units apart
    a = edge_polyline(p0, p1, reciprocal=True, flip=False,
                      centroid=centroid, diag=170.0)
    chord = p1 - p0
    d = a - p0
    perp = d - np.outer(d @ chord / (chord @ chord), chord)
    sagitta = np.linalg.norm(perp, axis=1).max()
    assert sagitta >= ARROW_DEFAULTS["bow_floor_frac"] * 170.0 * 0.9


def test_per_edge_widths_survive_bucketing():
    """A plotly Scatter3d carries one width, so the undirected path averages
    them; bucketing must preserve distinct widths instead."""
    n = 12
    pos = np.random.default_rng(0).normal(0, 30, (n, 3))
    m = np.zeros((n, n))
    widths = []
    edges = []
    for k in range(n - 1):
        m[k, k + 1] = 1.0
        edges.append(dict(i=k, j=k + 1, weight=1.0, reciprocal=False))
        widths.append(1.0 + k)          # every edge a different width
    traces, counts = build_directed_edge_traces(
        edges, pos, widths=widths, hovers=[""] * len(edges),
        pos_color="#ff0000", neg_color="#0000ff", units_per_px=0.25,
        centroid=pos.mean(0), diag=100.0, node_radius=1.0)
    line_widths = {round(float(t.line.width), 2) for t in traces
                   if t.type == "scatter3d"}
    assert len(line_widths) >= len(set(widths)) - 2   # bucketed, not averaged
    assert counts["pos"] == len(edges)


def test_arrowheads_share_their_lines_legendgroup():
    edges = [dict(i=0, j=1, weight=1.0, reciprocal=False)]
    pos = np.array([[0.0, 0, 0], [20.0, 0, 0], [0, 20.0, 0]])
    traces, _ = build_directed_edge_traces(
        edges, pos, widths=[3.0], hovers=[""], pos_color="#ff0000",
        neg_color="#0000ff", units_per_px=0.25, centroid=pos.mean(0),
        diag=100.0, node_radius=1.0)
    groups = {t.legendgroup for t in traces}
    assert groups == {"pos_edges"}
    meshes = [t for t in traces if t.type == "mesh3d"]
    assert meshes and all(m.showlegend is False for m in meshes)
