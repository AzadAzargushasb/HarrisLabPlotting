"""
Directed (asymmetric) connectivity support
==========================================

Everything needed to turn an asymmetric matrix into arrow geometry:

- :func:`check_matrix_symmetry` -- the report printed on every plot
- :func:`apply_matrix_orientation` -- row->col vs col->row
- :func:`extract_directed_edges` -- full-matrix edge list, reciprocal flags
- :func:`edge_polyline` -- straight chord, or a bowed arc for reciprocal pairs
- :func:`arrowhead_mesh` / :func:`arrowheads_to_trace` -- cone geometry

Orientation
-----------
``M[i, j]`` means **i -> j** (row = source, column = target), the numpy /
networkx convention.

**SPM's DCM uses the opposite convention** -- its state equation is
``dx/dt = A x``, and ``(A x)_i = sum_j A[i,j] x_j``, so ``A(i,j)`` is the
connection *from* region *j* *to* region *i*. Those matrices must be transposed,
with ``matrix_orientation='col-to-row'`` or ``hlplot utils transpose``.

Transition-probability matrices are usually **row-stochastic** (each row sums to
1, so row = current state = source) and need **no** transpose; only the
column-stochastic variety does. Check with ``M.sum(axis=1)`` vs
``M.sum(axis=0)`` -- whichever is ~1 tells you which index is the source.

Getting this wrong silently reverses every arrow, which is why the symmetry
report always states the orientation in use.

Why arrowheads are ``Mesh3d`` cones and not ``go.Cone``
-------------------------------------------------------
``go.Cone`` was measured and rejected:

* a Cone trace holding a **single point** renders nothing at all under
  ``sizemode='absolute'`` -- plotly derives an internal size factor from the
  spacing between successive points, and one point makes it degenerate;
* under ``sizemode='scaled'`` cone length is proportional to the vector norm,
  but the constant depends on the trace's maximum norm *and* on the point
  spacing, so the same edge width yields different arrowheads in different
  figures.

Explicit ``Mesh3d`` cones have exact length and radius in data units, are
reproducible across figures, and cost one trace per colour group.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

__all__ = [
    "check_matrix_symmetry",
    "format_symmetry_report",
    "apply_matrix_orientation",
    "extract_directed_edges",
    "edge_polyline",
    "trim_from_end",
    "arrowhead_mesh",
    "billboard_arrowhead",
    "arrowheads_to_trace",
    "arrowhead_size",
    "build_directed_edge_traces",
    "darken_color",
    "estimate_units_per_pixel",
    "ARROW_DEFAULTS",
]


# Defaults chosen from the rendered bake-off (see the directed-graph tutorial).
ARROW_DEFAULTS = dict(
    k_width=1.08,        # cone radius / that edge's own line half-width
    min_radius_px=1.2,   # floor so the thinnest edges still show a head
    slenderness=0.18,    # cone radius / cone length
    max_edge_frac=0.30,  # cone length <= this fraction of the edge's length
    bow_chord_frac=0.10,  # reciprocal arc sagitta, as a fraction of the chord
    bow_floor_frac=0.03,  # ... with this floor, as a fraction of the bbox diagonal
    darken=0.74,         # arrowheads are this shade of their line colour
    segments=18,         # facets around the cone base
    samples=48,          # points along an arc
)


# --------------------------------------------------------------------------
# symmetry
# --------------------------------------------------------------------------
def check_matrix_symmetry(matrix, tol: Optional[float] = None,
                          orientation: str = "row-to-col") -> Dict:
    """Describe how (a)symmetric a connectivity matrix is.

    Parameters
    ----------
    matrix : array-like, square
    tol : float, optional
        Absolute tolerance handed to :func:`numpy.allclose` (whose defaults,
        ``rtol=1e-5, atol=1e-8``, are used when this is ``None``). A matrix
        round-tripped through CSV typically differs from its transpose by
        ~1e-16, which must not be read as "directed".
    orientation : {'row-to-col', 'col-to-row'}
        Recorded in the report so the figure's direction convention is never
        ambiguous. It does **not** transform the matrix -- use
        :func:`apply_matrix_orientation` for that.

    Returns
    -------
    dict
        ``is_symmetric``, ``directed``, ``max_asymmetry``, ``n_asym_cells``,
        ``n_reciprocal``, ``n_oneway``, ``n_edges``, ``n_diagonal``,
        ``diagonal_sum``, ``orientation``, ``tol``.
    """
    m = np.asarray(matrix, dtype=float)
    if m.ndim != 2 or m.shape[0] != m.shape[1]:
        raise ValueError(f"matrix must be square, got shape {m.shape}")

    diag = np.diag(m)
    n_diagonal = int(np.count_nonzero(diag))
    off = m.copy()
    np.fill_diagonal(off, 0.0)

    if tol is None:
        is_sym = bool(np.allclose(off, off.T))
    else:
        is_sym = bool(np.allclose(off, off.T, rtol=0.0, atol=float(tol)))

    diff = np.abs(off - off.T)
    max_asym = float(diff.max()) if diff.size else 0.0
    thresh = (np.finfo(float).eps if tol is None else float(tol))
    n_asym_cells = int((diff > max(thresh, 1e-12)).sum())

    upper = np.triu(off, 1) != 0
    lower = (np.tril(off, -1) != 0).T          # transposed -> same indexing
    n_reciprocal = int((upper & lower).sum())
    n_edges = int(np.count_nonzero(off))
    n_oneway = n_edges - 2 * n_reciprocal

    return dict(
        is_symmetric=is_sym,
        directed=not is_sym,
        max_asymmetry=max_asym,
        n_asym_cells=n_asym_cells,
        n_reciprocal=n_reciprocal,
        n_oneway=n_oneway,
        n_edges=n_edges,
        n_diagonal=n_diagonal,
        diagonal_sum=float(np.abs(diag).sum()),
        orientation=orientation,
        tol=tol,
    )


def format_symmetry_report(report: Dict, drawing_directed: Optional[bool] = None
                           ) -> str:
    """Render :func:`check_matrix_symmetry` output as the printed block."""
    sym = "SYMMETRIC (undirected)" if report["is_symmetric"] else "ASYMMETRIC (directed)"
    lines = [f"  Matrix symmetry  : {sym}",
             f"    max|M - M.T|   : {report['max_asymmetry']:.6g}"]
    if not report["is_symmetric"]:
        lines.append(f"    asymmetric cells: {report['n_asym_cells']}")
    lines.append(f"    edges          : {report['n_edges']} "
                 f"({report['n_reciprocal']} reciprocal pairs, "
                 f"{report['n_oneway']} one-way)")
    if report["n_diagonal"]:
        lines.append(f"    diagonal       : {report['n_diagonal']} nonzero "
                     f"(self-loops) -> IGNORED, not drawn")
    if drawing_directed is not None:
        how = "DIRECTED (arrowheads on)" if drawing_directed else "UNDIRECTED"
        lines.append(f"  -> drawing {how}")
        if drawing_directed:
            arrow = ("row = source -> column = target"
                     if report["orientation"] == "row-to-col"
                     else "column = source -> row = target")
            lines.append(f"     orientation: {arrow}")
    return "\n".join(lines)


def apply_matrix_orientation(matrix, orientation: str = "row-to-col"):
    """Return the matrix in the package's ``M[i, j] == i -> j`` convention.

    ``'col-to-row'`` transposes -- use it for DCM / SPM / transition matrices,
    which store column = source and row = target.
    """
    m = np.asarray(matrix, dtype=float)
    if orientation == "row-to-col":
        return m
    if orientation == "col-to-row":
        return m.T.copy()
    raise ValueError("orientation must be 'row-to-col' or 'col-to-row', "
                     f"got {orientation!r}")


# --------------------------------------------------------------------------
# edges
# --------------------------------------------------------------------------
def extract_directed_edges(matrix, *, edge_threshold: float = 0.0,
                           valid_nodes: Optional[Sequence[int]] = None
                           ) -> List[Dict]:
    """Every off-diagonal cell above threshold, as a directed edge.

    Unlike the undirected path -- which walks only the upper triangle and so
    discards half of an asymmetric matrix -- this reads the full matrix.
    """
    m = np.asarray(matrix, dtype=float)
    n = m.shape[0]
    allowed = None if valid_nodes is None else set(int(v) for v in valid_nodes)
    edges = []
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            w = m[i, j]
            if w == 0 or abs(w) <= edge_threshold:
                continue
            if allowed is not None and (i not in allowed or j not in allowed):
                continue
            edges.append(dict(i=i, j=j, weight=float(w),
                              reciprocal=bool(m[j, i] != 0
                                              and abs(m[j, i]) > edge_threshold)))
    return edges


def _chord_frame(p0, p1, centroid):
    """(tangent, outward radial, lateral, length) for a chord."""
    d = np.asarray(p1, float) - np.asarray(p0, float)
    length = float(np.linalg.norm(d))
    if length <= 0:
        raise ValueError("degenerate edge: endpoints coincide")
    t = d / length
    r = 0.5 * (np.asarray(p0, float) + np.asarray(p1, float)) - centroid
    r = r - t * float(np.dot(r, t))
    if np.linalg.norm(r) < 1e-9:
        r = np.cross(t, np.array([0.0, 0.0, 1.0]))
        if np.linalg.norm(r) < 1e-9:
            r = np.cross(t, np.array([0.0, 1.0, 0.0]))
    r = r / np.linalg.norm(r)
    lat = np.cross(t, r)
    return t, r, lat / np.linalg.norm(lat), length


def edge_polyline(p_src, p_tgt, *, reciprocal: bool, flip: bool, centroid,
                  diag: float, view_dir=None,
                  bow_chord_frac: float = ARROW_DEFAULTS["bow_chord_frac"],
                  bow_floor_frac: float = ARROW_DEFAULTS["bow_floor_frac"],
                  samples: int = ARROW_DEFAULTS["samples"]) -> np.ndarray:
    """Points from source to target.

    One-way edges are always straight. Reciprocal pairs bow into an arc whose
    sagitta is ``max(bow_chord_frac * |chord|, bow_floor_frac * diag)`` -- the
    floor is what keeps very short pairs from collapsing onto each other, and
    the two members bow to opposite sides (``flip`` selects the side).

    ``view_dir`` selects camera-aware bowing: the arc is bowed perpendicular to
    the viewing axis so the pair separates as widely as that projection allows.
    Pass ``None`` for camera-independent geometry.
    """
    p0 = np.asarray(p_src, float)
    p1 = np.asarray(p_tgt, float)
    if not reciprocal:
        ctrl = 0.5 * (p0 + p1)
    else:
        # the frame is built from a canonical ordering so both members of a
        # pair share it and land on opposite sides
        a, b = (p0, p1) if not flip else (p1, p0)
        t, radial, lat, length = _chord_frame(a, b, centroid)
        if view_dir is not None:
            d = np.cross(t, np.asarray(view_dir, float))
            direction = lat if np.linalg.norm(d) < 1e-9 else d / np.linalg.norm(d)
        else:
            direction = lat
        sag = max(bow_chord_frac * length, bow_floor_frac * diag)
        ctrl = 0.5 * (a + b) + direction * (-1.0 if flip else 1.0) * (2.0 * sag)
    s = np.linspace(0.0, 1.0, samples)[:, None]
    return (1 - s) ** 2 * p0 + 2 * (1 - s) * s * ctrl + s ** 2 * p1


def trim_from_end(points: np.ndarray, distance: float):
    """Cut ``distance`` off the end of a polyline.

    Returns ``(trimmed, cut_point, unit_tangent_at_cut)``.
    """
    pts = np.asarray(points, float)
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    target = max(cum[-1] - float(distance), 0.0)
    k = int(np.searchsorted(cum, target))
    k = min(max(k, 1), len(pts) - 1)
    span = cum[k] - cum[k - 1]
    f = 0.0 if span <= 0 else (target - cum[k - 1]) / span
    cut = pts[k - 1] + f * (pts[k] - pts[k - 1])
    tang = pts[k] - pts[k - 1]
    nrm = float(np.linalg.norm(tang))
    tang = tang / nrm if nrm > 0 else np.array([1.0, 0.0, 0.0])
    return np.vstack([pts[:k], cut[None, :]]), cut, tang


# --------------------------------------------------------------------------
# arrowheads
# --------------------------------------------------------------------------
def estimate_units_per_pixel(vertices, width: int, height: int,
                             zoom: float = 1.0) -> float:
    """Data units covered by one screen pixel, approximately.

    Needed because line widths are in **pixels** while cone geometry is in
    **data units**, and the rule "a cone is never narrower than its own line"
    can only be enforced by comparing the two.

    The constant is empirical: with ``aspectmode='data'`` and the default eye
    distance, the scene's largest extent occupies about 84 % of the shorter
    canvas dimension (measured by rendering a mesh and counting pixels --
    159.9 data units spanned 426 px on a 700 px canvas). Perspective makes this
    approximate, which is fine: it only sets the floor on arrowhead size.
    """
    v = np.asarray(vertices, float)
    extent = float(np.max(v.max(axis=0) - v.min(axis=0)))
    px = 0.84 * float(min(width, height)) * float(zoom)
    return extent / px if px > 0 else 0.0


def arrowhead_mesh(tip, direction, length: float, radius: float,
                   segments: int = ARROW_DEFAULTS["segments"]
                   ) -> Tuple[np.ndarray, np.ndarray]:
    """Vertices/faces of one cone whose TIP is at ``tip``."""
    d = np.asarray(direction, float)
    nrm = float(np.linalg.norm(d))
    if nrm <= 0:
        raise ValueError("arrowhead direction must be nonzero")
    d = d / nrm
    tip = np.asarray(tip, float)
    base_c = tip - d * float(length)

    ref = np.array([0.0, 0.0, 1.0])
    if abs(float(np.dot(ref, d))) > 0.9:
        ref = np.array([1.0, 0.0, 0.0])
    e1 = np.cross(d, ref)
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(d, e1)

    th = np.linspace(0.0, 2.0 * np.pi, segments, endpoint=False)
    ring = base_c + float(radius) * (np.cos(th)[:, None] * e1
                                     + np.sin(th)[:, None] * e2)
    verts = np.vstack([tip[None, :], base_c[None, :], ring])
    faces = []
    for k in range(segments):
        a, b = 2 + k, 2 + (k + 1) % segments
        faces.append([0, a, b])      # side
        faces.append([1, b, a])      # base cap
    return verts, np.asarray(faces, dtype=int)


def billboard_arrowhead(tip, axis, view_dir, length: float, radius: float
                        ) -> Tuple[np.ndarray, np.ndarray]:
    """A flat, two-sided triangle turned to face the viewer.

    A 3-D cone collapses to a disc when its edge points at the camera; a
    billboard keeps a triangular silhouette from any angle. Only meaningful for
    a known camera, so it is an export-only mode.
    """
    axis = np.asarray(axis, float)
    axis = axis / np.linalg.norm(axis)
    b = np.cross(axis, np.asarray(view_dir, float))
    if np.linalg.norm(b) < 1e-9:
        b = np.cross(axis, np.array([0.0, 0.0, 1.0]))
        if np.linalg.norm(b) < 1e-9:
            b = np.cross(axis, np.array([0.0, 1.0, 0.0]))
    b = b / np.linalg.norm(b)
    base = np.asarray(tip, float) - axis * float(length)
    verts = np.vstack([tip, base + b * radius, base - b * radius])
    return verts, np.array([[0, 1, 2]], dtype=int)


def darken_color(color: str, factor: float = ARROW_DEFAULTS["darken"]) -> str:
    """Darken a hex or named color so arrowheads read against their lines."""
    named = {
        "red": "#ff0000", "blue": "#0000ff", "green": "#008000",
        "black": "#000000", "gray": "#808080", "grey": "#808080",
        "orange": "#ffa500", "purple": "#800080", "magenta": "#ff00ff",
        "cyan": "#00ffff", "yellow": "#ffff00", "white": "#ffffff",
    }
    c = named.get(str(color).lower(), str(color))
    if not c.startswith("#") or len(c) not in (4, 7):
        return color                      # rgb()/unknown -> leave alone
    if len(c) == 4:
        c = "#" + "".join(ch * 2 for ch in c[1:])
    r, g, b = (int(c[k:k + 2], 16) for k in (1, 3, 5))
    f = float(factor)
    return "#%02x%02x%02x" % (int(r * f), int(g * f), int(b * f))


def arrowheads_to_trace(heads, color: str, name: str = "arrowheads",
                        darken: bool = True):
    """Merge many arrowheads into ONE ``Mesh3d`` trace.

    ``heads`` is a sequence of ``(verts, faces)`` from :func:`arrowhead_mesh`
    or :func:`billboard_arrowhead`. Returns ``None`` when empty.
    """
    import plotly.graph_objects as go

    all_v, all_f, off = [], [], 0
    for verts, faces in heads:
        all_v.append(verts)
        all_f.append(np.asarray(faces) + off)
        off += len(verts)
    if not all_v:
        return None
    V = np.vstack(all_v)
    F = np.vstack(all_f)
    return go.Mesh3d(
        x=V[:, 0], y=V[:, 1], z=V[:, 2],
        i=F[:, 0], j=F[:, 1], k=F[:, 2],
        color=darken_color(color) if darken else color,
        opacity=1.0, flatshading=True,
        lighting=dict(ambient=0.85, diffuse=0.35, specular=0.05),
        hoverinfo="skip", showlegend=False, name=name,
    )


def build_directed_edge_traces(edges, positions, *, widths, hovers,
                               pos_color, neg_color, units_per_px, centroid,
                               diag, node_radius, view_dir=None,
                               width_bucket: float = 0.25, params=None,
                               colors=None, legendgroups=None,
                               show_legend: bool = True):
    """Traces for a directed edge set: bowed/straight lines plus arrowheads.

    Lines are grouped into buckets of similar width rather than one trace per
    edge. A plotly ``Scatter3d`` carries a single ``line.width``, so the
    undirected path averages every edge's width into one number; bucketing
    keeps widths accurate to ``width_bucket`` px while holding the trace count
    to a few dozen instead of one per edge.

    Arrowheads for each sign go into a single ``Mesh3d``, so their sizes stay
    exactly per-edge regardless of bucketing.

    ``colors`` overrides the sign colour per edge and ``legendgroups`` the
    legend group -- both used by the modularity path, where an edge is coloured
    by its source node's module and toggles with that module rather than with
    its sign. ``show_legend=False`` suppresses the legend entries entirely
    (the modularity path's node traces own them).

    Returns ``(traces, counts)`` where counts is ``{'pos': n, 'neg': n}``.
    """
    import plotly.graph_objects as go
    from collections import defaultdict

    p = dict(ARROW_DEFAULTS)
    if params:
        p.update(params)

    groups = defaultdict(lambda: dict(x=[], y=[], z=[], hover=[]))
    heads = {"pos": [], "neg": []}
    counts = {"pos": 0, "neg": 0}

    for idx, e in enumerate(edges):
        i, j, w = e["i"], e["j"], e["weight"]
        sign = "pos" if w > 0 else "neg"
        counts[sign] += 1
        width = float(widths[idx])
        pts = edge_polyline(
            positions[i], positions[j], reciprocal=e["reciprocal"],
            flip=(i > j), centroid=centroid, diag=diag, view_dir=view_dir,
            bow_chord_frac=p["bow_chord_frac"],
            bow_floor_frac=p["bow_floor_frac"], samples=p["samples"])
        edge_len = float(np.linalg.norm(np.asarray(positions[j], float)
                                        - np.asarray(positions[i], float)))
        radius, length = arrowhead_size(
            width, units_per_px, edge_len, k_width=p["k_width"],
            min_radius_px=p["min_radius_px"], slenderness=p["slenderness"],
            max_edge_frac=p["max_edge_frac"])

        # tip stops on the node marker's surface; line stops at the cone base
        _, tip, tang = trim_from_end(pts, node_radius)
        line_pts, _, _ = trim_from_end(pts, node_radius + length)
        heads[sign].append((idx, arrowhead_mesh(tip, tang, length, radius,
                                                segments=p["segments"])))

        key = (
            sign,
            colors[idx] if colors is not None else None,
            legendgroups[idx] if legendgroups is not None else None,
            round(width / width_bucket) * width_bucket,
        )
        g = groups[key]
        g["x"].extend(list(line_pts[:, 0]) + [None])
        g["y"].extend(list(line_pts[:, 1]) + [None])
        g["z"].extend(list(line_pts[:, 2]) + [None])
        g["hover"].extend([hovers[idx]] * len(line_pts) + [""])

    traces = []
    shown = {"pos": False, "neg": False}
    labels = {
        "pos": f"Positive Edges ({counts['pos']} directed)",
        "neg": f"Negative Edges ({counts['neg']} directed)",
    }
    for (sign, ecolor, egroup, width), g in sorted(
            groups.items(), key=lambda kv: -kv[0][3]):
        color = ecolor or (pos_color if sign == "pos" else neg_color)
        group = egroup or f"{sign}_edges"
        legend_now = show_legend and ecolor is None and not shown[sign]
        traces.append(go.Scatter3d(
            x=g["x"], y=g["y"], z=g["z"], mode="lines",
            line=dict(color=color, width=max(width, 0.1)),
            opacity=0.85, hoverinfo="text", hovertext=g["hover"],
            showlegend=legend_now, visible=True,
            name=labels[sign] if ecolor is None else "directed edges",
            legendgroup=group))
        if legend_now:
            shown[sign] = True

    # arrowheads: one Mesh3d per (colour, legendgroup) so they toggle with
    # whatever their lines belong to
    head_groups = defaultdict(list)
    for sign, hs in heads.items():
        for idx, mesh in hs:
            ecolor = colors[idx] if colors is not None else None
            egroup = legendgroups[idx] if legendgroups is not None else None
            head_groups[(sign, ecolor, egroup)].append(mesh)
    for (sign, ecolor, egroup), meshes in head_groups.items():
        color = ecolor or (pos_color if sign == "pos" else neg_color)
        tr = arrowheads_to_trace(meshes, color, name=f"{sign}_arrowheads",
                                 darken=True)
        if tr is not None:
            tr.update(legendgroup=egroup or f"{sign}_edges", showlegend=False)
            traces.append(tr)
    return traces, counts


def arrowhead_size(line_width_px: float, units_per_px: float, edge_length: float,
                   *, k_width: float = ARROW_DEFAULTS["k_width"],
                   min_radius_px: float = ARROW_DEFAULTS["min_radius_px"],
                   slenderness: float = ARROW_DEFAULTS["slenderness"],
                   max_edge_frac: float = ARROW_DEFAULTS["max_edge_frac"]
                   ) -> Tuple[float, float]:
    """(radius, length) in data units for one edge's arrowhead.

    The radius tracks that edge's own line so a thin edge gets a thin head,
    with a pixel floor so the very thinnest still show one. The length is
    capped at a fraction of the edge, because otherwise a short edge gets an
    arrowhead longer than itself -- on the 28-ROI example, a 9.9-unit edge was
    being capped by an 11.7-unit cone.
    """
    radius = max(k_width * 0.5 * float(line_width_px) * units_per_px,
                 float(min_radius_px) * units_per_px)
    length = min(radius / float(slenderness), float(max_edge_frac) * float(edge_length))
    return radius, length
