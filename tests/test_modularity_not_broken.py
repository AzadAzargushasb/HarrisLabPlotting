"""Directed support must not disturb the modularity path.

This is the highest-risk interaction in the change: the modularity function
rebuilds edges per module and deletes traces by NAME, so a new trace with the
wrong name either survives when it should not, or is deleted when it should not
be. These tests pin the behaviour for symmetric matrices (which must be
completely unaffected) and check that every viz_type / edge_color_mode
combination still works once arrows are involved.
"""
import io
import contextlib

import numpy as np
import pytest

from HarrisLabPlotting import create_brain_connectivity_plot_with_modularity as MOD
from HarrisLabPlotting import create_brain_connectivity_plot as PLOT


@pytest.fixture(scope="module")
def modules114(tutorial_files):
    p = tutorial_files / "k5_state_0" / "module_assignments.csv"
    if not p.exists():
        pytest.skip("module assignments not available")
    return str(p)


@pytest.fixture(scope="module")
def matrix114(tutorial_files):
    p = tutorial_files / "k5_state_0" / "connectivity_matrix.csv"
    if not p.exists():
        pytest.skip("k5 connectivity matrix not available")
    return str(p)


@pytest.fixture(scope="module")
def asym114(matrix114, tmp_path_factory):
    m = np.loadtxt(matrix114, delimiter=",")
    rng = np.random.default_rng(0)
    a = m.copy()
    mask = rng.random(m.shape) < 0.35
    a[mask] = a[mask] * 0.4                 # break symmetry deterministically
    p = tmp_path_factory.mktemp("asym") / "asym114.csv"
    np.savetxt(p, a, delimiter=",")
    return str(p)


def _render(**kw):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        fig, stats = MOD(**kw)
    return fig, stats, buf.getvalue()


@pytest.mark.parametrize("viz", ["all", "intra", "inter", "nodes_only"])
@pytest.mark.parametrize("mode", ["module", "sign"])
def test_symmetric_matrix_draws_no_arrows(mesh, coords114, matrix114,
                                          modules114, viz, mode, tmp_path):
    v, f = mesh
    fig, stats, _ = _render(
        vertices=v, faces=f, roi_coords_df=coords114,
        connectivity_matrix=matrix114, module_assignments=modules114,
        viz_type=viz, edge_color_mode=mode,
        save_path=str(tmp_path / "m.html"), no_html=True)
    assert stats["directed"] is False
    assert not [t for t in fig.data
                if t.type == "mesh3d" and "arrowhead" in (t.name or "")]


@pytest.mark.parametrize("viz", ["all", "intra", "inter", "nodes_only"])
@pytest.mark.parametrize("mode", ["module", "sign"])
def test_directed_matrix_renders_in_every_combination(mesh, coords114, asym114,
                                                      modules114, viz, mode,
                                                      tmp_path):
    v, f = mesh
    fig, stats, _ = _render(
        vertices=v, faces=f, roi_coords_df=coords114,
        connectivity_matrix=asym114, module_assignments=modules114,
        viz_type=viz, edge_color_mode=mode,
        save_path=str(tmp_path / "m.html"), no_html=True)
    assert stats["directed"] is True
    heads = [t for t in fig.data
             if t.type == "mesh3d" and "arrowhead" in (t.name or "")]
    if viz == "nodes_only":
        assert not heads               # no edges at all -> no arrowheads
    else:
        assert heads


def test_module_legend_entries_survive(mesh, coords114, asym114, modules114,
                                       tmp_path):
    """Clicking a module must still hide that module's nodes AND its arrows,
    which requires both to share the module_<id> legendgroup."""
    v, f = mesh
    fig, _, _ = _render(
        vertices=v, faces=f, roi_coords_df=coords114,
        connectivity_matrix=asym114, module_assignments=modules114,
        edge_color_mode="module", save_path=str(tmp_path / "m.html"),
        no_html=True)
    legend_entries = [t for t in fig.data if getattr(t, "showlegend", False)]
    assert legend_entries, "module legend entries disappeared"
    node_groups = {t.legendgroup for t in legend_entries}
    arrow_groups = {t.legendgroup for t in fig.data
                    if t.type == "mesh3d" and "arrowhead" in (t.name or "")}
    assert arrow_groups, "no arrowheads emitted"
    assert arrow_groups <= node_groups, (
        f"arrows in groups {arrow_groups - node_groups} have no legend entry")


def test_brain_surface_is_never_deleted(mesh, coords114, asym114, modules114,
                                        tmp_path):
    """The per-module teardown deletes traces by name; the mesh must survive."""
    v, f = mesh
    fig, _, _ = _render(
        vertices=v, faces=f, roi_coords_df=coords114,
        connectivity_matrix=asym114, module_assignments=modules114,
        save_path=str(tmp_path / "m.html"), no_html=True)
    assert [t for t in fig.data if t.name == "Brain Surface"]


def test_symmetry_reported_once_not_twice(mesh, coords114, asym114, modules114,
                                          tmp_path):
    """The inner delegation must not print its own (misleading) report."""
    v, f = mesh
    _, _, out = _render(
        vertices=v, faces=f, roi_coords_df=coords114,
        connectivity_matrix=asym114, module_assignments=modules114,
        save_path=str(tmp_path / "m.html"), no_html=True)
    assert out.count("Matrix symmetry") == 1


def test_viz_type_filtering_still_applies_to_arrows(mesh, coords114, asym114,
                                                    modules114, tmp_path):
    """intra and inter must draw strictly fewer arrows than all."""
    v, f = mesh
    counts = {}
    for viz in ("all", "intra", "inter"):
        _, _, out = _render(
            vertices=v, faces=f, roi_coords_df=coords114,
            connectivity_matrix=asym114, module_assignments=modules114,
            viz_type=viz, save_path=str(tmp_path / "m.html"), no_html=True)
        line = [l for l in out.splitlines() if "directed:" in l and "arrows" in l]
        counts[viz] = int(line[0].split()[1]) if line else 0
    assert counts["intra"] < counts["all"]
    assert counts["inter"] < counts["all"]
    assert counts["intra"] + counts["inter"] == counts["all"]


def test_pvalue_mode_works_directed(mesh, coords28, tutorial_files, tmp_path):
    pv = tutorial_files / "node_edge_28" / "pvalues_28.csv"
    if not pv.exists():
        pytest.skip("p-value matrix not available")
    p = np.loadtxt(pv, delimiter=",")
    rng = np.random.default_rng(1)
    a = p.copy()
    mask = rng.random(p.shape) < 0.3
    a[mask] = np.clip(a[mask] * 0.3, 1e-6, 1.0)
    ap = tmp_path / "asym_p.csv"
    np.savetxt(ap, a, delimiter=",")
    v, f = mesh
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        fig, stats = PLOT(
            vertices=v, faces=f, roi_coords_df=coords28,
            connectivity_matrix=str(ap), matrix_type="pvalue",
            pvalue_threshold=0.05, save_path=str(tmp_path / "p.html"),
            no_html=True)
    assert stats["directed"] is True
    assert [t for t in fig.data
            if t.type == "mesh3d" and "arrowhead" in (t.name or "")]
