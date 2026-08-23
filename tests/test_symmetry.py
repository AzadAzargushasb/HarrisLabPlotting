"""Symmetry detection -- the check that decides whether arrows are drawn."""
import numpy as np
import pytest

from HarrisLabPlotting.directed import check_matrix_symmetry, format_symmetry_report


def test_symmetric_matrix_reads_symmetric(symmetric_28):
    r = check_matrix_symmetry(symmetric_28)
    assert r["is_symmetric"] and not r["directed"]
    assert r["max_asymmetry"] == 0.0
    assert r["n_oneway"] == 0


def test_csv_float_noise_is_still_symmetric(symmetric_28):
    """A matrix round-tripped through CSV differs from its transpose by ~1e-16.
    Reading that as 'directed' would put arrows on every existing figure."""
    rng = np.random.default_rng(0)
    noisy = symmetric_28 + rng.normal(0, 1e-14, symmetric_28.shape)
    noisy = (noisy + noisy.T) / 2
    noisy[0, 1] += 3e-14
    assert check_matrix_symmetry(noisy)["is_symmetric"]


def test_tolerance_can_be_tightened(symmetric_28):
    noisy = symmetric_28.copy()
    noisy[0, 1] += 1e-9
    assert check_matrix_symmetry(noisy)["is_symmetric"]
    assert not check_matrix_symmetry(noisy, tol=1e-12)["is_symmetric"]


def test_directed_counts(directed_28):
    r = check_matrix_symmetry(directed_28)
    assert r["directed"]
    off = directed_28.copy()
    np.fill_diagonal(off, 0)
    assert r["n_edges"] == int(np.count_nonzero(off))
    assert r["n_edges"] == 2 * r["n_reciprocal"] + r["n_oneway"]
    assert r["n_reciprocal"] > 0 and r["n_oneway"] > 0


def test_diagonal_is_counted_but_excluded_from_edges(directed_28):
    r = check_matrix_symmetry(directed_28)
    assert r["n_diagonal"] == 1
    off = directed_28.copy()
    np.fill_diagonal(off, 0)
    assert r["n_edges"] == int(np.count_nonzero(off))


def test_report_states_orientation(directed_28):
    txt = format_symmetry_report(check_matrix_symmetry(directed_28),
                                 drawing_directed=True)
    assert "ASYMMETRIC" in txt
    assert "row = source -> column = target" in txt
    assert "self-loops" in txt

    txt2 = format_symmetry_report(
        check_matrix_symmetry(directed_28, orientation="col-to-row"),
        drawing_directed=True)
    assert "column = source -> row = target" in txt2


def test_non_square_rejected():
    with pytest.raises(ValueError):
        check_matrix_symmetry(np.zeros((3, 4)))
