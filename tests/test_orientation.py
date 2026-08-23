"""Orientation: the failure that silently reverses every arrow."""
import numpy as np
import pytest

from HarrisLabPlotting.directed import (
    apply_matrix_orientation, extract_directed_edges,
)


def test_default_is_row_to_col():
    """M[i, j] must mean i -> j."""
    m = np.zeros((3, 3))
    m[0, 1] = 1.0
    e = extract_directed_edges(apply_matrix_orientation(m, "row-to-col"))
    assert len(e) == 1
    assert (e[0]["i"], e[0]["j"]) == (0, 1)


def test_col_to_row_flips_the_arrow():
    """A DCM-style matrix must come out pointing the other way."""
    m = np.zeros((3, 3))
    m[0, 1] = 1.0
    e = extract_directed_edges(apply_matrix_orientation(m, "col-to-row"))
    assert (e[0]["i"], e[0]["j"]) == (1, 0)


def test_col_to_row_is_exactly_the_transpose(directed_28):
    assert np.array_equal(apply_matrix_orientation(directed_28, "col-to-row"),
                          directed_28.T)


def test_round_trip_is_lossless(directed_28):
    once = apply_matrix_orientation(directed_28, "col-to-row")
    twice = apply_matrix_orientation(once, "col-to-row")
    assert np.array_equal(twice, directed_28)


def test_orientation_does_not_mutate_the_caller(directed_28):
    before = directed_28.copy()
    apply_matrix_orientation(directed_28, "col-to-row")
    assert np.array_equal(directed_28, before)


def test_bad_orientation_rejected():
    with pytest.raises(ValueError):
        apply_matrix_orientation(np.zeros((2, 2)), "sideways")
