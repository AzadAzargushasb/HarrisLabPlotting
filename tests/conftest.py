"""Shared fixtures.

The repo had no test suite before these features; fixtures deliberately lean on
the data already committed under test_files/tutorial_files rather than adding
new binary files.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[1]
TF = REPO / "test_files" / "tutorial_files"


@pytest.fixture(scope="session")
def tutorial_files():
    if not TF.is_dir():
        pytest.skip(f"tutorial files not found at {TF}")
    return TF


@pytest.fixture(scope="session")
def mesh(tutorial_files):
    from HarrisLabPlotting import load_mesh_file
    gii = tutorial_files / "brain_mesh.gii"
    if not gii.exists():
        pytest.skip("brain_mesh.gii not available")
    v, f = load_mesh_file(str(gii))
    return np.asarray(v, float), np.asarray(f, int)


@pytest.fixture(scope="session")
def coords28(tutorial_files):
    p = tutorial_files / "output" / "atlas_28_test_comma.csv"
    if not p.exists():
        pytest.skip("28-ROI coordinates not available")
    return pd.read_csv(p)


@pytest.fixture(scope="session")
def coords114(tutorial_files):
    p = tutorial_files / "atlas_114_coordinates.csv"
    if not p.exists():
        pytest.skip("114-ROI coordinates not available")
    return pd.read_csv(p)


@pytest.fixture(scope="session")
def symmetric_28(tutorial_files):
    p = tutorial_files / "node_edge_28" / "connectivity_28.edge"
    if not p.exists():
        pytest.skip("connectivity_28.edge not available")
    return np.loadtxt(p, delimiter="\t")


@pytest.fixture
def directed_28(symmetric_28):
    """A deterministic directed matrix: one-way edges in both triangles,
    reciprocal pairs with unequal weights, and a nonzero diagonal."""
    m = np.triu(symmetric_28, 1)
    out = np.zeros_like(m)
    ei, ej = np.where(m != 0)
    for rank, (i, j) in enumerate(sorted(zip(ei.tolist(), ej.tolist()))):
        w = float(m[i, j])
        if rank % 3 == 0:
            out[i, j] = w                       # one-way, upper
        elif rank % 3 == 1:
            out[j, i] = w                       # one-way, LOWER only
        else:
            out[i, j] = w                       # reciprocal, unequal
            out[j, i] = round(w * 0.45, 6)
    out[0, 0] = 0.31                            # self-loop
    return out
