"""
Build the deterministic DIRECTED 28x28 demo matrix used by the directed-graph
tutorial.

No RNG -- the recipe is index-driven, so the file is byte-reproducible. The
structure exercises every case the renderer has to handle:

  * one-way edges stored in the UPPER triangle only        (i -> j)
  * one-way edges stored in the LOWER triangle only        (j -> i)
      -- the ones an upper-triangle-only loop silently drops
  * reciprocal pairs with UNEQUAL weights                  (i <-> j)
      -- forced to include the shortest and longest node pairs, so the arc
         geometry is stressed at both extremes
  * negative weights (inherited from the source matrix)
  * a nonzero diagonal, to exercise the "ignored, but reported" path

Outputs (into node_edge_28/, alongside the other shipped fixtures):
  directed_28.csv        the matrix
  directed_28_roles.csv  per-edge roles + inter-node distance

Run with the project env:
  /home/aazarg/.conda/envs/pre_env/bin/python generate_directed_demo.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
TF = HERE.parent
EDGE = TF / "node_edge_28" / "connectivity_28.edge"
COORDS = TF / "output" / "atlas_28_test_comma.csv"
OUT_DIR = TF / "node_edge_28"

REVERSE_RATIO = 0.45
N_FORCED_SHORT = 3
N_FORCED_LONG = 3
SELF_LOOPS = {0: 0.31, 7: 0.52, 16: 0.18}


def main() -> None:
    src = np.loadtxt(EDGE, delimiter="\t")
    n = src.shape[0]
    coords = pd.read_csv(COORDS)
    xyz = coords[["cog_x", "cog_y", "cog_z"]].to_numpy(float)
    names = coords["roi_name"].tolist()
    dist = np.linalg.norm(xyz[:, None, :] - xyz[None, :, :], axis=2)

    ei, ej = np.where(np.triu(src, 1) != 0)
    pairs = sorted(zip(ei.tolist(), ej.tolist()))

    iu = np.triu_indices(n, 1)
    by_dist = [(int(iu[0][k]), int(iu[1][k])) for k in np.argsort(dist[iu])]
    existing = set(pairs)
    forced_short = [p for p in by_dist if p not in existing][:N_FORCED_SHORT]
    forced_long = [p for p in by_dist[::-1] if p not in existing][:N_FORCED_LONG]

    M = np.zeros((n, n), float)
    rows = []

    def note(i, j, role, w_fwd, w_rev=0.0):
        rows.append(dict(source=names[i], target=names[j], role=role,
                         w_forward=w_fwd, w_reverse=w_rev,
                         distance=float(dist[i, j])))

    for rank, (i, j) in enumerate(pairs):
        w = float(src[i, j])
        mode = rank % 3
        if mode == 0:
            M[i, j] = w
            note(i, j, "oneway_upper", w)
        elif mode == 1:
            M[j, i] = w
            note(j, i, "oneway_lower", w)
        else:
            M[i, j] = w
            M[j, i] = round(w * REVERSE_RATIO, 6)
            note(i, j, "reciprocal", w, M[j, i])

    for k, (i, j) in enumerate(forced_short):
        w = 0.80 - 0.10 * k
        M[i, j], M[j, i] = w, round(w * REVERSE_RATIO, 6)
        note(i, j, "reciprocal_short", w, M[j, i])
    for k, (i, j) in enumerate(forced_long):
        w = 0.75 - 0.10 * k
        M[i, j], M[j, i] = w, round(w * REVERSE_RATIO, 6)
        note(i, j, "reciprocal_long", w, M[j, i])

    for idx, val in SELF_LOOPS.items():
        M[idx, idx] = val

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    np.savetxt(OUT_DIR / "directed_28.csv", M, delimiter=",", fmt="%.6f")
    pd.DataFrame(rows).sort_values(["role", "distance"]).to_csv(
        OUT_DIR / "directed_28_roles.csv", index=False)

    from HarrisLabPlotting.directed import (
        check_matrix_symmetry, format_symmetry_report,
    )
    print(f"wrote {OUT_DIR / 'directed_28.csv'}")
    print(format_symmetry_report(check_matrix_symmetry(M)))
    print(pd.DataFrame(rows).groupby("role").agg(
        n=("role", "size"), dmin=("distance", "min"),
        dmax=("distance", "max")).round(1).to_string())


if __name__ == "__main__":
    main()
