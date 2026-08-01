"""
Generate a SECOND p-value matrix for the 28-ROI tutorial network, with
significance spread evenly across several orders of magnitude.

Why this exists
---------------
The original `node_edge_28/pvalues_28.csv` was generated with a noise term, and
its surviving p-values bunch up just under the 0.05 threshold:

    5e-05, 6e-05, 6e-05, 0.00085, 0.0018, 0.0033, 0.0038, 0.0042, 0.0094,
    0.017, 0.019, 0.020, 0.022, 0.022, 0.032, 0.041, 0.043, 0.043, 0.046, 0.047
                                        ^^^ 11 of 20 crammed here

Because edge width scales with -log10(p), that clustering makes almost every edge
render at a similar thin width, with only the three outliers standing out. It is
a poor demonstration of "edge width encodes significance".

This script writes a NEW file (it does NOT touch pvalues_28.csv, which the p-value
tutorial still uses) in which the edges are ranked by their connection strength in
`connectivity_28.edge` and then assigned LOG-SPACED p-values:

  * the top N_SIGNIFICANT edges get p log-spaced over [P_MIN, P_MAX_SIG]
    (all survive a 0.05 threshold, spanning ~4.6 orders of magnitude), and
  * the remaining edges get p log-spaced over [P_MIN_NS, P_MAX_NS]
    (all dropped by a 0.05 threshold, so the thresholding demo still works).

Stronger connection => smaller p, so significance still lines up with where the
real connections are. Fully deterministic: no RNG, same output every run.

Outputs (committed, alongside the originals):
  node_edge_28/pvalues_28_spread.csv
  node_edge_28/pvalues_28_spread.npy

The existing `pvalues_28_signs.csv` applies unchanged: this matrix keeps exactly
the same edge topology as connectivity_28.edge.

Run with the project env:
  /home/aazarg/.conda/envs/pre_env/bin/python generate_pvalue_spread.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
NE28 = HERE.parent / "node_edge_28"
EDGE = NE28 / "connectivity_28.edge"

# ----- Tunable knobs --------------------------------------------------------
N_SIGNIFICANT = 20        # how many edges should survive a p <= 0.05 threshold
P_MIN = 1e-6              # p of the strongest connection
P_MAX_SIG = 0.045         # p of the weakest SURVIVING connection (< 0.05)
P_MIN_NS = 0.08           # p of the strongest DROPPED connection (> 0.05)
P_MAX_NS = 0.9            # p of the weakest dropped connection
THRESHOLD = 0.05          # the threshold this file is designed around


def main():
    mat = np.loadtxt(EDGE, delimiter="\t")
    n = mat.shape[0]

    # Collect the upper-triangle edges that actually exist, strongest first.
    iu = np.triu_indices(n, 1)
    weights = np.abs(mat[iu])
    present = np.where(weights > 0)[0]
    order = present[np.argsort(-weights[present])]      # descending |weight|
    n_edges = len(order)
    n_sig = min(N_SIGNIFICANT, n_edges)
    n_ns = n_edges - n_sig
    print(f"{n_edges} edges in {EDGE.name}: {n_sig} -> significant, {n_ns} -> dropped")

    # Log-spaced p-values: strongest edge gets P_MIN, weakest survivor P_MAX_SIG.
    p_sig = np.logspace(np.log10(P_MIN), np.log10(P_MAX_SIG), n_sig)
    p_ns = (np.logspace(np.log10(P_MIN_NS), np.log10(P_MAX_NS), n_ns)
            if n_ns > 0 else np.array([]))
    p_ranked = np.concatenate([p_sig, p_ns])

    P = np.ones((n, n), dtype=float)                     # no edge -> p = 1
    for rank, idx in enumerate(order):
        i, j = iu[0][idx], iu[1][idx]
        P[i, j] = P[j, i] = p_ranked[rank]
    np.fill_diagonal(P, 1.0)

    np.save(NE28 / "pvalues_28_spread.npy", P)
    pd.DataFrame(P).to_csv(NE28 / "pvalues_28_spread.csv", index=False, header=False)

    # Report the spread vs the original, so the improvement is visible.
    surviving = np.sort(P[iu][(P[iu] > 0) & (P[iu] <= THRESHOLD)])
    w = -np.log10(surviving)
    print(f"\nsurviving edges (p <= {THRESHOLD}): {len(surviving)}")
    print("p-values:", [f"{v:.2g}" for v in surviving])
    print(f"-log10(p) range: {w.min():.2f} .. {w.max():.2f}  "
          f"(quartiles {np.percentile(w, [25, 50, 75]).round(2).tolist()})")
    print(f"\nwrote {NE28/'pvalues_28_spread.csv'}")
    print(f"wrote {NE28/'pvalues_28_spread.npy'}")


if __name__ == "__main__":
    main()
