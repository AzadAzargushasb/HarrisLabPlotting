# Threshold a connectivity matrix

Dense connectivity matrices look like spaghetti. Three thresholding tools
in `hlplot utils` keep the strongest signal:

## At plot time

```bash
hlplot plot ... --edge-threshold 0.3
```

`--edge-threshold` drops edges whose absolute weight is below the cutoff.
Cheap, reversible — re-run with a different value to compare.

## Persistent threshold (keep top N edges)

```bash
hlplot utils threshold \
  --matrix connectivity_matrix.csv \
  --output thresh_top200.npy \
  --top-n 200
```

This zeroes out everything except the top 200 edges by absolute value, so
downstream tools see a much sparser matrix.

## Other thresholding modes

```bash
# Keep edges in the top 5% of |weight|
hlplot utils threshold --matrix conn.csv --output thresh.npy --percentile 95

# Keep edges with |weight| ≥ 0.4
hlplot utils threshold --matrix conn.csv --output thresh.npy --absolute 0.4
```

The output is a thresholded copy. The original is never modified.

## In Python

```python
from HarrisLabPlotting import threshold_matrix_top_n, filter_matrix_by_sign

matrix = np.load("connectivity_matrix.npy")

# top 200 edges by |weight|
top200 = threshold_matrix_top_n(matrix, n=200)

# keep only positive edges
pos_only = filter_matrix_by_sign(matrix, sign="positive")
```

## See also

- Inspect the matrix before thresholding:
  `hlplot utils info --matrix connectivity_matrix.csv`
- For p-values, threshold by significance instead:
  [P-value plotting §5](../tutorials/pvalue_plotting.md)