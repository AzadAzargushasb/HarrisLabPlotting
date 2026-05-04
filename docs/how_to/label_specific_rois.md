# Label specific ROIs

By default `hlplot` labels every ROI, which gets crowded on networks larger
than ~30 nodes. You can switch labels off, label only specific ROIs, or
label only the highest-degree hubs.

For the full demonstration with screenshots, see
[CLI walkthrough §14](../tutorials/cli_walkthrough.md#14-selectively-labelling-rois).

## Off entirely

```bash
hlplot plot ... --show-node-labels none
```

## A short list of ROIs by name

```bash
hlplot plot ... --show-node-labels "rACC_L,LIPS_L,LIPS_R,SPL_L,SPL_R"
```

The names must match `roi_name` values in your coordinates CSV exactly.

## A short list by ROI index

Wrap integers in brackets so the parser doesn't read them as names:

```bash
hlplot plot ... --show-node-labels "[1,5,9,42]"
```

## Top-K hubs by degree

`hubs:N` labels the N most-connected nodes (by absolute weighted degree).
This is the right default for "show the most important ROIs":

```bash
hlplot plot ... --show-node-labels hubs:10
```

```{interactive-plot}
:image: images/cli_tutorial/14c_labels_hubs.png
:caption: Top-10 hub labels only — readable even on a 114-ROI network.
:height: 480
```

## In Python

`create_brain_connectivity_plot` and friends accept `show_labels` as a list,
boolean, integer (top-K hubs), or string (`"all"`, `"none"`, `"hubs:N"`):

```python
fig, _ = create_brain_connectivity_plot(
    ..., show_labels="hubs:10",
)
```

## See also

- Hover tooltips with metric values: pass `--node-metrics` (and see
  [CLI walkthrough §5](../tutorials/cli_walkthrough.md#5-114-roi-network-with-metrics))
- Custom font size: `--label-font-size 8`
