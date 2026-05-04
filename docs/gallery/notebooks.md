# Notebook gallery

Three Jupyter notebooks that exercise the package end-to-end. The first
two are rendered inline below — you can read every cell, every output, and
every plot without leaving the page. The third (the legend & multi-view
notebook) is too large to embed inline; download it from GitHub.

::::{grid} 1 2 2 3
:gutter: 3

:::{grid-item-card} 🚀 Quickstart notebook
:link: notebooks/quickstart
:link-type: doc
End-to-end: load mesh + coords + matrix, build a connectivity plot, color by module, export. **60 KB.**
:::

:::{grid-item-card} 📊 P-value plotting
:link: notebooks/pvalue_plotting
:link-type: doc
Threshold a permutation-test p-value matrix, render with `--pvalue-mode`, signed and unsigned. **38 KB.**
:::

:::{grid-item-card} 🎨 Legend + multi-view
:link: https://github.com/AzadAzargushasb/HarrisLabPlotting/blob/main/tutorial/legend%20key%20and%203%20view%20display%20test.ipynb
The full legend-keys / multi-view export demo. **28 MB** — view on GitHub.
:::
::::

## Why some notebooks are inline and others aren't

The two inline notebooks have small, vector-style cell outputs that render
fast in the browser. The legend & multi-view notebook embeds many full
Plotly figures (~4 MB each), which is great for offline use but too heavy
for a docs page. The same content is covered by the rendered tutorial:
[Legends & multi-view](../tutorials/legends_and_multiview.md).

```{toctree}
:hidden:

notebooks/quickstart
notebooks/pvalue_plotting
```
