# Static image export

For papers, posters, and slide decks you usually want a static image, not an
HTML file. HarrisLabPlotting exports PNG, SVG, and PDF via
[`kaleido`](https://github.com/plotly/Kaleido) — no browser required.

For a full CLI demonstration, see
[CLI walkthrough §8–9](cli_walkthrough.md#8-static-image-exports).

## Format is selected by the output extension

```bash
hlplot plot ... --output figure.png   # raster
hlplot plot ... --output figure.svg   # vector
hlplot plot ... --output figure.pdf   # vector
```

The matrix/coords/mesh flags are identical to the interactive case; only the
output extension changes.

## DPI and dimensions

```bash
hlplot plot ... \
  --output figure.png \
  --image-width 2000 \
  --image-height 1600 \
  --image-dpi 300
```

`--image-dpi` sets the static-export resolution. For print figures, 300 DPI
at the size you'll publish at is the standard.

## Export canvas size (`--export-size`) — keep it square

Single-image exports render on a **square 1200×1200 canvas by default**. That does
two things:

- **Even margins** on the left and right, so the brain sits centered with balanced
  whitespace rather than being cropped tight against the frame.
- **A DPI-stable 3D aspect.** This is the important one: on a **non-square** canvas
  kaleido renders the 3D scene with a *scale-dependent* aspect, so the brain's
  proportions visibly change as you raise `--image-dpi` (the "squished at high DPI"
  problem). On a square canvas the aspect stays put at any DPI.

```bash
hlplot plot ... --export-image brain.png                       # 1200x1200 (default)
hlplot plot ... --export-image brain.png --export-size "1600,1600"
```

:::{warning}
**Keep width == height if you intend to change `--image-dpi`.** `--image-dpi` is a
supersampling factor (`min(dpi/72, 8)`) applied to this canvas, and on a non-square
canvas the rendered 3D aspect shifts between DPIs. A non-square `--export-size`
prints a warning for this reason.
:::

Python: `export_size=(1200, 1200)` on `create_brain_connectivity_plot` /
`create_brain_connectivity_plot_with_modularity`.

## Tight crops (`--export-autocrop`)

If you want the figure trimmed tight to its content instead of the even margins,
opt in with `--export-autocrop` (Python `export_autocrop=True`). It is **off by
default**. It is a pure crop — the aspect is never stretched — but note the output
dimensions then depend on the content rather than being a fixed canvas.

```bash
hlplot plot ... --export-image brain.png --export-autocrop   # trim to content
```

SVG/PDF exports are vector and are never auto-cropped. The multi-view stitched
export already crops each panel, so it is unaffected by this flag.

## "Clean" exports for publications

Strip the title and legend so the figure drops straight into a manuscript
panel:

```bash
hlplot plot ... \
  --output panel.svg \
  --hide-title \
  --hide-legend
```

```{interactive-plot}
:image: images/cli_tutorial/09a_clean.png
:caption: Clean export — no title, no legend. Panel-ready.
:height: 480
```

The same flags apply to `hlplot modular`.

## Custom or transparent background

By default the figure background is white. `--background-color` sets any
background — a named color, a hex code, or `transparent` for a transparent PNG
that drops onto any slide or poster. It applies to **both** the saved
interactive HTML and the static export.

```bash
# Transparent PNG (real RGBA alpha channel)
hlplot plot ... --export-image brain.png --background-color transparent

# Any named color or hex
hlplot plot ... --export-image brain.png --background-color "#1e1e1e"
```

```{interactive-plot}
:image: images/static_export/transparent_bg_demo.png
:caption: A transparent-background export composited over a checkerboard so the alpha channel is visible. Transparency also works for multi-view stitched strips (written as RGBA), and the same flag exists on hlplot modular.
:height: 420
```

Python: pass `background_color="transparent"` (or a color/hex) to
`create_brain_connectivity_plot` / `create_brain_connectivity_plot_with_modularity`.

## Multi-view stitched PNG strips

For a three- or five-panel figure that shows the network from multiple
camera angles in one PNG, use the multi-view export. This is the most
efficient way to produce a publication-ready figure that conveys the full
3D structure on paper.

```bash
hlplot plot ... \
  --multi-view-output panels.png \
  --multi-view-views "left,superior,right" \
  --multi-view-panel-width 800 \
  --multi-view-panel-height 800 \
  --image-dpi 300
```

```{interactive-plot}
:image: images/multi_view/cli_mv_default.png
:html: plots/cli_mv_default_dummy.html
:caption: 3-view stitched PNG (left / superior / right) at 300 DPI.
:height: 360
```

For the full multi-view recipe — including custom camera dicts, panel
labels, and the standalone `export_multi_view_stitched_png` Python helper —
see [Legends & multi-view](legends_and_multiview.md).

## Python equivalent

`Plotly.Figure.write_image` is the underlying call:

```python
fig.write_image("figure.png", width=1600, height=1200, scale=2)
```

`scale` is the multiplier that produces high-DPI output. `scale=2` on an
1600 × 1200 figure yields a 3200 × 2400 PNG.
