# Batch processing

When you're rendering the same figure for every subject in a study, every
state in a sliding-window analysis, or every condition in an experiment,
calling `hlplot plot` once per output gets old fast. `hlplot batch` reads a
YAML config and runs many plots from one command.

## When to use it

- **Per-subject grids** — one plot per participant, identical layout
- **Per-state animations** — one plot per dynamic-connectivity state
- **Format sweeps** — render the same plot to HTML, PNG, and SVG in one go
- **Reproducible figures** — config-driven plotting is auditable: the YAML
  *is* the spec

## Minimal config

```yaml
# minimal.yaml
mesh_file: "test_files/tutorial_files/brain_mesh.gii"
roi_coords_file: "test_files/tutorial_files/atlas_114_coordinates.csv"
connectivity_matrix: "test_files/tutorial_files/k5_state_0/connectivity_matrix.csv"

output_dir: "./batch_out"
output_format: "html"

plot:
  title: "State 0 — Baseline"
  node_size: 10
  edge_threshold: 0.2
```

Run it:

```bash
hlplot batch --config minimal.yaml
```

This writes one HTML to `./batch_out/`.

## Generate a starter config

```bash
hlplot config init --output my_config.yaml
```

That drops a fully-commented template into `my_config.yaml` showing every
option with its default.

## Batch over multiple inputs

The `batch:` list at the bottom of the config sweeps over runs. Each item
overrides any top-level setting:

```yaml
mesh_file: "test_files/tutorial_files/brain_mesh.gii"
roi_coords_file: "test_files/tutorial_files/atlas_114_coordinates.csv"
output_dir: "./batch_out"
output_format: "html"

plot:
  node_size: 10
  edge_threshold: 0.2

batch:
  - name: "state_0"
    connectivity_matrix: "test_files/tutorial_files/k5_state_0/connectivity_matrix.csv"
  - name: "state_1"
    connectivity_matrix: "test_files/tutorial_files/k5_state_1/connectivity_matrix.csv"
  - name: "state_2"
    connectivity_matrix: "test_files/tutorial_files/k5_state_2/connectivity_matrix.csv"
```

Output files are named `state_0.html`, `state_1.html`, `state_2.html` (the
`name:` field is the basename).

## Modularity batch

To run the modular variant per item, add a `modularity:` block:

```yaml
modularity:
  enabled: true
  edge_color_mode: "module"

batch:
  - name: "state_0"
    connectivity_matrix: ".../k5_state_0/connectivity_matrix.csv"
    modularity:
      module_file: ".../k5_state_0/module_assignments.csv"
      q_score: 0.41
      z_score: 8.7
  - name: "state_1"
    connectivity_matrix: ".../k5_state_1/connectivity_matrix.csv"
    modularity:
      module_file: ".../k5_state_1/module_assignments.csv"
      q_score: 0.38
      z_score: 7.9
```

Per-item `q_score` / `z_score` show up in each figure's title automatically.

## Format sweep

Override `output_format` at the CLI to render every plot in the config to a
different format:

```bash
hlplot batch --config my_config.yaml --format png
hlplot batch --config my_config.yaml --format svg
hlplot batch --config my_config.yaml --format pdf
```

This is the fastest way to produce a paper-ready set: render once to HTML
for the supplement, once to SVG for the main figure.

## Parallel and dry-run

```bash
hlplot batch --config my_config.yaml --parallel
hlplot batch --config my_config.yaml --dry-run
```

`--parallel` runs the items concurrently (one process per CPU). `--dry-run`
shows what *would* be rendered without actually rendering — useful for
debugging path errors in a large config before you commit to a long build.

## Camera options in the config

```yaml
camera:
  view: "anterolateral_left"     # any preset name
  # ── or ──
  eye: [1.5, 0.0, 1.0]            # custom angle
  center: [0, 0, 0]
  up: [0, 0, 1]
```

If you specify `eye` it overrides `view`. See
[Customize the camera view](../how_to/customize_camera.md) for the full
list of presets.

## See also

- Generate the YAML scaffold: `hlplot config init --output my.yaml`
- Per-subject pipeline example: [CLI walkthrough §15 Complete pipeline](cli_walkthrough.md#complete-pipeline-example)
