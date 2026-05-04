# Installation

HarrisLabPlotting requires Python 3.9 or newer.

## With conda (recommended)

The conda environment ships with all dependencies pinned to versions known
to work together — the simplest path on every platform.

```bash
git clone https://github.com/AzadAzargushasb/HarrisLabPlotting.git
cd HarrisLabPlotting
conda env create -f environment.yml
conda activate harris_lab_plotting
pip install -e .
```

After this, `hlplot --version` should print `1.0.0`.

## With pip only

If you already have a working scientific-Python environment:

```bash
git clone https://github.com/AzadAzargushasb/HarrisLabPlotting.git
cd HarrisLabPlotting
pip install -e .
```

This pulls all required dependencies from PyPI:

- numpy, pandas, scipy, networkx
- nibabel
- plotly, kaleido, pillow
- click, pyyaml, rich

## Optional extras

Install groups beyond the runtime baseline:

```bash
pip install -e ".[dev]"      # pytest, black, flake8, mypy
pip install -e ".[jupyter]"  # jupyter, jupyterlab
pip install -e ".[docs]"     # to build the docs site locally
```

## Verify the install

```bash
hlplot --version
hlplot plot --help
python -c "import HarrisLabPlotting; print(HarrisLabPlotting.__file__)"
```

## Sample data

The repo ships with `test_files/tutorial_files/` — atlas, mesh, ROI
coordinates, and connectivity matrices used throughout this documentation.
You don't need to download anything separately.

```bash
ls test_files/tutorial_files/
```

## Common install issues

- **`kaleido` missing on Linux** — `sudo apt-get install libgbm1` then
  `pip install --upgrade kaleido`.
- **NumPy AttributeError on import** — pin to `"numpy>=1.21,<2.1"`.
- **`hlplot` not found after install** — your environment's `bin/` isn't
  on `$PATH`; activate the env or run via `python -m HarrisLabPlotting`.

Full troubleshooting catalog: [Troubleshooting](../about/troubleshooting.md).

## Next

- Five-minute quickstart: [Quickstart](quickstart.md)
- Mental model of the three required inputs: [Concepts](concepts.md)
