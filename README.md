# Affine Diffusions

![pytest](https://github.com/khrapovs/diffusions/actions/workflows/workflow.yaml/badge.svg)
[![!pypi](https://img.shields.io/pypi/v/affidiff)](https://pypi.org/project/affidiff)
[![!python-versions](https://img.shields.io/pypi/pyversions/diffusions)](https://pypi.org/project/affidiff)

Simulation and estimation of Affine Diffusion models.

## Installation

```shell
pip install affidiff
```

## Interactive UI

A lightweight web-based interface for exploring affine diffusion model simulations.
After installation, start the interactive simulator:

```shell
ui
```

This opens a Streamlit app in your browser at `http://localhost:8501`.

Features:

- Select from 5 affine diffusion models (GBM, Vasicek, CIR, Heston, CentTend)
- Adjust model parameters and see live preview
- Control simulation length and random seed for reproducibility
- Smart innovation reuse: change parameters to see their isolated effect without noise interference
- Interactive Plotly charts for visualization

## Documentation

[khrapovs.github.io/diffusions](https://khrapovs.github.io/diffusions/)

## Contribute

### Setup

Install project in editable mode and sync all dependencies:

```shell
uv sync --all-groups
```

### Build

The project uses [scikit-build-core](https://scikit-build-core.readthedocs.io/) with CMake to compile Cython extensions.
The build process is automatic during installation, but you can manually trigger a build:

```shell
uv build
```

This compiles the Cython simulation module (`src/affidiff/simulate.pyx` → `.c` → `.so` extension).

### Code Quality

Use pre-commit to automatically format and lint code:

```shell
uv run prek install
uv run prek run --all-files
```

### Testing

Run the test suite:

```shell
uv run pytest
```
