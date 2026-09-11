# Affine Diffusions

Simulation and estimation of Affine Diffusion models.

Install:

```shell
pip install affidiff
```

## Contribute

### Setup

Install project in editable mode and sync all dependencies:

```shell
uv sync --all-groups
```

### Build

The project uses [scikit-build-core](https://scikit-build-core.readthedocs.io/) with CMake to compile Cython extensions. The build process is automatic during installation, but you can manually trigger a build:

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

This runs:
- Code formatting and linting (ruff)
- Type checking (ty)
- YAML validation
- Common checks (trailing whitespace, end-of-file fixers, etc.)

### Testing

Run the test suite:

```shell
uv run pytest
```
