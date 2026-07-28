# Purpose

`src/` is the root container directory for the `affidiff` Python production package.

# Ownership

Owns all production Python source modules, Cython extension code, and package initialization logic under `src/`.

# Local Contracts

- All production code must be placed inside the `src/affidiff` package directory.
- `src/affidiff/__init__.py` defines the public package API.
- All classes and functions within production modules must expose only methods and attributes used externally; internal helpers, attributes, and methods must be private (`_name`).

# Work Guidance

- Follow strict typing and docstring standards established in `pyproject.toml` (Ruff/Numpy docstring conventions).
- Keep public API exports clean and aligned with `affidiff/__init__.py`.

# Verification

- Run test suite: `uv run pytest`
- Run pre-commit checks: `uv run prek run -v --show-diff-on-failure --all-files`

# Child DOX Index

- `src/affidiff/AGENTS.md` - Core affine diffusion models, parameter classes, moments estimation, and simulation utilities
