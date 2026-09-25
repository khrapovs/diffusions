# Purpose

`` contains Sphinx configuration, build instructions, and source files for generating the documentation website and API
reference for `affidiff`.

# Ownership

Owns Sphinx documentation source files and build configurations:

- Build automation Makefile (``)
- Documentation sources (``)

# Local Contracts

- Docstrings across production modules must adhere to standard NumPy format (as configured in `pyproject.toml` under
  `tool.ruff.lint.pydocstyle`).
- Sphinx build files must compile without errors or missing module references.

# Work Guidance

- Ensure new public functions, classes, and parameter types added to `affidiff` are documented and visible in the Sphinx
  source index.

# Verification

- Build documentation via `Makefile` inside `` directory when Sphinx is installed.

# Child DOX Index

None (leaf boundary).
