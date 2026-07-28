# Purpose

`tests/` contains the automated test suite for validating the correctness of `affidiff` models, parameters, moments estimation, simulation routines, and helper functions.

# Ownership

Owns all pytest test modules under `tests/`:
- Model tests (`test_generic.py`, `test_ajd.py`, `test_param_*.py`)
- Moment estimation tests (`test_rmoms_ct.py`, `test_rmoms_gbm.py`, `test_rmoms_heston.py`)
- Helper and simulation tests (`test_helpers.py`, `test_simulations.py`)

# Local Contracts

- Unit tests MUST NOT access private attributes or private methods (prefixed with `_`) of production classes.
- Tests must pass cleanly when executed via `uv run pytest`.
- Coverage metrics are configured via `.coveragerc` and enforced during testing.

# Work Guidance

- Group tests logically by component (parameter tests, model trajectory tests, moment condition tests).
- Prefer deterministic test cases with set random seeds where stochastic simulations are involved.

# Verification

- Run full test suite: `uv run pytest`
- Run test suite with coverage report: `uv run pytest --cov=src/affidiff`

# Child DOX Index

None (leaf boundary).
