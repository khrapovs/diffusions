# Purpose

`tests/` contains the automated test suite for validating the correctness of `affidiff` models, parameters, moments estimation, simulation routines, and helper functions.

# Ownership

Owns all pytest test modules under `tests/`:
- Model tests mirroring `affidiff.models` (`models/test_ct.py`, `models/test_gbm.py`, `models/test_heston.py`)
- Parameter tests mirroring `affidiff.parameters` (`parameters/test_cir.py`, `parameters/test_ct.py`, `parameters/test_gbm.py`, `parameters/test_heston.py`, `parameters/test_vasicek.py`)
- Cross-model tests (`test_generic.py`, `test_ajd.py`)
- Moment estimation tests (`test_rmoms_ct.py`, `test_rmoms_gbm.py`, `test_rmoms_heston.py`)
- Helper and simulation tests (`test_helpers.py`, `test_simulations.py`, `test_simulations_centtend.py`, `test_simulations_heston.py`)

# Local Contracts

- Unit tests MUST NOT access private attributes or private methods (prefixed with `_`) of production classes.
- Test subpackages (`models/`, `parameters/`, `test_data/`) carry an `__init__.py` so modules with identical basenames stay collectible.
- Shared helpers `assertions.py` and `config.py` are importable from any test module via the `pythonpath = ["tests"]` pytest setting in `pyproject.toml`.
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
