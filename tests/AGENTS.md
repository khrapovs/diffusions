# Purpose

`tests/` contains the automated test suite for validating the correctness of `affidiff` models, parameters, moments
estimation, simulation routines, and helper functions.

# Ownership

Owns all pytest test modules under `tests/`:

- Model tests mirroring `affidiff.models` (`test_models/test_ct.py`, `test_models/test_gbm.py`,
  `test_models/test_heston.py`)
- Parameter tests mirroring `affidiff.parameters` (`test_parameters/test_cir.py`, `test_parameters/test_ct.py`,
  `test_parameters/test_gbm.py`, `test_parameters/test_heston.py`, `test_parameters/test_vasicek.py`)
- Cross-model tests (`test_generic.py`, `test_ajd.py`)
- Moment estimation tests (`test_rmoms_ct.py`, `test_rmoms_gbm.py`, `test_rmoms_heston.py`)
- Helper and simulation tests (`test_helpers.py`, `test_simulations.py`, `test_simulations_centtend.py`,
  `test_simulations_heston.py`)

# Local Contracts

- Unit tests MUST NOT access private attributes or private methods (prefixed with `_`) of production classes.
- Test subpackages (`test_models/`, `test_parameters/`, `test_data/`) carry an `__init__.py` so modules with identical
  basenames stay collectible; this also keeps `tests/` on `sys.path` so shared helpers `assertions.py` and `config.py`
  remain importable from nested test modules.
- Tests must pass cleanly when executed via `uv run pytest`.
- `uv run pytest` also collects docstring `Examples` from `src/` via `--doctest-modules`; examples that cannot run
  without side effects (network, plotting) must not be written as `>>>` blocks.
- Coverage metrics are configured via `.coveragerc` and enforced during testing.

# Work Guidance

- Group tests logically by component (parameter tests, model trajectory tests, moment condition tests).
- Prefer deterministic test cases with set random seeds where stochastic simulations are involved.

# Verification

- Run full test suite: `uv run pytest`
- Run test suite with coverage report: `uv run pytest --cov=src/affidiff`

# Child DOX Index

None (leaf boundary).
