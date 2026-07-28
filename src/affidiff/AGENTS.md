# Purpose

`src/affidiff/` is the core package implementing affine jump-diffusion stochastic differential equation (SDE) models, parameter classes, moment computations, characteristic functions, and Cython simulation routines.

# Ownership

Owns all core stochastic modeling logic, including:
- Generic affine SDE definition (`model_generic.py`, `param_generic.py`)
- Specific diffusion models: CIR (`model_cir.py`, `param_cir.py`), Vasicek (`model_vasicek.py`, `param_vasicek.py`), GBM (`model_gbm.py`, `param_gbm.py`), Heston (`model_heston.py`, `param_heston.py`), and Central Tendency (`model_ct.py`, `param_ct.py`)
- Mathematical helpers and moment functions (`helper_functions.py`)
- C/Cython fast simulation routines (`simulate.pyx`)
- Public exports (`__init__.py`)

# Local Contracts

- Every model class inherits from or complies with `SDE` and uses its corresponding parameter class (`*param.py`).
- Attributes and helper methods not accessed outside their owning class MUST be private (prefixed with `_`).
- Any new model or parameter class introduced must be exported in `__init__.py`.

# Work Guidance

- Ensure parameter bounds and matrix dimensions (drift, diffusion, jump intensities) are validated upon instantiation in parameter classes.
- Maintain mathematical accuracy in ODE solving, characteristic function evaluation, and numerical integration.
- All public methods and functions use `*` after `self`/`cls` to enforce keyword-only arguments (PLR0917 compliance). When calling internal methods that are keyword-only from contexts where positional calls are unavoidable (e.g., `numdifftools`), add a local positional wrapper function.

# Verification

- Run model test suite: `uv run pytest tests/`
- Run pre-commit hooks: `uv run prek run -v --show-diff-on-failure --all-files`

# Child DOX Index

None (leaf boundary).
