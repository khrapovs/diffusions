# Purpose

`src/affidiff/models/` implements the affine jump-diffusion SDE model classes: drift/diffusion specification,
characteristic functions, moment conditions, GMM estimation, and simulation entry points.

# Ownership

- `generic.py` — `SDE` base class defining the generic affine model API (simulation, realized measures, GMM
  estimation, ODE solving).
- `cir.py` — `CIR`.
- `vasicek.py` — `Vasicek`.
- `gbm.py` — `GBM`.
- `heston.py` — `Heston`.
- `ct.py` — `CentTend` (central tendency).

# Local Contracts

- Every model class inherits from `SDE` and pairs with its same-named module in `affidiff.parameters`
  (`models/heston.py` ↔ `parameters/heston.py`).
- Module names carry no `model_` prefix; the package name supplies the context.
- Attributes and helper methods not accessed outside their owning class MUST be private (prefixed with `_`).
- Shared mathematical helpers live in `affidiff.helper_functions`, not in individual model modules.

# Work Guidance

- Maintain mathematical accuracy in ODE solving, characteristic function evaluation, and numerical integration.
- All public methods and functions use `*` after `self`/`cls` to enforce keyword-only arguments (PLR0917 compliance).
  When calling internal keyword-only methods from contexts where positional calls are unavoidable (e.g.
  `numdifftools`), add a local positional wrapper function.
- Mirror any new model with a test module under `tests/test_models/`.

# Verification

- Run model tests: `uv run pytest tests/test_models`
- Run pre-commit hooks: `uv run prek run -v --show-diff-on-failure --all-files`

# Child DOX Index

None (leaf boundary).
