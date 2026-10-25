# Purpose

`src/affidiff/parameters/` holds the parameter classes for each diffusion model: storage, validation, bounds,
constraints, and conversion between parameter objects and flat `theta` arrays used by estimation routines.

# Ownership

- `generic.py` — `GenericParam` base class defining the shared parameter API (`get_model_name`, `theta`
  conversion, bounds, constraints, measure handling).
- `cir.py` — `CIRparam`.
- `vasicek.py` — `VasicekParam`.
- `gbm.py` — `GBMparam`.
- `heston.py` — `HestonParam`.
- `ct.py` — `CentTendParam`.

# Local Contracts

- Every parameter class inherits from `GenericParam` and pairs with its same-named module in `affidiff.models`
  (`parameters/heston.py` ↔ `models/heston.py`).
- Module names carry no `param_` prefix; the package name supplies the context.
- Parameter bounds and matrix dimensions (drift, diffusion, jump intensities) MUST be validated upon instantiation.
- Attributes and helper methods not accessed outside their owning class MUST be private (prefixed with `_`).

# Work Guidance

- Keep `get_model_name` in sync with the paired model class name used in documentation and output.
- Mirror any new parameter class with a test module under `tests/test_parameters/`.

# Verification

- Run parameter tests: `uv run pytest tests/test_parameters`
- Run pre-commit hooks: `uv run prek run -v --show-diff-on-failure --all-files`

# Child DOX Index

None (leaf boundary).
