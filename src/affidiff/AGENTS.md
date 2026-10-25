# Purpose

`src/affidiff/` is the core package implementing affine jump-diffusion stochastic differential equation (SDE) models,
parameter classes, moment computations, characteristic functions, market data loaders, and Cython simulation routines.

# Ownership

Owns the package-level glue and shared utilities; model, parameter, and data specifics belong to the child packages
listed in the Child DOX Index.

Root-owned modules:

- `helper_functions.py` — shared mathematical helpers and moment functions (`ajd_drift`, `ajd_diff`, `poly_coef`,
  `rolling_window`, `columnwise_prod`, `instruments`, `nice_errors`).
- `simulate.pyx` / `simulate.c` — Cython fast simulation routines (see root `AGENTS.md` for the build workflow).
- `_cython.py` — optional loading of the compiled `simulate` extension, falling back to the pure-Python path.
- `random.py` — `get_random_generator`, the single entry point for seeded `numpy` generators.
- `types.py` — shared enums (`Measure`, `Subset`).
- `__init__.py` — package marker; it is intentionally empty, so consumers import from fully qualified submodules
  (e.g. `from affidiff.models.heston import Heston`).

# Local Contracts

- Model modules and their parameter modules pair by matching file name across `models/` and `parameters/`.
- Attributes and helper methods not accessed outside their owning class MUST be private (prefixed with `_`).
- Cross-cutting mathematics belongs in `helper_functions.py`; do not duplicate it inside model modules.
- Stochastic code draws randomness through `affidiff.random.get_random_generator`, never through global `numpy` state.

# Work Guidance

- All public methods and functions use `*` after `self`/`cls` to enforce keyword-only arguments (PLR0917 compliance).
  When calling internal keyword-only methods from contexts where positional calls are unavoidable (e.g.
  `numdifftools`), add a local positional wrapper function.
- Follow strict typing and NumPy docstring conventions configured in `pyproject.toml`.
- Docstring `Examples` sections are executed as tests (`--doctest-modules`), so every `>>>` line must run and match
  its expected output exactly. Write plain numpydoc examples; do not wrap them in Sphinx `.. doctest::` directives.

# Verification

- Run test suite: `uv run pytest`
- Run pre-commit hooks: `uv run prek run -v --show-diff-on-failure --all-files`

# Child DOX Index

- `src/affidiff/models/AGENTS.md` - SDE model classes (`SDE`, `CIR`, `Vasicek`, `GBM`, `Heston`, `CentTend`)
- `src/affidiff/parameters/AGENTS.md` - Parameter classes, validation, bounds, and `theta` conversion
- `src/affidiff/data/AGENTS.md` - Market data loaders returning `polars` lazy frames
