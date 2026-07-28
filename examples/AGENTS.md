# Purpose

`examples/` contains runnable Python scripts illustrating usage of `affidiff` models, trajectory simulation, moment calculation, and real financial dataset loading.

# Ownership

Owns executable demonstration scripts:
- Model trial scripts (`try_cir.py`, `try_gbm.py`, `try_heston.py`, `try_vasicek.py`, `try_centtend.py`)
- Real data loading utility example (`load_real_data.py`)

# Local Contracts

- Example scripts must import solely from the public `affidiff` package interface.
- Scripts must run executable without unhandled runtime exceptions using `uv run python examples/<script_name>.py`.

# Work Guidance

- Keep example scripts clean, well-commented, and representative of real-world use cases (e.g. calibration, simulation, plotting).

# Verification

- Execute scripts: `uv run python examples/try_gbm.py` (or other example scripts)
- Ensure clean linting: `uv run prek run -v --show-diff-on-failure --all-files`

# Child DOX Index

None (leaf boundary).
