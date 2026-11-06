# Proposal: Add Streamlit UI for Affine Diffusion Model Simulations

## Why

Users currently need to write Python code to run simulations of affine diffusion models. A lightweight web-based UI will lower the barrier to entry and enable non-technical users to explore model behavior, adjust parameters, and visualize results interactively. This MVP targets the most common use case: single-path simulation with basic parameter tuning and visualization.

## What Changes

- Add a new `src/ui/` module with Streamlit-based interactive interface
- Implement model selector dropdown (GBM, Vasicek, CIR, Heston, CentTend)
- Auto-generate parameter input forms with sensible defaults per model type
- Add simulation controls: number of observations slider, random seed input, simulate button
- Render interactive line chart of simulated paths using Plotly
- Add CLI command `affidiff-ui` to `pyproject.toml` for launching the app
- Add Streamlit and Plotly as project dependencies

## Capabilities

### New Capabilities

- `ui/simulation-frontend`: Interactive web interface for running and visualizing affine diffusion model simulations. Allows single-model-at-a-time selection, parameter configuration with defaults, observation count control, and Plotly-based visualization of a single simulated path.

### Modified Capabilities

- None

## Impact

- **New Dependencies**: `streamlit`, `plotly` (dev/optional)
- **New Module**: `src/ui/` with app entry point
- **New CLI Entry**: `affidiff-ui` command
- **Affected Code**: `pyproject.toml` (script configuration), possibly `src/affidiff/__init__.py` if UI needs to be exported
- **No Breaking Changes**: Pure addition, does not modify existing APIs or behavior
