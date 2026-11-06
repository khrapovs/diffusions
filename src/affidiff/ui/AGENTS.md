# UI Module (src/affidiff/ui)

## Purpose

Provide a lightweight, interactive web-based interface for running and visualizing single affine diffusion model simulations using Streamlit and Plotly.

## Ownership

GitHub Copilot (AI agent)

## Local Contracts

### Scope

This module owns:
- Interactive web UI components built with Streamlit
- Simulation wrapper layer bridging Streamlit to core affidiff models
- CLI entry point (`affidiff-ui` command)
- All dependencies related to UI rendering (streamlit, plotly)

This module does NOT own:
- Core model simulation logic (owned by `src/affidiff/models`)
- Parameter classes (owned by `src/affidiff/parameters`)

### Exports

- `affidiff.ui.app:main()` — CLI entry point, launches Streamlit application

### External Dependencies

- `streamlit>=1.50.0` — Web UI framework
- `plotly>=5.20.0` — Interactive charting library

## Work Guidance

### Coding Standards

Follow all project standards from root `AGENTS.md`:
- Format with `ruff`; type-check with `ty`
- No assertions in production code
- Action-verb function names (`render_*`, `extract_*`, `run_*`)
- Private attributes/methods unless used externally
- Docstrings for all functions (numpy style)
- Type hints required on all signatures

### Streamlit Patterns

- Use `st.session_state` for persistence across reruns
- Mock streamlit components in tests; integration testing via manual UI interaction
- Suppress exceptions and display errors via `st.error()`

### Simulate Button Behavior

The Simulate button implements smart innovation (random error) reuse:

1. **First simulation or new seed provided**: Generate new innovations (`new_innov=True`)
2. **Change parameters only**: Reuse stored innovations, recalculate paths with new params (`new_innov=False`)
   - Allows direct visual comparison of parameter effects without noise interference
3. **Reset Seed button**: Clears stored innovations and seed, next Simulate generates entirely new series

This workflow enables:
- **Same parameters → same series** (perfect reproducibility)
- **New parameters → old errors, new responses** (see param effect in isolation)
- **Total reset** via dedicated button when needed

### Adding New Models

To support a new affine diffusion model:
1. Update `MODEL_REGISTRY` in `simulation.py`
2. Test with existing tests (parametrized for all models)
3. Test UI form generation manually

## Verification

### Testing

- Unit tests in `tests/test_ui_simulation.py` cover:
  - Model registry and parameter extraction
  - Simulation wrapper with various parameters and seeds
  - Data extraction for visualization
- Run: `uv run pytest tests/test_ui_simulation.py -xvs`

### Manual Integration Testing

1. Launch app: `affidiff-ui` (opens Streamlit in browser on http://localhost:8501)
2. Test all 5 models: GBM, Vasicek, CIR, Heston, CentTend
3. Verify parameter defaults match model definitions
4. Test simulation with various nobs values (10–5000)
5. Test seed reproducibility: same seed produces identical paths
6. Verify Plotly chart renders interactively
7. Test Rerun button generates new random paths

### Code Quality

- Pre-commit checks: `uv run prek run --all-files`
  - Ruff format/check
  - Type checking (ty)
  - Trailing whitespace, YAML, etc.
- All checks must pass before merge

## Child DOX Index

None (leaf module)
