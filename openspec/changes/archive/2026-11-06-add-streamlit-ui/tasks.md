# Tasks: Implement Streamlit UI for Affine Diffusion Simulations

## Setup & Dependencies

### T1. Add Streamlit and Plotly to Dependencies
- [x] Open `pyproject.toml`
- [x] Add `streamlit` and `plotly` to optional dependencies or dev dependencies
- [x] Add CLI script entry: `affidiff-ui = "affidiff.ui.app:main"`
- [x] Run `uv sync` to install new dependencies
- [x] Verify dependencies resolve without conflicts

### T2. Create UI Package Structure
- [x] Create `src/ui/` directory
- [x] Create `src/ui/__init__.py` (empty or with version export)
- [x] Verify import works: `from affidiff.ui import ...`

## Core Implementation

### T3. Implement `src/ui/simulation.py` — Model Registry & Wrapper
- [x] Import all model classes: `GBM, Vasicek, CIR, Heston, CentTend`
- [x] Import all parameter classes
- [x] Create `MODEL_REGISTRY = { "GBM": (GBM, GBMparam), "Vasicek": (Vasicek, Vasicekparam), ... }`
- [x] Implement `get_model_defaults(model_name: str) -> dict`:
  - Use `inspect.signature()` to read parameter class `__init__`
  - Extract default values
  - Return dict mapping param names to defaults
  - Test with all 5 models
- [x] Implement `run_simulation(model_name: str, params: dict, nobs: int, seed: int | None) -> np.ndarray`:
  - Validate model_name exists in registry
  - Instantiate parameter class with provided params
  - Instantiate model class
  - Call `.simulate(nobs=nobs, nsim=1, seed=seed)`
  - Return result array
  - Handle exceptions and return user-friendly error messages
- [x] Implement `extract_simulation_data(simulated: np.ndarray) -> np.ndarray`:
  - Input shape: (nobs, 2, nvars) from nsim=1 with antithetic sampling
  - Extract one path: `simulated[:, 0, :]` (or handle multi-variate)
  - Return 1D array suitable for plotting

**Tests for T3:**
- [x] `test_get_model_defaults()` for each model
- [x] `test_run_simulation()` with valid parameters
- [x] `test_run_simulation()` with invalid parameters (raises appropriately)
- [x] `test_extract_simulation_data()` with mock array

### T4. Implement `src/ui/components.py` — UI Widgets
- [x] Implement `render_model_selector(default_model: str = "GBM") -> str`:
  - Use `streamlit.selectbox()` with model list
  - Return selected model name
- [x] Implement `render_parameter_form(model_name: str) -> dict`:
  - Get defaults via `get_model_defaults(model_name)`
  - For each parameter, render `streamlit.number_input()` with default and label
  - Collect and return form values as dict
- [x] Implement `render_simulation_controls(default_nobs: int = 500) -> tuple`:
  - nobs: Use `streamlit.slider()` (range 10-5000, default 500)
  - seed: Use `streamlit.number_input()` with `value=None` (optional)
  - Return `(nobs, seed)`
- [x] Implement `render_chart(simulated_path: np.ndarray) -> None`:
  - Create Plotly figure: line chart of simulation path
  - X-axis: index (0 to nobs-1)
  - Y-axis: simulated values
  - Use `st.plotly_chart()` to render
  - Add hover tooltip showing (index, value)

**Tests for T4:**
- [x] Mock Streamlit components and verify correct widgets are called
- [x] Test parameter form generation for each model

### T5. Implement `src/ui/app.py` — Main Streamlit App
- [x] Initialize Streamlit page config: title, layout, etc.
- [x] Initialize session state keys: `model`, `last_simulation`, `last_seed`
- [x] Layout structure:
  - **Header**: Title and brief description
  - **Column 1**: Model selector + Parameter form
  - **Column 2 or Below**: Simulation controls
  - **Main area**: Chart display
- [x] Model selection logic:
  - Render `render_model_selector()`
  - If model changes, reset parameter form and clear last_simulation
  - Store in session_state["model"]
- [x] Parameter form:
  - Render `render_parameter_form(session_state["model"])`
  - Store form values in session state
- [x] Simulation controls:
  - Render `render_simulation_controls()`
  - "Simulate" button:
    - Call `run_simulation()` with current params, nobs, seed
    - Store result in session_state["last_simulation"]
    - Store seed in session_state["last_seed"]
    - Show success/error message
  - "Rerun" button (only visible if simulation exists):
    - Generate new random seed
    - Call `run_simulation()` with same params but new seed
    - Update session state
- [x] Chart rendering:
  - If session_state["last_simulation"] exists, render via `render_chart()`
  - Include seed information in chart caption
- [x] Error handling:
  - Wrap simulation calls in try/except
  - Display errors via `st.error()`
  - Allow user to retry

**Tests for T5:**
- [x] Mock Streamlit and verify page config is set
- [x] Mock model selection and verify parameter form updates
- [x] Mock button clicks and verify simulation is called

### T6. Create CLI Entry Point
- [x] Define `main()` function in `app.py`:
  - Entry point for `affidiff-ui` command
  - Calls Streamlit server (or delegates to streamlit CLI)
- [x] Verify `pyproject.toml` script entry is set correctly
- [x] Test: Run `affidiff-ui` from command line, verify Streamlit app starts

## Integration & Validation

### T7. Manual Integration Testing
- [x] Run `affidiff-ui` command
- [x] Verify Streamlit app loads in browser
- [x] Test model selector: Select each model, verify dropdown updates
- [x] Test parameter form: Verify correct parameters appear for each model
- [x] Test simulation:
  - Set nobs=100, leave seed empty, click Simulate
  - Verify chart renders
  - Click Rerun, verify new chart (different seed)
  - Modify a parameter and click Simulate again
  - Verify chart updates with new parameters
- [x] Test seed reproducibility:
  - Set nobs=100, seed=12345, click Simulate, save the chart
  - Rerun app, set same nobs and seed, verify chart is identical
- [x] Test all 5 models (GBM, Vasicek, CIR, Heston, CentTend)

### T8. Code Quality & Documentation
- [x] Run `uv run prek run -v --all-files` (pre-commit, ruff, type checking)
- [x] Fix any linting/formatting issues
- [x] Add docstrings to all functions
- [x] Add type hints to all function signatures
- [x] Check that UI code imports are clean (no unused imports)

### T9. Update AGENTS.md
- [x] Read root `AGENTS.md` and `src/AGENTS.md` to understand DOX hierarchy
- [x] Create `src/ui/AGENTS.md`:
  - Purpose: UI module for interactive simulation
  - Ownership: (your name/team)
  - Work Guidance: Follow project coding standards, use Streamlit patterns
  - Verification: Manual testing via `affidiff-ui`
- [x] Update `src/AGENTS.md` to add `src/ui/AGENTS.md` to Child DOX Index
- [x] Update root `AGENTS.md` Child DOX Index if needed

## Completion & Verification

### T10. Final Verification
- [x] All tests pass: `uv run pytest tests/ -xvs` (including any new UI tests)
- [x] App runs without errors: `affidiff-ui` launches successfully
- [x] All 5 models work end-to-end
- [x] Parameter defaults match model class defaults
- [x] Charts render correctly with Plotly
- [x] Seed reproducibility verified
- [x] Pre-commit checks pass
- [x] DOX hierarchy updated

**Definition of Done:**
- [x] All tasks completed and checked
- [x] Code passes linting and type checking
- [x] Manual testing confirms all features work as specified
- [x] Documentation updated
- [x] Ready for user acceptance testing
