# Design: Streamlit UI for Affine Diffusion Simulations

## Architecture Overview

```
┌─────────────────────────────────────────────────┐
│            Streamlit Application                │
│              (src/ui/app.py)                    │
├─────────────────────────────────────────────────┤
│                                                 │
│  [Model Selector]        [Parameter Form]       │
│   Dropdown               (auto-generated)       │
│   (GBM/Vasicek/etc)      (numeric inputs)       │
│        │                       │                │
│        └───────────┬───────────┘                │
│                    ▼                            │
│        [Simulation Controls]                    │
│        nobs slider, seed input                  │
│        [SIMULATE] [RERUN]                       │
│                    │                            │
│                    ▼                            │
│        [Plotly Chart Display]                   │
│        Interactive line chart                   │
│                                                 │
├─────────────────────────────────────────────────┤
│          Supporting Modules                     │
│                                                 │
│  • components.py — Form builders, widgets      │
│  • simulation.py — Model/param wrappers        │
│  • __init__.py   — Package exports             │
│                                                 │
└─────────────────────────────────────────────────┘
         ↓
    affidiff package
    (models, parameters, simulate)
```

## Module Structure

### `src/ui/__init__.py`
Package initialization. May export main app entry point or version info.

### `src/ui/app.py`
Main Streamlit application entry point. Responsibilities:
- Initialize Streamlit session state for persistence across reruns
- Render model selector dropdown
- Dynamically render parameter form when model changes
- Handle "Simulate" button and rerun logic
- Display Plotly chart
- Manage simulation state (last result, current seed, etc.)

### `src/ui/components.py`
Reusable UI components:
- `render_model_selector()` — Dropdown for model selection
- `render_parameter_form(model_name)` — Auto-generate form fields from model parameter class
- `render_simulation_controls()` — nobs slider, seed input, buttons
- `render_chart(data)` — Plotly line chart from simulation output

### `src/ui/simulation.py`
Wrapper layer bridging Streamlit to affidiff models:
- `MODEL_REGISTRY` — Dict mapping model names to (model_class, param_class)
- `get_model_defaults(model_name)` — Extract default parameter values
- `run_simulation(model_name, params, nobs, seed)` — Instantiate model, validate params, call .simulate()
- `extract_simulation_data(simulated_array)` — Convert (nobs, 2*nsim, nvars) to plottable format for single path

## Key Design Decisions

### 1. Single Path Visualization
Per requirements, we simulate only one path (nsim=1). The output shape is (nobs, 2, nvars) due to antithetic sampling (default behavior). We extract one of the two paths for visualization.

### 2. Dynamic Parameter Forms
Instead of hardcoding parameter fields, we introspect each model's parameter class `__init__` signature to auto-generate form fields. This scales to new models without code changes.

**Approach:**
```python
import inspect
sig = inspect.signature(ModelParamClass.__init__)
for param_name, param in sig.parameters.items():
    if param_name == "self":
        continue
    default_value = param.default if param.default != inspect.Parameter.empty else None
    streamlit.number_input(param_name, value=default_value, ...)
```

### 3. Session State for Persistence
Streamlit reruns the entire script on every interaction. We use `st.session_state` to persist:
- Last simulated path (to avoid re-running on interaction)
- Current seed used (for reference)
- Current model selection

### 4. Error Handling
- Parameter validation happens before calling `.simulate()`
- User-friendly error messages in Streamlit alerts
- No exceptions bubble to CLI; all caught and displayed in UI

### 5. Random Seed Management
- If user provides a seed, use it (for reproducibility)
- If "Rerun" is clicked without a seed, generate a new random seed
- Display the seed used for each run so results are reproducible

## Implementation Steps

See `tasks.md` for the detailed implementation task breakdown.

## Data Flow Example

```
User selects "GBM" from dropdown
         ↓
app.py detects model_change in session_state
         ↓
components.render_parameter_form("GBM")
         ↓
simulation.get_model_defaults("GBM")
         ↓
Returns {"mean": 0.0, "sigma": 0.2}
         ↓
render_parameter_form creates number_input widgets
         ↓
User enters nobs=500, mean=0.05, sigma=0.15, seed=42
         ↓
User clicks "Simulate"
         ↓
simulation.run_simulation("GBM", {"mean": 0.05, "sigma": 0.15}, nobs=500, seed=42)
         ↓
Instantiates GBMparam, GBM model
         ↓
model.simulate(nobs=500, nsim=1, seed=42)
         ↓
Returns (500, 2, 1) array (one column is the path to plot)
         ↓
components.render_chart(path)
         ↓
Plotly line chart displays the path
```

## Dependencies

- **streamlit**: Web UI framework
- **plotly**: Interactive charting
- **affidiff**: Existing package (models, parameters, simulation)
- **inspect**: Standard library (introspection for auto-form generation)
- **numpy**: Already a dependency of affidiff

## Testing Strategy

Unit tests for:
- `get_model_defaults()` returns correct shape and types
- `run_simulation()` with valid/invalid parameters
- Chart rendering (mocked Streamlit, verify Plotly figure structure)

Integration tests via manual Streamlit interaction during development.

## Future Enhancements

1. Risk-neutral measure (Q) support
2. Multi-model batch comparison
3. Parameter preset saving/loading
4. Advanced statistics display (mean, volatility, quantiles)
5. Export to CSV/PNG
6. Custom time axis labels (start/end dates instead of indices)
