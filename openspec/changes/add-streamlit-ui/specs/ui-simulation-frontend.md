# Simulation Frontend UI — System Specification

## Purpose

Provide a lightweight, interactive web-based interface for running and visualizing single affine diffusion model simulations, enabling non-technical users to explore model behavior without writing code.

## Scope

This capability covers:
- Interactive model selection from available affine diffusion models (GBM, Vasicek, CIR, Heston, CentTend)
- Dynamic parameter input forms with model-specific fields and sensible default values
- Control over simulation parameters (number of observations, random seed for reproducibility)
- Real-time simulation execution with a button trigger
- Interactive visualization of simulated paths using line charts
- Session-based workflow: load parameters, run simulation, view result, optionally rerun with different seed

Out of scope (future enhancement):
- Risk-neutral measure (Q) support; physical measure (P) only
- Batch simulations or multi-model comparison
- Parameter persistence or saved presets
- Advanced statistics or metrics beyond the simulated path plot
- Export functionality (CSV, PNG)

## Requirements

### Model Selection
1. The UI SHALL present a dropdown menu listing all available simulation models
2. The dropdown options SHALL include: GBM, Vasicek, CIR, Heston, CentTend
3. Only ONE model SHALL be selectable at a time
4. Selection of a model SHALL immediately update the parameter form to show that model's parameters

### Parameter Configuration
5. The UI SHALL auto-generate a parameter input form based on the selected model's parameter class
6. Each numeric parameter field SHALL display its name, current value, and accept numeric input
7. The UI SHALL populate each parameter field with sensible default values from the model's parameter class
8. Default values SHALL be used when the user runs a simulation without modifying them
9. Parameter values SHALL be validated as numeric and within reasonable bounds before simulation

### Simulation Controls
10. The UI SHALL provide a slider or numeric input for number of observations (nobs), with a reasonable default (e.g., 500)
11. The UI SHALL provide an optional text input for random seed for reproducibility
12. The UI SHALL provide a "Simulate" button that triggers model instantiation and simulation execution
13. After simulation completes, the UI SHALL provide a "Rerun" option that generates a new random seed and re-runs the simulation
14. Between reruns, the user SHALL be able to modify parameters and run a fresh simulation

### Visualization
15. The UI SHALL display a line chart of the simulated path after each successful simulation
16. The chart SHALL show the single simulated path as a continuous line
17. The chart SHALL use Plotly library for interactivity (zoom, pan, hover tooltips)
18. The chart interface SHALL include standard Plotly controls (download, reset zoom, etc.)

### CLI Integration
19. The application SHALL be launchable via a CLI command `affidiff-ui`
20. The CLI command SHALL start the Streamlit development server with the UI app
21. The CLI command SHALL accept optional arguments for host, port (forwarded to Streamlit)

### Dependencies
22. The UI module SHALL depend on `streamlit` and `plotly` libraries
23. These dependencies SHALL be declared in `pyproject.toml` as optional or dev dependencies

## Non-Functional Requirements

- **Responsiveness**: Simulations shall complete within 2 seconds for typical parameter sets (nobs ≤ 1000)
- **Accessibility**: Form fields shall have clear labels and sensible defaults
- **Code Organization**: UI code SHALL reside in `src/ui/` and be modular (separate components for form generation, simulation wrapper, visualization)

## Interactions

```
User starts CLI (affidiff-ui)
         ↓
   [Streamlit loads]
         ↓
   Model selector displayed
         ↓
User selects model
         ↓
Parameter form auto-generated
         ↓
User enters/modifies parameters and nobs
         ↓
User clicks "Simulate"
         ↓
Simulation runs with provided seed (or random)
         ↓
Plotly chart displays result
         ↓
User can: modify parameters → new simulation
          OR rerun with new seed
```

## Acceptance Criteria

- [ ] CLI command `affidiff-ui` launches the Streamlit app without errors
- [ ] Model dropdown contains all 5 models and responds to selection
- [ ] Parameter form updates dynamically when model is changed
- [ ] Default parameter values are populated and match model class defaults
- [ ] Simulate button executes successfully and returns a simulated path
- [ ] Plotly chart displays the simulated path with interactivity intact
- [ ] Rerun button generates a new path with a different (random) seed
- [ ] Parameters can be modified between simulations
- [ ] Seeds can be entered manually for reproducibility
