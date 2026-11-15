"""Reusable Streamlit UI components."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from affidiff.models.registry import MODEL_REGISTRY
from affidiff.ui.simulation import get_model_defaults

if TYPE_CHECKING:
    from collections.abc import Sequence

    from numpy.typing import NDArray


def render_model_selector(default_model: str = "GBM") -> str:
    """Render dropdown for model selection.

    Parameters
    ----------
    default_model : str
        Default model to select

    Returns
    -------
    str
        Selected model name
    """
    models = list(MODEL_REGISTRY.keys())
    return st.selectbox(
        "Select Model",
        options=models,
        index=models.index(default_model),
        help="Choose an affine diffusion model to simulate",
    )


def render_parameter_form(model_name: str) -> dict[str, Any]:
    """Auto-generate parameter input form for selected model.

    Parameters
    ----------
    model_name : str
        Name of the model

    Returns
    -------
    dict[str, Any]
        Dictionary of parameter values from form inputs
    """
    defaults = get_model_defaults(model_name)
    params = {}

    with st.expander(f"{model_name} Parameters", expanded=True):
        for param_name, default_value in defaults.items():
            if param_name == "measure":
                continue
            value = st.number_input(param_name, value=default_value, format="%.6f", help=f"Parameter {param_name}")
            params[param_name] = value

    return params


def render_simulation_controls(default_nobs: int = 500) -> tuple[int, int | None]:
    """Render simulation control widgets.

    Parameters
    ----------
    default_nobs : int
        Default number of observations

    Returns
    -------
    tuple[int, int | None]
        (nobs, seed) where seed is None if not provided
    """
    col1, col2 = st.columns(2)

    with col1:
        nobs = st.slider(
            "Number of Observations",
            min_value=10,
            max_value=5000,
            value=default_nobs,
            step=10,
            help="Number of time points to simulate",
        )

    with col2:
        seed_input = st.number_input(
            "Random Seed (optional)",
            value=None,
            min_value=0,
            help="Leave blank for random seed, or enter an integer for reproducibility",
        )
        seed = int(seed_input) if seed_input is not None else None

    return nobs, seed


def render_chart(simulated_path: NDArray, *, series_names: Sequence[str], seed: int | None) -> None:
    """Render interactive Plotly charts of simulated series, one subplot per series.

    Parameters
    ----------
    simulated_path : NDArray
        Simulated values of shape (nobs,) for 1-variable models, or (nobs, nvars)
    series_names : Sequence[str]
        Name of each simulated series, one per column of `simulated_path`
    seed : int | None
        Random seed used for simulation
    """
    paths = simulated_path.reshape(len(simulated_path), -1)
    nvars = paths.shape[1]
    time_index = np.arange(len(paths))

    fig = make_subplots(rows=nvars, cols=1, shared_xaxes=True, vertical_spacing=0.08, subplot_titles=series_names)

    for row, (name, path) in enumerate(zip(series_names, paths.T, strict=True), start=1):
        fig.add_trace(
            go.Scatter(
                y=path,
                x=time_index,
                mode="lines",
                name=name,
                line=dict(width=2),
                hovertemplate="<b>Time: %{x}</b><br>Value: %{y:.4f}<extra></extra>",
            ),
            row=row,
            col=1,
        )

    fig.update_xaxes(title_text="Time Index", row=nvars, col=1)
    fig.update_layout(
        title=f"Simulated Paths (seed = {seed})",
        template="plotly_white",
        hovermode="x unified",
        showlegend=False,
        height=150 + 300 * nvars,
    )

    st.plotly_chart(fig, width="stretch")
