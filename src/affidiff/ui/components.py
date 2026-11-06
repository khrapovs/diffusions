"""Reusable Streamlit UI components."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from affidiff.ui.simulation import get_model_defaults

if TYPE_CHECKING:
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
    models = ["GBM", "Vasicek", "CIR", "Heston", "CentTend"]
    selected = st.selectbox(
        "Select Model",
        options=models,
        index=models.index(default_model),
        help="Choose an affine diffusion model to simulate",
    )
    return selected


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
            value = st.number_input(
                param_name,
                value=default_value,
                format="%.6f",
                help=f"Parameter {param_name}",
            )
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


def render_chart(simulated_path: NDArray) -> None:
    """Render interactive Plotly line chart of simulated path.

    Parameters
    ----------
    simulated_path : NDArray
        1D array of simulated values
    """
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            y=simulated_path,
            x=np.arange(len(simulated_path)),
            mode="lines",
            name="Simulated Path",
            line=dict(color="rgb(31, 119, 180)", width=2),
            hovertemplate="<b>Time: %{x}</b><br>Value: %{y:.4f}<extra></extra>",
        )
    )

    fig.update_layout(
        title="Simulated Path",
        xaxis_title="Time Index",
        yaxis_title="Value",
        template="plotly_white",
        hovermode="x unified",
        height=500,
    )

    st.plotly_chart(fig, use_container_width=True)
