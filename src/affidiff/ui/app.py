"""Main Streamlit application for affine diffusion simulation."""

from __future__ import annotations

from random import randint

import streamlit as st

from affidiff.ui.components import (
    render_chart,
    render_model_selector,
    render_parameter_form,
    render_simulation_controls,
)
from affidiff.ui.simulation import extract_simulation_data, run_simulation


def _init_session_state() -> None:
    """Initialize session state variables."""
    if "model" not in st.session_state:
        st.session_state.model = "GBM"
    if "last_simulation" not in st.session_state:
        st.session_state.last_simulation = None
    if "last_seed" not in st.session_state:
        st.session_state.last_seed = None
    if "last_params" not in st.session_state:
        st.session_state.last_params = None
    if "last_nobs" not in st.session_state:
        st.session_state.last_nobs = None


def _main() -> None:
    """Run the main Streamlit application."""
    st.set_page_config(page_title="Affine Diffusion Simulator", layout="wide", initial_sidebar_state="auto")

    _init_session_state()

    st.title("Affine Diffusion Model Simulator")

    col_controls, col_params = st.columns(2)

    with col_controls:
        selected_model = render_model_selector(default_model=st.session_state.model)

        if selected_model != st.session_state.model:
            st.session_state.model = selected_model
            st.session_state.last_simulation = None
            st.session_state.last_params = None

        nobs, seed = render_simulation_controls()

    with col_params:
        params = render_parameter_form(st.session_state.model)

    with col_controls:
        if st.button("Simulate", key="simulate_btn", width="stretch"):
            try:
                new_innov = st.session_state.last_simulation is None or seed is not None

                if seed is None:
                    seed = st.session_state.last_seed or randint(0, 2**31 - 1)

                simulated = run_simulation(
                    model_name=st.session_state.model, params=params, nobs=nobs, seed=seed, new_innov=new_innov
                )
                path = extract_simulation_data(simulated)
                st.session_state.last_simulation = path
                st.session_state.last_seed = seed
                st.session_state.last_params = params.copy()
                st.session_state.last_nobs = nobs
            except ValueError as e:
                st.error(f"Simulation failed: {e}")

        if st.button("🔄 Reset Seed", key="reset_btn", width="stretch"):
            st.session_state.last_seed = None
            st.session_state.last_simulation = None
            st.session_state.last_params = None
            st.session_state.last_nobs = None
            st.info("Seed reset. Click Simulate for a new series.")

    if st.session_state.last_simulation is not None:
        render_chart(simulated_path=st.session_state.last_simulation, seed=st.session_state.last_seed)


def main() -> None:
    """Entry point for CLI.

    Launches the Streamlit app, forwarding any CLI arguments like
    --logger.level=debug or --server.port=8502.
    """
    import subprocess
    import sys
    from pathlib import Path

    app_path = str(Path(__file__).resolve())
    result = subprocess.run(
        [sys.executable, "-m", "streamlit", "run", app_path, *sys.argv[1:]],
        check=False,
    )
    sys.exit(result.returncode)


if __name__ == "__main__":
    _main()
