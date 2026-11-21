"""Simulation wrapper bridging Streamlit to affidiff models."""

from __future__ import annotations

import inspect
from typing import TYPE_CHECKING, Any

import numpy as np

from affidiff.models.registry import MODEL_REGISTRY

if TYPE_CHECKING:
    from numpy.typing import NDArray


def get_series_names(model_name: str) -> tuple[str, ...]:
    """Get names of the series simulated by a model.

    Parameters
    ----------
    model_name : str
        Name of the model (key in MODEL_REGISTRY)

    Returns
    -------
    tuple[str, ...]
        Names of the simulated series in simulation column order

    Raises
    ------
    ValueError
        If model_name is not in MODEL_REGISTRY
    """
    if model_name not in MODEL_REGISTRY:
        msg = f"Unknown model: {model_name}. Available: {list(MODEL_REGISTRY.keys())}"
        raise ValueError(msg)

    return MODEL_REGISTRY[model_name][0]().series_names


def get_model_defaults(model_name: str) -> dict[str, Any]:
    """Extract default parameter values from a model's parameter class.

    Parameters
    ----------
    model_name : str
        Name of the model (key in MODEL_REGISTRY)

    Returns
    -------
    dict[str, Any]
        Mapping of parameter names to their default values

    Raises
    ------
    ValueError
        If model_name is not in MODEL_REGISTRY
    """
    if model_name not in MODEL_REGISTRY:
        msg = f"Unknown model: {model_name}. Available: {list(MODEL_REGISTRY.keys())}"
        raise ValueError(msg)

    param_class = MODEL_REGISTRY[model_name][1]
    sig = inspect.signature(param_class.__init__)
    defaults = {}

    for param_name, param in sig.parameters.items():
        if param_name == "self":
            continue
        if param.default != inspect.Parameter.empty:
            defaults[param_name] = param.default

    return defaults


def run_simulation(
    *, model_name: str, params: dict[str, Any], nobs: int, seed: int | None = None, new_innov: bool = True
) -> NDArray:
    """Run a single simulation with given parameters.

    Parameters
    ----------
    model_name : str
        Name of the model (key in MODEL_REGISTRY)
    params : dict[str, Any]
        Parameter values to pass to the model's parameter class
    nobs : int
        Number of observations to simulate
    seed : int | None
        Random seed for reproducibility. If None, random seed is used.
    new_innov : bool
        Whether to generate new innovations (True), or use already stored (False)

    Returns
    -------
    NDArray
        Simulated data array of shape (nobs, 2*nsim, nvars)

    Raises
    ------
    ValueError
        If model_name is unknown or parameters are invalid
    """
    if model_name not in MODEL_REGISTRY:
        msg = f"Unknown model: {model_name}. Available: {list(MODEL_REGISTRY.keys())}"
        raise ValueError(msg)

    model_class, param_class = MODEL_REGISTRY[model_name]

    try:
        param_obj = param_class(**params)
    except TypeError as e:
        msg = f"Invalid parameters for {model_name}: {e}"
        raise ValueError(msg) from e

    model = model_class(param_obj)
    simulated = model.simulate(nobs=nobs, nsim=1, seed=seed, new_innov=new_innov, cython=False)

    return simulated


def extract_simulation_data(simulated: NDArray) -> NDArray:
    """Extract a single path from simulation output for plotting.

    Parameters
    ----------
    simulated : NDArray
        Simulation output of shape (nobs, 2*nsim, nvars)

    Returns
    -------
    NDArray
        Single path of shape (nobs,) for 1-variable models, or (nobs, nvars)
    """
    path = simulated[:, 0, :]
    if path.shape[1] == 1:
        return path.squeeze(axis=1)
    return path


def apply_display_transforms(*, path: NDArray, raw_series_names: tuple[str, ...]) -> tuple[NDArray, tuple[str, ...]]:
    """Transform raw simulated path into display-ready series for the chart.

    For every series named ``"log(Price)"``:
    - Replace it with ``exp(log(Price))`` = Price.
    - Append a new ``"Log Return"`` series = ``diff(log(Price))``, with 0
      prepended so the length matches the other series.

    All other series are passed through unchanged.

    Parameters
    ----------
    path : NDArray
        Raw simulated path of shape ``(nobs,)`` (univariate) or ``(nobs, nvars)``.
    raw_series_names : tuple[str, ...]
        Series names from the model, one per column of ``path``.

    Returns
    -------
    tuple[NDArray, tuple[str, ...]]
        ``(display_path, display_names)`` where ``display_path`` is 2-D of
        shape ``(nobs, n_display_series)`` and ``display_names`` lists one
        label per column.
    """
    cols = path.reshape(len(path), -1).T.tolist()
    names: list[str] = []
    display_cols: list[NDArray] = []

    for name, col in zip(raw_series_names, cols, strict=True):
        col_arr = np.asarray(col)
        if name == "log(Price)":
            display_cols.append(np.exp(col_arr))
            names.append("Price")
            log_return = np.concatenate(([0.0], np.diff(col_arr)))
            display_cols.append(log_return)
            names.append("Log Return")
        else:
            display_cols.append(col_arr)
            names.append(name)

    return np.column_stack(display_cols), tuple(names)
