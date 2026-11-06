"""Simulation wrapper bridging Streamlit to affidiff models."""

from __future__ import annotations

import inspect
from typing import TYPE_CHECKING, Any

from affidiff.models.cir import CIR
from affidiff.models.ct import CentTend
from affidiff.models.gbm import GBM
from affidiff.models.heston import Heston
from affidiff.models.vasicek import Vasicek
from affidiff.parameters.cir import CIRparam
from affidiff.parameters.ct import CentTendParam
from affidiff.parameters.gbm import GBMparam
from affidiff.parameters.heston import HestonParam
from affidiff.parameters.vasicek import VasicekParam

if TYPE_CHECKING:
    from numpy.typing import NDArray

MODEL_REGISTRY: dict[str, tuple[type, type]] = {
    "GBM": (GBM, GBMparam),
    "Vasicek": (Vasicek, VasicekParam),
    "CIR": (CIR, CIRparam),
    "Heston": (Heston, HestonParam),
    "CentTend": (CentTend, CentTendParam),
}


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
