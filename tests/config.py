"""Test configuration and fixtures for diffusion model tests."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from affidiff.model_cir import CIR
from affidiff.model_ct import CentTend
from affidiff.model_gbm import GBM
from affidiff.model_heston import Heston
from affidiff.model_vasicek import Vasicek
from affidiff.param_cir import CIRparam
from affidiff.param_ct import CentTendParam
from affidiff.param_gbm import GBMparam
from affidiff.param_heston import HestonParam
from affidiff.param_vasicek import VasicekParam

if TYPE_CHECKING:
    from affidiff.model_generic import SDE
    from affidiff.param_generic import GenericParam


@dataclass(frozen=True)
class SimulationConfigForTests:
    """Configuration for simulation tests.

    Attributes
    ----------
    nobs : int
        Number of observations
    nsub : int
        Subsampling interval
    ndiscr : int
        Number of discretization steps
    nsim : int
        Number of simulations
    seed : int
        Random seed for reproducibility

    """

    nobs: int = 500
    nsub: int = 2
    ndiscr: int = 10
    nsim: int = 2
    seed: int = 42


def get_model_fixtures() -> list[tuple[type[SDE], GenericParam]]:
    """Return list of (ModelClass, param_instance) tuples.

    Returns
    -------
    list[tuple[type[SDE], GenericParam]]
        List of model fixtures where each tuple contains:
        - Model class (e.g., GBM, Vasicek)
        - Parameter instance (initialized with test values)

    """
    return [
        (GBM, GBMparam(mean=0.05, sigma=0.2)),
        (Vasicek, VasicekParam(mean=0.5, kappa=0.1, eta=0.2)),
        (Heston, HestonParam(riskfree=0.0, lmbd=0.0, mean_v=0.5, kappa=0.1, eta=0.02**0.5, rho=-0.9)),
        (CIR, CIRparam(mean=0.5, kappa=0.1, eta=0.2)),
        (
            CentTend,
            CentTendParam(
                riskfree=0.01,
                lmbd=0.01,
                mean_v=0.5,
                kappa_s=1.5,
                kappa_y=0.05,
                eta_s=0.02**0.5,
                eta_y=0.001**0.5,
                rho=-0.9,
            ),
        ),
    ]
