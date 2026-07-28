"""CIR model class."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from affidiff.model_generic import SDE

if TYPE_CHECKING:
    import numpy as np


class CIR(SDE):
    """Cox-Ingersoll-Ross (CIR) model."""

    def __init__(self, param: Any = None) -> None:  # noqa: ANN401
        """Initialize the class.

        Parameters
        ----------
        param : parameter instance
            True parameters used for simulation of the data

        """
        super().__init__(param)

    @staticmethod
    def drift(*, state: np.ndarray | float, theta: Any) -> np.ndarray | float:  # noqa: ANN401
        """Drift function.

        Parameters
        ----------
        state : (nvars, nsim) array_like
            Current value of the process
        theta : parameter instance
            Model parameter

        Returns
        -------
        scalar
            Drift value

        """
        return theta.kappa * (theta.mean - state)

    @staticmethod
    def diff(*, state: np.ndarray | float, theta: Any) -> np.ndarray | float:  # noqa: ANN401
        """Diffusion (instantaneous volatility) function.

        Parameters
        ----------
        state : (nvars, nsim) array_like
            Current value of the process
        theta : parameter instance
            Model parameter

        Returns
        -------
        scalar
            Diffusion value

        """
        return theta.eta * state**0.5
