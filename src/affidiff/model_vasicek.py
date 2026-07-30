"""Vasicek model class."""

from __future__ import annotations

from typing import TYPE_CHECKING

from affidiff.model_generic import SDE

if TYPE_CHECKING:
    import numpy as np

    from affidiff.param_vasicek import VasicekParam


class Vasicek(SDE):
    """Vasicek model."""

    param: VasicekParam | None

    def __init__(self, param: VasicekParam | None = None) -> None:
        """Initialize the class.

        Parameters
        ----------
        param : parameter instance
            True parameters used for simulation of the data

        """
        super().__init__(param)

    def get_start(self) -> list[float]:
        """Get starting values for simulation.

        Returns
        -------
        list[float]
            Starting value at the long-run mean

        """
        assert self.param is not None
        return [float(self.param.mean)]

    @staticmethod
    def drift(*, state: np.ndarray | float, theta: VasicekParam) -> np.ndarray | float:
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
    def diff(*, state: np.ndarray | float, theta: VasicekParam) -> float:
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
        _ = state
        return theta.eta


if __name__ == "__main__":
    pass
