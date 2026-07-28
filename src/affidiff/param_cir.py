# !/usr/bin/env python
# -*- coding: utf-8 -*-
"""CIR parameter class."""

from __future__ import annotations

from typing import TYPE_CHECKING, Sequence

import numpy as np

from .param_generic import GenericParam

if TYPE_CHECKING:
    from typing_extensions import Self


class CIRparam(GenericParam):
    """Parameter storage for CIR model.

    Attributes
    ----------
    mean : float
        Mean of the process
    kappa : float
        Mean reversion speed
    eta : float
        Instantaneous standard deviation
    measure : str
        Under which measure (P or Q)

    """

    def __init__(self, mean: float = 0.5, kappa: float = 1.5, eta: float = 0.1, measure: str = "P") -> None:  # noqa: PLR0917, ARG002
        """Initialize class.

        Parameters
        ----------
        mean : float
            Mean of the process
        kappa : float
            Mean reversion speed
        eta : float
            Instantaneous standard deviation
        measure : str

            Under which measure:
                - 'P' : physical measure
                - 'Q' : risk-neutral

        """
        super().__init__()
        self.mean = mean
        self.kappa = kappa
        self.eta = eta
        self.measure = "P"
        self.update_ajd()

    def is_valid(self) -> bool:
        """Check validity of parameters.

        Returns
        -------
        bool
            True for valid parameters, False for invalid

        """
        posit = (self.kappa > 0) & (self.eta > 0)
        feller = 2 * self.kappa * self.mean - self.eta**2 > 0
        return bool(posit & feller)

    def update_ajd(self) -> None:
        """Update AJD representation."""
        # AJD parameters
        self.mat_k0 = self.kappa * self.mean
        self.mat_k1 = -self.kappa
        self.mat_h0 = 0.0
        self.mat_h1 = self.eta**2

    @classmethod
    def from_theta(cls, theta: np.ndarray | Sequence[float]) -> Self:
        """Initialize parameters from parameter vector.

        Parameters
        ----------
        theta : (nparams, ) array
            Parameter vector

        """
        param = cls(mean=float(theta[0]), kappa=float(theta[1]), eta=float(theta[2]))
        param.update_ajd()
        return param

    def update(self, theta: np.ndarray | Sequence[float], subset: str = "all", measure: str = "P") -> None:  # noqa: PLR0917, ARG002
        """Update attributes from parameter vector.

        Parameters
        ----------
        theta : (nparams, ) array
            Parameter vector
        subset : str
            Which parameters to update
        measure : str
            Probability measure

        """
        self.mean, self.kappa, self.eta = float(theta[0]), float(theta[1]), float(theta[2])
        self.update_ajd()

    @staticmethod
    def get_model_name() -> str:
        """Return model name.

        Returns
        -------
        str
            Parameter vector

        """
        return "CIR"

    @staticmethod
    def get_names(subset: str = "all", measure: str = "PQ") -> list[str]:  # noqa: PLR0917, ARG004
        """Return parameter names.

        Returns
        -------
        (3, ) list of str
            Parameter names

        """
        return ["mean", "kappa", "eta"]

    def get_theta(self, subset: str = "all", measure: str = "PQ") -> np.ndarray:  # noqa: PLR0917, ARG002
        """Return vector of parameters.

        Returns
        -------
        (3, ) array
            Parameter vector

        """
        return np.array([self.mean, self.kappa, self.eta])
