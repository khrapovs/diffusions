# !/usr/bin/env python
# -*- coding: utf-8 -*-
"""GBM parameter class."""

from __future__ import annotations

from typing import TYPE_CHECKING, Sequence

import numpy as np

from .param_generic import GenericParam

if TYPE_CHECKING:
    from typing_extensions import Self

__all__ = ["GBMparam"]


class GBMparam(GenericParam):
    """Parameter storage for GBM model.

    Attributes
    ----------
    mean : float
        Mean of the process
    sigma : float
        Instantaneous standard deviation
    measure : str
        Under which measure (P or Q)

    """

    def __init__(self, mean: float = 0.0, sigma: float = 0.2, measure: str = "P") -> None:  # noqa: PLR0917, ARG002
        """Initialize class.

        Parameters
        ----------
        mean : float
            Mean of the process
        sigma : float
            Instantaneous standard deviation
        measure : str

            Under which measure:
                - 'P' : physical measure
                - 'Q' : risk-neutral

        """
        super().__init__()
        self.mean = mean
        self.sigma = sigma
        self.measure = "P"
        self.update_ajd()

    def is_valid(self) -> bool:
        """Check validity of parameters.

        Returns
        -------
        bool
            True for valid parameters, False for invalid

        """
        return bool(self.sigma > 0)

    def update_ajd(self) -> None:
        """Update AJD representation."""
        # AJD parameters
        self.mat_k0 = self.mean - self.sigma**2 / 2
        self.mat_k1 = 0.0
        self.mat_h0 = self.sigma**2
        self.mat_h1 = 0.0

    @classmethod
    def from_theta(cls, theta: np.ndarray | Sequence[float]) -> Self:
        """Initialize parameters from parameter vector.

        Parameters
        ----------
        theta : (nparams, ) array
            Parameter vector

        """
        param = cls(mean=float(theta[0]), sigma=float(theta[1]))
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
        self.mean, self.sigma = float(theta[0]), float(theta[1])
        self.update_ajd()

    @staticmethod
    def get_model_name() -> str:
        """Return model name.

        Returns
        -------
        str
            Parameter vector

        """
        return "GBM"

    @staticmethod
    def get_names(subset: str = "all", measure: str = "PQ") -> list[str]:  # noqa: PLR0917, ARG004
        """Return parameter names.

        Returns
        -------
        (2, ) list of str
            Parameter names

        """
        return ["mean", "sigma"]

    def get_theta(self, subset: str = "all", measure: str = "PQ") -> np.ndarray:  # noqa: PLR0917, ARG002
        """Return vector of parameters.

        Returns
        -------
        (2, ) array
            Parameter vector

        """
        return np.array([self.mean, self.sigma])
