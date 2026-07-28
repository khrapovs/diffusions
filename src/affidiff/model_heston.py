# !/usr/bin/env python
# -*- coding: utf-8 -*-
"""Heston model class."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

import numpy as np
from statsmodels.tsa.tsatools import lagmat

from .model_generic import SDE
from .param_heston import HestonParam

if TYPE_CHECKING:
    pass

__all__ = ["Heston"]


class Heston(SDE):
    """Heston model."""

    param: Any

    def __init__(self, param: Any = None) -> None:  # noqa: ANN401
        """Initialize the class.

        Parameters
        ----------
        param : parameter instance
            True parameters used for simulation of the data

        """
        if param is None:
            param = HestonParam()
        super().__init__(param)

    def get_start(self) -> list[float]:
        """Get starting values for simulation.

        Returns
        -------
        array_like
            Starting values for price and variance

        """
        return [1.0, float(self.param.mean_v)]

    @staticmethod
    def coef_big_a(param: Any, aggh: float) -> float:  # noqa: PLR0917, ANN401
        """Coefficient A_h in exact discretization of volatility.

        Parameters
        ----------
        param : parameter instance
            Model parameters
        aggh : float
            Interval length

        Returns
        -------
        float

        """
        return float(np.exp(-param.kappa * aggh))

    def coef_big_c(self, param: Any, aggh: float) -> float:  # noqa: PLR0917, ANN401
        """Coefficient C_h in exact discretization of volatility.

        Parameters
        ----------
        param : parameter instance
            Model parameters
        aggh : float
            Interval length

        Returns
        -------
        float

        """
        return float(param.mean_v * (1 - self.coef_big_a(param, aggh)))

    def coef_small_a(self, param: Any, aggh: float) -> float:  # noqa: PLR0917, ANN401
        """Coefficient a_h in exact discretization of volatility.

        Parameters
        ----------
        param : parameter instance
            Model parameters
        aggh : float
            Interval length

        Returns
        -------
        float

        """
        return float((1 - self.coef_big_a(param, aggh)) / param.kappa / aggh)

    def coef_small_c(self, param: Any, aggh: float) -> float:  # noqa: PLR0917, ANN401
        """Coefficient c_h in exact discretization of volatility.

        Parameters
        ----------
        param : parameter instance
            Parameter vector
        aggh : float
            Interval length

        Returns
        -------
        float

        """
        return float(param.mean_v * (1 - self.coef_small_a(param, aggh)))

    @staticmethod
    def mean_vol(param: Any, aggh: float) -> float:  # noqa: PLR0917, ARG004, ANN401
        """Unconditional mean of realized volatiliy.

        Parameters
        ----------
        param : parameter instance
            Model parameters
        aggh : float
            Interval length

        Returns
        -------
        float

        """
        return float(param.mean_v)

    def mean_vol2(self, param: Any, aggh: float) -> float:  # noqa: PLR0917, ANN401
        """Unconditional mean of squared realized volatiliy.

        Parameters
        ----------
        param : parameter instance
            Model parameters
        aggh : float
            Interval length

        Returns
        -------
        float

        """
        return float((param.eta / param.kappa) ** 2 * self.coef_small_c(param, aggh) / aggh + param.mean_v**2)

    @staticmethod
    def mean_ret(param: Any, aggh: float) -> float:  # noqa: PLR0917, ARG004, ANN401
        """Unconditional mean of realized returns.

        Parameters
        ----------
        param : parameter instance
            Model parameters
        aggh : float
            Interval length

        Returns
        -------
        float

        """
        return float((param.lmbd - 0.5) * param.mean_v)

    def mean_cross(self, param: Any, aggh: float) -> float:  # noqa: PLR0917, ANN401
        """Unconditional mean of realized returns times volatility.

        Parameters
        ----------
        param : parameter instance
            Model parameters
        aggh : float
            Interval length

        Returns
        -------
        float

        """
        p = param
        val = (p.lmbd - 0.5) * self.mean_vol2(param, aggh) + (
            p.rho * p.eta / p.kappa * self.coef_small_c(param, aggh) / aggh
        )
        return float(val)

    def realized_const(self, param: Any = None, aggh: float = 1, subset: slice | None = None) -> np.ndarray:  # noqa: PLR0917, ANN401
        """Intercept in the realized moment conditions.

        Parameters
        ----------
        param : parameter instance
            Model parameters
        aggh : float
            Interval length
        subset : slice
            Which moments to use

        Returns
        -------
        (4, ) array
            Intercept

        """
        res = (
            (self.mat_a0(param, 1) + self.mat_a1(param, 1) + self.mat_a2(param, 1)) * self.depvar_unc_mean(param, aggh)
        ).sum(1)
        if subset is not None:
            res = res[subset]
        return np.squeeze(res)

    @staticmethod
    def mat_a0(param: Any, aggh: float) -> np.ndarray:  # noqa: PLR0917, ARG004, ANN401
        """Matrix A_0 in integrated moments.

        Parameters
        ----------
        param : parameter instance
            Model parameters
        aggh : float
            Interval length

        Returns
        -------
        (4, 4) array
            Matrix A_0

        """
        return np.diag([0, 1, 0, 0]).astype(float)

    def mat_a1(self, param: Any, aggh: float) -> np.ndarray:  # noqa: PLR0917, ARG002, ANN401
        """Matrix A_1 in integrated moments.

        Parameters
        ----------
        param : parameter instance
            Model parameters
        aggh : float
            Interval length

        Returns
        -------
        (4, 4) array
            Matrix A_1

        """
        mat_a = np.diag([1, 0, 0, 1]).astype(float)
        mat_a[1, 1] = -self.coef_big_a(param, 1) * (1 + self.coef_big_a(param, 1))
        mat_a[3, 1] = 0.5 - param.lmbd
        return mat_a

    def mat_a2(self, param: Any, aggh: float) -> np.ndarray:  # noqa: PLR0917, ARG002, ANN401
        """Matrix A_2 in integrated moments.

        Parameters
        ----------
        param : parameter instance
            Model parameters
        aggh : float
            Interval length

        Returns
        -------
        (4, 4) array
            Matrix A_2

        """
        mat_a = np.diag([-self.coef_big_a(param, 1), self.coef_big_a(param, 1) ** 3, 1, -self.coef_big_a(param, 1)])
        mat_a[2, 0] = 0.5 - param.lmbd
        mat_a[3, 1] = (param.lmbd - 0.5) * self.coef_big_a(param, 1)
        return mat_a

    def mat_a(self, param: Any, subset: slice | None = None) -> np.ndarray:  # noqa: PLR0917, ANN401
        """Matrix A in integrated moments.

        Parameters
        ----------
        param : parameter instance
            Model parameters
        subset : slice
            Which moments to use

        Returns
        -------
        (nmoms, 3*nmoms) array
            Matrix A

        """
        mat_a_tuple = (self.mat_a0(param, 1), self.mat_a1(param, 1), self.mat_a2(param, 1))
        res = np.hstack(mat_a_tuple)
        if subset is not None:
            res = res[subset]
        return np.squeeze(res)

    @staticmethod
    def realized_depvar(data: Any, subset: slice | None = None) -> np.ndarray:  # noqa: PLR0917, ANN401
        """Array of the left-hand side variables in realized moment conditions.

        Parameters
        ----------
        data : (2, nobs) array
            Returns and realized variance
        subset : slice
            Which moments to use

        Returns
        -------
        (nobs, 3*nmoms) array
            Dependend variables

        """
        data_arr = np.asarray(data)
        ret, rvar = data_arr[0], data_arr[1]
        var = np.vstack([rvar, rvar**2, ret, ret * rvar])
        if subset is not None:
            var = var[subset]
        var_s = np.squeeze(var)
        return cast(np.ndarray, lagmat(var_s.T, maxlag=2, original="in"))
