"""Heston model class."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Sequence, cast

import numpy as np
from statsmodels.tsa.tsatools import lagmat

from affidiff.model_generic import SDE
from affidiff.param_heston import HestonParam

if TYPE_CHECKING:
    from affidiff.param_generic import GenericParam


class Heston(SDE):
    """Heston model."""

    param: HestonParam | None

    def __init__(self, param: HestonParam | None = None) -> None:
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
        assert self.param is not None
        return [1.0, float(self.param.mean_v)]

    @staticmethod
    def coef_big_a(*, param: HestonParam, aggh: float) -> float:
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

    def coef_big_c(self, *, param: HestonParam, aggh: float) -> float:
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
        return float(param.mean_v * (1 - self.coef_big_a(param=param, aggh=aggh)))

    def coef_small_a(self, *, param: HestonParam, aggh: float) -> float:
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
        return float((1 - self.coef_big_a(param=param, aggh=aggh)) / param.kappa / aggh)

    def coef_small_c(self, *, param: HestonParam, aggh: float) -> float:
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
        return float(param.mean_v * (1 - self.coef_small_a(param=param, aggh=aggh)))

    @staticmethod
    def mean_vol(*, param: GenericParam, aggh: float) -> float:
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
        _ = aggh
        assert isinstance(param, HestonParam)
        return float(param.mean_v)

    def mean_vol2(self, *, param: GenericParam, aggh: float) -> float:
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
        assert isinstance(param, HestonParam)
        return float(
            (param.eta / param.kappa) ** 2 * self.coef_small_c(param=param, aggh=aggh) / aggh + param.mean_v**2
        )

    @staticmethod
    def mean_ret(*, param: GenericParam, aggh: float) -> float:
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
        _ = aggh
        assert isinstance(param, HestonParam)
        return float((param.lmbd - 0.5) * param.mean_v)

    def mean_cross(self, *, param: GenericParam, aggh: float) -> float:
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
        assert isinstance(param, HestonParam)
        p = param
        val = (p.lmbd - 0.5) * self.mean_vol2(param=param, aggh=aggh) + (
            p.rho * p.eta / p.kappa * self.coef_small_c(param=param, aggh=aggh) / aggh
        )
        return float(val)

    def realized_const(
        self, *, param: GenericParam | None = None, aggh: float = 1, subset: slice | None = None
    ) -> np.ndarray:
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
        if param is None:
            param = self.param
        assert param is not None
        res = (
            (self.mat_a0(param=param, aggh=1) + self.mat_a1(param=param, aggh=1) + self.mat_a2(param=param, aggh=1))
            * self.depvar_unc_mean(param=param, aggh=aggh)
        ).sum(1)
        if subset is not None:
            res = res[subset]
        return np.squeeze(res)

    @staticmethod
    def mat_a0(*, param: Any, aggh: float) -> np.ndarray:  # noqa: ANN401
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
        _ = (param, aggh)
        return np.diag([0, 1, 0, 0]).astype(float)

    def mat_a1(self, *, param: Any, aggh: float) -> np.ndarray:  # noqa: ARG002, ANN401
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
        mat_a[1, 1] = -self.coef_big_a(param=param, aggh=1) * (1 + self.coef_big_a(param=param, aggh=1))
        mat_a[3, 1] = 0.5 - param.lmbd
        return mat_a

    def mat_a2(self, *, param: Any, aggh: float) -> np.ndarray:  # noqa: ARG002, ANN401
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
        mat_a = np.diag(
            [
                -self.coef_big_a(param=param, aggh=1),
                self.coef_big_a(param=param, aggh=1) ** 3,
                1,
                -self.coef_big_a(param=param, aggh=1),
            ]
        )
        mat_a[2, 0] = 0.5 - param.lmbd
        mat_a[3, 1] = (param.lmbd - 0.5) * self.coef_big_a(param=param, aggh=1)
        return mat_a

    def mat_a(self, *, param: GenericParam, subset: slice | None = None) -> np.ndarray:
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
        assert isinstance(param, HestonParam)
        mat_a_tuple = (
            self.mat_a0(param=param, aggh=1),
            self.mat_a1(param=param, aggh=1),
            self.mat_a2(param=param, aggh=1),
        )
        res = np.hstack(mat_a_tuple)
        if subset is not None:
            res = res[subset]
        return np.squeeze(res)

    @staticmethod
    def realized_depvar(*, data: np.ndarray | Sequence[np.ndarray], subset: slice | None = None) -> np.ndarray:
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
