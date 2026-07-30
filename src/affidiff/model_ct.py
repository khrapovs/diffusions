"""CT model class."""

from __future__ import annotations

from math import exp
from typing import TYPE_CHECKING, Sequence, cast

import numpy as np
from statsmodels.tsa.tsatools import lagmat

from affidiff.helper_functions import poly_coef
from affidiff.model_generic import SDE
from affidiff.param_ct import CentTendParam

if TYPE_CHECKING:
    from affidiff.param_generic import GenericParam


class CentTend(SDE):
    """Central Tendency model."""

    param: CentTendParam | None

    def __init__(self, param: CentTendParam | None = None) -> None:
        """Initialize the class.

        Parameters
        ----------
        param : parameter instance
            True parameters used for simulation of the data

        """
        if param is None:
            param = CentTendParam()
        super().__init__(param)

    def get_start(self) -> list[float]:
        """Return starting values for simulation."""
        assert self.param is not None
        return [1.0, float(self.param.mean_v), float(self.param.mean_v)]

    @staticmethod
    def coef_big_as(*, param: CentTendParam, aggh: float) -> float:
        r"""Coefficient A^\sigma_h in exact discretization of volatility.

        Parameters
        ----------
        param : parameter instance
            Model parameters
        aggh : float
            Interval length

        Returns
        -------
        float
            Coefficient A^\sigma_h

        """
        return float(np.exp(-param.kappa_s * aggh))

    def coef_big_bs(self, *, param: CentTendParam, aggh: float) -> float:
        r"""Coefficient B^\sigma_h in exact discretization of volatility.

        Parameters
        ----------
        param : parameter instance
            Model parameters
        aggh : float
            Interval length

        Returns
        -------
        float
            Coefficient B^\sigma_h

        """
        p = param
        return float(
            p.kappa_s
            / (p.kappa_s - p.kappa_y)
            * (self.coef_big_ay(param=param, aggh=aggh) - self.coef_big_as(param=param, aggh=aggh))
        )

    def coef_big_cs(self, *, param: CentTendParam, aggh: float) -> float:
        """Coefficient C^s_h in exact discretization of volatility.

        Parameters
        ----------
        param : parameter instance
            Model parameters
        aggh : float
            Interval length

        Returns
        -------
        float
            Coefficient C^s_h

        """
        return float(
            param.mean_v * (1 - self.coef_big_as(param=param, aggh=aggh) - self.coef_big_bs(param=param, aggh=aggh))
        )

    @staticmethod
    def coef_big_ay(*, param: CentTendParam, aggh: float) -> float:
        """Coefficient A^v_h in exact discretization of volatility.

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
        return float(np.exp(-param.kappa_y * aggh))

    def coef_big_cy(self, *, param: CentTendParam, aggh: float) -> float:
        """Coefficient C^y_h in exact discretization of volatility.

        Parameters
        ----------
        param : parameter instance
            Model parameters
        aggh : float
            Interval length

        Returns
        -------
        float
            Coefficient C^y_h

        """
        return float(param.mean_v * (1 - self.coef_big_ay(param=param, aggh=aggh)))

    def coef_small_as(self, *, param: CentTendParam, aggh: float) -> float:
        """Coefficient a^s_h in exact discretization of volatility.

        Parameters
        ----------
        param : parameter instance
            Model parameters
        aggh : float
            Interval length

        Returns
        -------
        float
            Coefficient a^s_h

        """
        return float((1 - self.coef_big_as(param=param, aggh=aggh)) / param.kappa_s / aggh)

    def coef_small_bs(self, *, param: CentTendParam, aggh: float) -> float:
        """Coefficient b^s_h in exact discretization of volatility.

        Parameters
        ----------
        param : parameter instance
            Model parameters
        aggh : float
            Interval length

        Returns
        -------
        float
            Coefficient b^s_h

        """
        p = param
        return float(
            p.kappa_s
            / (p.kappa_s - p.kappa_y)
            * (self.coef_small_ay(param=param, aggh=aggh) - self.coef_small_as(param=param, aggh=aggh))
        )

    def coef_small_cs(self, *, param: CentTendParam, aggh: float) -> float:
        """Coefficient c^s_h in exact discretization of volatility.

        Parameters
        ----------
        param : parameter instance
            Model parameters
        aggh : float
            Interval length

        Returns
        -------
        float
            Coefficient c^s_h

        """
        return float(
            param.mean_v * (1 - self.coef_small_as(param=param, aggh=aggh) - self.coef_small_bs(param=param, aggh=aggh))
        )

    def coef_small_ay(self, *, param: CentTendParam, aggh: float) -> float:
        """Coefficient a^v_h in exact discretization of volatility.

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
        return float((1 - self.coef_big_ay(param=param, aggh=aggh)) / param.kappa_y / aggh)

    def roots(self, *, param: CentTendParam, aggh: float) -> list[float]:
        r"""Roots of the polynomial in moment restrictions.

        .. math::

            \left(1-A_{2h}^{\sigma}L\right)
            \left(1-A_{2h}^{y}L\right)
            \left(1-A_{h}^{\sigma}A_{h}^{y}L\right)
            \left(1-A_{h}^{y}L\right)
            \left(1-A_{h}^{\sigma}L\right)

        Parameters
        ----------
        param : parameter instance
            Model parameters
        aggh : float
            Interval length

        Returns
        -------
        list of floats

        """
        return [
            self.coef_big_as(param=param, aggh=aggh),
            self.coef_big_ay(param=param, aggh=aggh),
            self.coef_big_as(param=param, aggh=aggh) ** 2,
            self.coef_big_ay(param=param, aggh=aggh) ** 2,
            self.coef_big_as(param=param, aggh=aggh) * self.coef_big_ay(param=param, aggh=aggh),
        ]

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
        assert isinstance(param, CentTendParam)
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
        assert isinstance(param, CentTendParam)
        return float(
            self.coef_small_as(param=param, aggh=aggh) ** 2 * unc_var_sigma(param)
            + self.coef_small_bs(param=param, aggh=aggh) ** 2 * unc_var_ct(param)
            + unc_var_error(param=param, aggh=aggh)
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
        assert isinstance(param, CentTendParam)
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
        assert isinstance(param, CentTendParam)
        p = param
        return float(
            (p.lmbd - 0.5) * self.mean_vol2(param=param, aggh=aggh)
            + p.rho * p.mean_v * p.eta_s / p.kappa_s * (1 - self.coef_small_as(param=param, aggh=aggh)) / aggh
        )

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
        assert isinstance(param, CentTendParam)
        res = (
            (
                self.mat_a0(param=param, aggh=1)
                + self.mat_a1(param=param, aggh=1)
                + self.mat_a2(param=param, aggh=1)
                + self.mat_a3(param=param, aggh=1)
                + self.mat_a4(param=param, aggh=1)
                + self.mat_a5(param=param, aggh=1)
            )
            * self.depvar_unc_mean(param=param, aggh=aggh)
        ).sum(1)
        if subset is not None:
            res = res[subset]
        return np.squeeze(res)

    def mat_a0(self, *, param: CentTendParam, aggh: float) -> np.ndarray:
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
        mat = np.zeros((4, 4))
        mat[1, 1] = poly_coef(self.roots(param=param, aggh=aggh))[0]
        return mat

    def mat_a1(self, *, param: CentTendParam, aggh: float) -> np.ndarray:
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
        mat = np.zeros((4, 4))
        mat[1, 1] = poly_coef(self.roots(param=param, aggh=aggh))[1]
        return mat

    def mat_a2(self, *, param: CentTendParam, aggh: float) -> np.ndarray:
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
        mat = np.zeros((4, 4))
        mat[1, 1] = poly_coef(self.roots(param=param, aggh=aggh))[2]
        return mat

    def mat_a3(self, *, param: CentTendParam, aggh: float) -> np.ndarray:
        """Matrix A_3 in integrated moments.

        Parameters
        ----------
        param : parameter instance
            Model parameters
        aggh : float
            Interval length

        Returns
        -------
        (4, 4) array
            Matrix A_3

        """
        mat = np.zeros((4, 4))
        mat[0, 0] = poly_coef(self.roots(param=param, aggh=aggh)[:2])[0]
        mat[1, 1] = poly_coef(self.roots(param=param, aggh=aggh))[3]
        mat[3, 1] = 0.5 - param.lmbd
        mat[3, 3] = mat[0, 0]
        return mat

    def mat_a4(self, *, param: CentTendParam, aggh: float) -> np.ndarray:
        """Matrix A_4 in integrated moments.

        Parameters
        ----------
        param : parameter instance
            Model parameters
        aggh : float
            Interval length

        Returns
        -------
        (4, 4) array
            Matrix A_4

        """
        mat = np.zeros((4, 4))
        mat[0, 0] = poly_coef(self.roots(param=param, aggh=aggh)[:2])[1]
        mat[1, 1] = poly_coef(self.roots(param=param, aggh=aggh))[4]
        mat[3, 1] = (0.5 - param.lmbd) * mat[0, 0]
        mat[3, 3] = mat[0, 0]
        return mat

    def mat_a5(self, *, param: CentTendParam, aggh: float) -> np.ndarray:
        """Matrix A_5 in integrated moments.

        Parameters
        ----------
        param : parameter instance
            Model parameters
        aggh : float
            Interval length

        Returns
        -------
        (4, 4) array
            Matrix A_5

        """
        mat = np.zeros((4, 4))
        mat[0, 0] = poly_coef(self.roots(param=param, aggh=aggh)[:2])[2]
        mat[1, 1] = poly_coef(self.roots(param=param, aggh=aggh))[5]
        mat[2, 2] = 1
        mat[3, 3] = mat[0, 0]
        mat[2, 0] = 0.5 - param.lmbd
        mat[3, 1] = (0.5 - param.lmbd) * mat[0, 0]
        return mat

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
        assert isinstance(param, CentTendParam)
        mat_a_tuple = (
            self.mat_a0(param=param, aggh=1),
            self.mat_a1(param=param, aggh=1),
            self.mat_a2(param=param, aggh=1),
            self.mat_a3(param=param, aggh=1),
            self.mat_a4(param=param, aggh=1),
            self.mat_a5(param=param, aggh=1),
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
        (nobs, 5*nmoms) array
            Dependend variables

        """
        data_arr = np.asarray(data)
        ret, rvar = data_arr[0], data_arr[1]
        var = np.vstack([rvar, rvar**2, ret, ret * rvar])
        if subset is not None:
            var = var[subset]
        var_s = np.squeeze(var)
        return cast(np.ndarray, lagmat(var_s.T, maxlag=5, original="in"))


def unc_mean_ct2(param: CentTendParam) -> float:
    """Calculate unconditional second moment of CT, E[y_t**4].

    Parameters
    ----------
    param : parameter instance
        Model parameters

    Returns
    -------
    float

    """
    p = param
    return float(p.mean_v * p.eta_y**2 / p.kappa_y / 2)


def unc_mean_sigma2(param: CentTendParam) -> float:
    r"""Calculate unconditional second moment of volatility, E[\sigma_t**4].

    Parameters
    ----------
    param : parameter instance
        Model parameters

    Returns
    -------
    float

    """
    p = param
    return float(unc_mean_ct2(param) * p.kappa_s / (p.kappa_s + p.kappa_y) + p.mean_v * p.eta_s**2 / p.kappa_s / 2)


def unc_var_ct(param: CentTendParam) -> float:
    """Calculate unconditional variance of CT, V[y_t**2].

    Parameters
    ----------
    param : parameter instance
        Model parameters

    Returns
    -------
    float

    """
    p = param
    return float(p.mean_v**2 + unc_mean_ct2(param))


def unc_var_sigma(param: CentTendParam) -> float:
    r"""Calculate unconditional variance of volatility, V[\sigma_t**2].

    Parameters
    ----------
    param : parameter instance
        Model parameters

    Returns
    -------
    float

    """
    p = param
    return float(p.mean_v**2 + unc_mean_sigma2(param))


def unc_var_error(*, param: CentTendParam, aggh: float) -> float:
    r"""Calculate unconditional variance of aggregated volatility error.

    :math:`V\left[\frac{1}{H}\int_{0}^{H}\epsilon_{t,s}^{\sigma}ds\right]`.

    Derived symbolically in symbolic.py

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
    mu = p.mean_v
    kappa_s = p.kappa_s
    kappa_y = p.kappa_y
    eta_s = p.eta_s
    eta_y = p.eta_y

    return float(
        mu
        * (
            eta_s**2
            * kappa_y**3
            * (kappa_s - kappa_y) ** 2
            * (kappa_s + kappa_y)
            * (2 * aggh * kappa_s * exp(2 * aggh * kappa_s) - 3 * exp(2 * aggh * kappa_s) + 4 * exp(aggh * kappa_s) - 1)
            * exp(2 * aggh * (kappa_s + 2 * kappa_y))
            + eta_y**2
            * kappa_s**2
            * (
                -(kappa_s**4) * exp(2 * aggh * kappa_s)
                + 4 * kappa_s**4 * exp(aggh * (2 * kappa_s + kappa_y))
                - kappa_s**3 * kappa_y * exp(2 * aggh * kappa_s)
                + 4 * kappa_s**2 * kappa_y**2 * exp(aggh * (kappa_s + kappa_y))
                - 4 * kappa_s**2 * kappa_y**2 * exp(aggh * (kappa_s + 2 * kappa_y))
                - 4 * kappa_s**2 * kappa_y**2 * exp(aggh * (2 * kappa_s + kappa_y))
                - kappa_s * kappa_y**3 * exp(2 * aggh * kappa_y)
                - kappa_y**4 * exp(2 * aggh * kappa_y)
                + 4 * kappa_y**4 * exp(aggh * (kappa_s + 2 * kappa_y))
                + (
                    2
                    * aggh
                    * kappa_s
                    * kappa_y
                    * (kappa_s**3 - kappa_s**2 * kappa_y - kappa_s * kappa_y**2 + kappa_y**3)
                    + kappa_s**3 * (kappa_s + kappa_y)
                    - 4 * kappa_s**2 * kappa_y**2
                    + 4 * kappa_s**2 * (-(kappa_s**2) + kappa_y**2)
                    + kappa_y**3 * (kappa_s + kappa_y)
                    + 4 * kappa_y**2 * (kappa_s**2 - kappa_y**2)
                )
                * exp(2 * aggh * (kappa_s + kappa_y))
            )
            * exp(2 * aggh * (kappa_s + kappa_y))
        )
        * exp(aggh * (-4 * kappa_s - 4 * kappa_y))
        / (2 * aggh**2 * kappa_s**3 * kappa_y**3 * (kappa_s - kappa_y) ** 2 * (kappa_s + kappa_y))
    )
