"""CT model class."""

from __future__ import annotations

from math import expm1
from typing import TYPE_CHECKING, Callable, Sequence, cast

import numpy as np
from statsmodels.tsa.tsatools import lagmat

from affidiff.helper_functions import poly_coef
from affidiff.models.generic import SDE
from affidiff.parameters.ct import CentTendParam

if TYPE_CHECKING:
    from affidiff.parameters.generic import GenericParam


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

    @property
    def series_names(self) -> tuple[str, ...]:
        """Names of the simulated series, one per state variable in simulation order."""
        return ("log(Price)", "Volatility", "Central tendency of volatility")

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

    @staticmethod
    def mean_vol2(*, param: GenericParam, aggh: float) -> float:
        r"""Unconditional mean of squared realized volatiliy.

        .. math::

            E\left[IV_h^2\right] = \mu^2 + \frac{2}{h^2}\int_0^h (h - \tau) C(\tau) d\tau,

        where :math:`C(\tau)` is the autocovariance of :math:`\sigma_t^2`.

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
        x_s, x_y = p.kappa_s * aggh, p.kappa_y * aggh
        cov_sigma_ct = p.kappa_s / (p.kappa_s + p.kappa_y) * unc_mean_ct2(param)
        var_iv = unc_mean_sigma2(param) * _compute_integrated_autocorr(x_s) - cov_sigma_ct * x_s * (
            _compute_divided_difference(func=_compute_integrated_autocorr, x=x_s, y=x_y)
        )
        return float(p.mean_v**2 + var_iv)

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
    """Calculate unconditional variance of CT, V[y_t].

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
    r"""Calculate unconditional variance of volatility, V[\sigma_t^2].

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


def _compute_integrated_autocorr(x: float) -> float:
    r"""Compute :math:`\frac{2}{h^2}\int_0^h (h - \tau) e^{-\kappa\tau} d\tau` with :math:`x = \kappa h`."""
    return 2 * (x + expm1(-x)) / x**2


def _compute_divided_difference(*, func: Callable[[float], float], x: float, y: float) -> float:
    """Compute (func(x) - func(y)) / (x - y), falling back to a central derivative when x and y nearly coincide."""
    half_step = max(abs(x - y), 1e-6 * max(abs(x), abs(y))) / 2
    mid = (x + y) / 2
    return (func(mid + half_step) - func(mid - half_step)) / (2 * half_step)
