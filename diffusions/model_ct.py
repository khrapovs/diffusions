#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
CT model class
~~~~~~~~~~~~~~

"""
from __future__ import print_function, division

from math import exp

import numpy as np
from statsmodels.tsa.tsatools import lagmat

from .model_generic import SDE
from .param_ct import CentTendParam
from .helper_functions import poly_coef

__all__ = ['CentTend']


class CentTend(SDE):

    """Central Tendency model.

    """

    def __init__(self, param=None):
        """Initialize the class.

        Parameters
        ----------
        param : parameter instance
            True parameters used for simulation of the data

        """
        super(CentTend, self).__init__(param)

    @staticmethod
    def coef_big_as(param, aggh):
        """Coefficient A^\sigma_h in exact discretization of volatility.

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
        return np.exp(-param.kappa_s * aggh)

    def coef_big_bs(self, param, aggh):
        """Coefficient B^\sigma_h in exact discretization of volatility.

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
        return param.kappa_s / (param.kappa_s - param.kappa_y) \
            * (self.coef_big_ay(param, aggh) - self.coef_big_as(param, aggh))

    def coef_big_cs(self, param, aggh):
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
        return param.mean_v * (1 - self.coef_big_as(param, aggh)
            - self.coef_big_bs(param, aggh))

    @staticmethod
    def coef_big_ay(param, aggh):
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
        return np.exp(-param.kappa_y * aggh)

    def coef_big_cy(self, param, aggh):
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
        return param.mean_v * (1 - self.coef_big_ay(param, aggh))

    def coef_small_as(self, param, aggh):
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
        return (1 - self.coef_big_as(param, aggh)) / param.kappa_s / aggh

    def coef_small_bs(self, param, aggh):
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
        return param.kappa_s / (param.kappa_s - param.kappa_y) \
            * (self.coef_small_ay(param, aggh)
                - self.coef_small_as(param, aggh))

    def coef_small_cs(self, param, aggh):
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
        return param.mean_v * (1 - self.coef_small_as(param, aggh)
            - self.coef_small_bs(param, aggh))

    def coef_small_ay(self, param, aggh):
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
        return (1 - self.coef_big_ay(param, aggh)) / param.kappa_y / aggh

    def roots(self, param, aggh):
        """Roots of the polynomial in moment restrictions.

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
        return [self.coef_big_as(param, aggh),
                self.coef_big_ay(param, aggh),
                self.coef_big_as(param, aggh)**2,
                self.coef_big_ay(param, aggh)**2,
                self.coef_big_as(param, aggh) * self.coef_big_ay(param, aggh)]

    @staticmethod
    def mean_vol(param, aggh):
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
        return param.mean_v

    def mean_vol2(self, param, aggh):
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
        return (self.coef_small_as(param, aggh)**2 * unc_var_sigma(param)
            + self.coef_small_bs(param, aggh)**2 * unc_var_ct(param)
            + unc_var_error(param, aggh))

    @staticmethod
    def mean_ret(param, aggh):
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
        return (param.lmbd - .5) * param.mean_v

    def mean_cross(self, param, aggh):
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
        return ((param.lmbd - .5) * self.mean_vol2(param, aggh)
            + param.rho * param.mean_v * param.eta_s / param.kappa_s
            * (1 - self.coef_small_as(param, aggh)) / aggh)

    def realized_const(self, param, aggh, subset=None):
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
        return ((self.mat_a0(param, 1)
            + self.mat_a1(param, 1)
            + self.mat_a2(param, 1)
            + self.mat_a3(param, 1)
            + self.mat_a4(param, 1)
            + self.mat_a5(param, 1))
            * self.depvar_unc_mean(param, aggh)).sum(1)[subset].squeeze()

    def mat_a0(self, param, aggh):
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
        mat[1, 1] = poly_coef(self.roots(param, aggh))[0]
        return mat

    def mat_a1(self, param, aggh):
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
        mat[1, 1] = poly_coef(self.roots(param, aggh))[1]
        return mat

    def mat_a2(self, param, aggh):
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
        mat[1, 1] = poly_coef(self.roots(param, aggh))[2]
        return mat

    def mat_a3(self, param, aggh):
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
        mat[0, 0] = poly_coef(self.roots(param, aggh)[:2])[0]
        mat[1, 1] = poly_coef(self.roots(param, aggh))[3]
        mat[3, 1] = .5 - param.lmbd
        mat[3, 3] = mat[0, 0]
        return mat

    def mat_a4(self, param, aggh):
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
        mat[0, 0] = poly_coef(self.roots(param, aggh)[:2])[1]
        mat[1, 1] = poly_coef(self.roots(param, aggh))[4]
        mat[3, 1] = (.5 - param.lmbd) * mat[0, 0]
        mat[3, 3] = mat[0, 0]
        return mat

    def mat_a5(self, param, aggh):
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
        mat[0, 0] = poly_coef(self.roots(param, aggh)[:2])[2]
        mat[1, 1] = poly_coef(self.roots(param, aggh))[5]
        mat[2, 2] = 1
        mat[3, 3] = mat[0, 0]
        mat[2, 0] = .5 - param.lmbd
        mat[3, 1] = (.5 - param.lmbd) * mat[0, 0]
        return mat

    def mat_a(self, param, subset=None):
        """Matrix A in integrated moments.

        Parameters
        ----------
        param : parameter instance
            Model parameters
        subset : slice
            Which moments to use

        Returns
        -------
        (nmoms, 6*nmoms) array
            Matrix A

        """
        mat_a = (self.mat_a0(param, 1),
                 self.mat_a1(param, 1),
                 self.mat_a2(param, 1),
                 self.mat_a3(param, 1),
                 self.mat_a4(param, 1),
                 self.mat_a5(param, 1))
        return np.hstack(mat_a)[subset].squeeze()

    @staticmethod
    def realized_depvar(data, subset=None):
        """Array of the left-hand side variables
        in realized moment conditions.

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
        ret, rvar = data
        var = np.vstack([rvar, rvar**2, ret, ret * rvar])[subset].squeeze()
        return lagmat(var.T, maxlag=5, original='in')

    @staticmethod
    def convert(theta, subset='all', measure='P'):
        """Convert parameter vector to instance.

        Parameters
        ----------
        theta : array
            Model parameters
        subset : str
            Which parameters to estimate. Belongs to
                - 'all' : all parameters, including those related to returns
                - 'vol' : only those related to volatility
        measure : str
            Under which measure:
                - 'P' : physical measure
                - 'Q' : risk-neutral

        Returns
        -------
        param : CentTendParam instance
            Model parameters
        subset_sl : slice
            Which moments to use

        """
        param = CentTendParam()
        param.update(theta=theta, subset=subset, measure=measure)
        subset_sl = None
        if subset == 'vol':
            subset_sl = slice(2)
        return param, subset_sl


def unc_mean_ct2(param):
    """Unconditional second moment of CT, E[y_t**4].

    Parameters
    ----------
    param : parameter instance
        Model parameters

    Returns
    -------
    float

    """
    return param.mean_v * param.eta_y**2 / param.kappa_y / 2


def unc_mean_sigma2(param):
    """Unconditional second moment of volatility, E[\sigma_t**4].

    Parameters
    ----------
    param : parameter instance
        Model parameters

    Returns
    -------
    float

    """
    return unc_mean_ct2(param) * param.kappa_s \
        / (param.kappa_s + param.kappa_y) \
        + param.mean_v * param.eta_s**2 / param.kappa_s / 2


def unc_var_ct(param):
    """Unconditional variance of CT, V[y_t**2].

    Parameters
    ----------
    param : parameter instance
        Model parameters

    Returns
    -------
    float

    """
    return param.mean_v**2 + unc_mean_ct2(param)


def unc_var_sigma(param):
    """Unconditional variance of volatility, V[\sigma_t**2].

    Parameters
    ----------
    param : parameter instance
        Model parameters

    Returns
    -------
    float

    """
    return param.mean_v**2 + unc_mean_sigma2(param)


def unc_var_error(param, aggh):
    """Unconditional variance of aggregated volatility error,
    :math:`V\left[\frac{1}{H}\int_{0}^{H}\epsilon_{t,s}^{\sigma}ds\right]`

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
    mu = param.mean_v
    kappa_s = param.kappa_s
    kappa_y = param.kappa_y
    eta_s = param.eta_s
    eta_y = param.eta_y

    return (mu*(eta_s**2*kappa_y**3*(kappa_s - kappa_y)**2*(kappa_s +
        kappa_y)*(2*aggh*kappa_s*exp(2*aggh*kappa_s) - 3*exp(2*aggh*kappa_s) +
        4*exp(aggh*kappa_s) - 1)*exp(2*aggh*(kappa_s + 2*kappa_y)) +
        eta_y**2*kappa_s**2*(-kappa_s**4*exp(2*aggh*kappa_s) +
        4*kappa_s**4*exp(aggh*(2*kappa_s + kappa_y)) -
        kappa_s**3*kappa_y*exp(2*aggh*kappa_s) +
        4*kappa_s**2*kappa_y**2*exp(aggh*(kappa_s + kappa_y)) -
        4*kappa_s**2*kappa_y**2*exp(aggh*(kappa_s + 2*kappa_y)) -
        4*kappa_s**2*kappa_y**2*exp(aggh*(2*kappa_s + kappa_y)) -
        kappa_s*kappa_y**3*exp(2*aggh*kappa_y)
        - kappa_y**4*exp(2*aggh*kappa_y) +
        4*kappa_y**4*exp(aggh*(kappa_s + 2*kappa_y)) +
        (2*aggh*kappa_s*kappa_y*(kappa_s**3 - kappa_s**2*kappa_y -
        kappa_s*kappa_y**2 + kappa_y**3) + kappa_s**3*(kappa_s + kappa_y) -
        4*kappa_s**2*kappa_y**2 + 4*kappa_s**2*(-kappa_s**2 + kappa_y**2) +
        kappa_y**3*(kappa_s + kappa_y) + 4*kappa_y**2*(kappa_s**2 -
        kappa_y**2))*exp(2*aggh*(kappa_s + kappa_y)))*exp(2*aggh*(kappa_s +
        kappa_y)))*exp(aggh*(-4*kappa_s -
        4*kappa_y))/(2*aggh**2*kappa_s**3*kappa_y**3*(kappa_s -
        kappa_y)**2*(kappa_s + kappa_y)))


if __name__ == '__main__':
    pass
