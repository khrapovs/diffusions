#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Heston model class
~~~~~~~~~~~~~~~~~~

"""
from __future__ import print_function, division

import numpy as np
from statsmodels.tsa.tsatools import lagmat

from .generic_model import SDE
from .heston_param import HestonParam

__all__ = ['Heston']


class Heston(SDE):

    """Heston model.

    Methods
    -------

    """

    def __init__(self, theta_true=None):
        """Initialize the class.

        Parameters
        ----------
        theta_true : parameter instance
            True parameters used for simulation of the data

        """
        super(Heston, self).__init__(theta_true)

    @staticmethod
    def coef_big_a(param, aggh):
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
        return np.exp(-param.kappa * aggh)

    def coef_big_c(self, param, aggh):
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
        return param.mean_v * (1 - self.coef_big_a(param, aggh))

    def coef_small_a(self, param, aggh):
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
        return (1 - self.coef_big_a(param, aggh)) / param.kappa / aggh

    def coef_small_c(self, param, aggh):
        """Coefficient c_h in exact discretization of volatility.

        Parameters
        ----------
        theta : (nparams, ) array
            Parameter vector
        aggh : float
            Interval length

        Returns
        -------
        float

        """
        return param.mean_v * (1 - self.coef_small_a(param, aggh))

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
        return ((param.eta / param.kappa)**2
            * self.coef_small_c(param, aggh) / aggh + param.mean_v**2)

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
            + param.rho * param.eta / param.kappa
            * self.coef_small_c(param, aggh) / aggh)

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
            + self.mat_a2(param, 1))
            * self.depvar_unc_mean(param, aggh)).sum(1)[subset].squeeze()

    @staticmethod
    def mat_a0(param, aggh):
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
        mat_a = np.diag([1, 0, 0, 1]).astype(float)
        mat_a[1, 1] = -self.coef_big_a(param, 1) \
            * (1 + self.coef_big_a(param, 1))
        mat_a[3, 1] = .5 - param.lmbd
        return mat_a

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
        mat_a = np.diag([-self.coef_big_a(param, 1),
                         self.coef_big_a(param, 1)**3, 1,
                         -self.coef_big_a(param, 1)])
        mat_a[2, 0] = .5 - param.lmbd
        mat_a[3, 1] = (param.lmbd - .5) * self.coef_big_a(param, 1)
        return mat_a

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
        (nmoms, 3*nmoms) array
            Matrix A

        """
        mat_a = (self.mat_a0(param, 1),
                 self.mat_a1(param, 1),
                 self.mat_a2(param, 1))
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
        (nobs, 3*nmoms) array
            Dependend variables

        """
        ret, rvar = data
        var = np.vstack([rvar, rvar**2, ret, ret * rvar])[subset].squeeze()
        return lagmat(var.T, maxlag=2, original='in')

    @staticmethod
    def convert(theta, subset):
        """Convert parameter vector to instance.

        Parameters
        ----------
        theta : array
            Model parameters
        subset : str
            Which parameters to estimate. Belongs to ['all', 'vol']

        Returns
        -------
        param : HestonParam instance
            Model parameters
        subset_sl : slice
            Which moments to use

        """
        param = HestonParam()
        param.update(theta=theta, subset=subset)
        subset_sl = None
        if subset == 'vol':
            subset_sl = slice(2)
        return param, subset_sl
