#!/usr/bin/env python
# -*- coding: utf-8 -*-
r"""
Heston model for stochastic volatility
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The model is

.. math::
    dp_{t}&=\left(r-\frac{1}{2}\sigma_{t}^{2}\right)dt+\sigma_{t}dW_{t}^{r},\\
    d\sigma_{t}^{2}&=\kappa\left(\mu-\sigma_{t}^{2}\right)dt
    +\eta\sigma_{t}dW_{t}^{\sigma},

with :math:`p_{t}=\log S_{t}`,
and :math:`Corr\left[dW_{s}^{r},dW_{s}^{\sigma}\right]=\rho`,
or in other words
:math:`W_{t}^{\sigma}=\rho W_{t}^{r}+\sqrt{1-\rho^{2}}W_{t}^{v}`.
Also let :math:`R\left(Y_{t}\right)=r`.

Feller condition for positivity of the volatility process is
:math:`\kappa\mu>\frac{1}{2}\eta^{2}`.


"""
from __future__ import print_function, division

import numpy as np
from statsmodels.tsa.tsatools import lagmat

from .generic_model import SDE
from .heston_param import HestonParam

__all__ = ['Heston']


class Heston(SDE):

    """Heston model.

    Attributes
    ----------

    Methods
    -------

    """

    def __init__(self, theta_true=None):
        """Initialize the class.

        """
        super(Heston, self).__init__(theta_true)

    def coef_big_a(self, param, aggh):
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

    def depvar_unc_mean(self, param, aggh):
        """Array of the left-hand side variables
        in realized moment conditions.

        Parameters
        ----------
        param : parameter instance
            Model parameters
        aggh : float
            Interval length

        Returns
        -------
        (nobs, 3*nmoms) array
            Dependend variables

        """
        mean_vol = param.mean_v

        mean_vol2 = ((param.eta / param.kappa)**2
                     * self.coef_small_c(param, aggh) / aggh
                     + param.mean_v**2)

        mean_ret = (param.lmbd - .5) * param.mean_v

        mean_cross = ((param.lmbd - .5) * mean_vol2
                      + param.rho * param.eta / param.kappa
                      * self.coef_small_c(param, aggh) / aggh)

        return np.array([mean_vol, mean_vol2, mean_ret, mean_cross])

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

    def realized_depvar(self, data, subset=None):
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

    def convert(self, theta, subset):
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


if __name__ == '__main__':
    pass
