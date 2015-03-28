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

__all__ = ['Heston', 'HestonParam']


class HestonParam(object):

    """Parameter storage for Heston model.

    Attributes
    ----------
    mean_r : float
        Instantaneous rate of return
    mean_v : float
        Mean of the volatility process
    kappa : float
        Mean reversion speed
    eta : float
        Instantaneous standard deviation of volatility
    rho : float
        Correlation

    """

    def __init__(self, riskfree=.0, lmbd = .1,
                 mean_v=.5, kappa=1.5, eta=.1, rho=-.5):
        """Initialize class.

        Parameters
        ----------
        riskfree : float
            Risk-free rate of return
        lmbd : float
            Equity risk premium
        mean_v : float
            Mean of the volatility process
        kappa : float
            Mean reversion speed
        eta : float
            Instantaneous standard deviation of volatility
        rho : float
            Correlation

        """
        self.riskfree = riskfree
        self.lmbd = lmbd
        self.mean_v = mean_v
        self.kappa = kappa
        self.eta = eta
        self.rho = rho
        # AJD parameters
        self.mat_k0 = [riskfree, kappa * mean_v]
        self.mat_k1 = [[0, lmbd - .5], [0, -kappa]]
        self.mat_h0 = np.zeros((2, 2))
        self.mat_h1 = [np.zeros((2, 2)),
                       [[1, eta*rho], [eta*rho, eta**2]]]

    def is_valid(self):
        """Check Feller condition.

        Returns
        -------
        bool
            True for valid parameters, False for invalid

        """
        return 2 * self.kappa * self.mean_v - self.eta**2 > 0

    def update(self, theta):
        """Update attributes from parameter vector.

        Parameters
        ----------
        theta : (6, ) array
            Parameter vector

        """
        [self.riskfree, self.lmbd, self.mean_v,
         self.kappa, self.eta, self.rho] = theta

    def get_theta(self):
        """Return vector of parameters.

        Returns
        -------
        (6, ) array
            Parameter vector

        """
        return [self.riskfree, self.lmbd, self.mean_v,
                self.kappa, self.eta, self.rho]


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

    def coef_big_a(self, theta):
        """Coefficient A_h in exact discretization of volatility.

        Parameters
        ----------
        theta : (6, ) array
            Parameter vector

        Returns
        -------
        float
            Coefficient A_h

        """
        param = HestonParam()
        param.update(theta=theta)
        return np.exp(-param.kappa * self.interval)

    def coef_small_a(self, theta):
        """Coefficient a_h in exact discretization of volatility.

        Parameters
        ----------
        theta : (6, ) array
            Parameter vector

        Returns
        -------
        float
            Coefficient a_h

        """
        param = HestonParam()
        param.update(theta=theta)
        return (1 - self.coef_big_a(theta)) / param.kappa

    def coef_big_c(self, theta):
        """Coefficient C_h in exact discretization of volatility.

        Parameters
        ----------
        theta : (6, ) array
            Parameter vector

        Returns
        -------
        float
            Coefficient C_h

        """
        param = HestonParam()
        param.update(theta=theta)
        return (1 - self.coef_big_a(theta)) * param.mean_v

    def coef_small_c(self, theta):
        """Coefficient c_h in exact discretization of volatility.

        Parameters
        ----------
        theta : (6, ) array
            Parameter vector

        Returns
        -------
        float
            Coefficient c_h

        """
        param = HestonParam()
        param.update(theta=theta)
        return (self.interval - self.coef_small_a(theta)) * param.mean_v

    def coef_d1(self, theta):
        """Coefficient D_1 in exact discretization of volatility.

        Parameters
        ----------
        theta : (6, ) array
            Parameter vector

        Returns
        -------
        float
            Coefficient D_1

        """
        param = HestonParam()
        param.update(theta=theta)
        return param.eta**2 / param.kappa * self.coef_big_a(theta) \
            * (1 - self.coef_big_a(theta))

    def coef_f1(self, theta):
        """Coefficient F_1 in exact discretization of volatility.

        Parameters
        ----------
        theta : (6, ) array
            Parameter vector

        Returns
        -------
        float
            Coefficient F_1

        """
        param = HestonParam()
        param.update(theta=theta)
        return param.mean_v * param.eta**2 / param.kappa / 2 \
            * (1 - self.coef_big_a(theta)**2) \
            * (1 - self.coef_big_a(theta))

    def coef_d2(self, theta):
        """Coefficient D_2 in exact discretization of volatility.

        Parameters
        ----------
        theta : (6, ) array
            Parameter vector

        Returns
        -------
        float
            Coefficient D_2

        """
        param = HestonParam()
        param.update(theta=theta)
        return (param.eta / param.kappa / self.interval)**2 \
            * ((1 - self.coef_big_a(theta)**2) / param.kappa \
            - 2 * self.interval * self.coef_big_a(theta))

    def coef_f2(self, theta):
        """Coefficient F_2 in exact discretization of volatility.

        Parameters
        ----------
        theta : (6, ) array
            Parameter vector

        Returns
        -------
        float
            Coefficient F_2

        """
        param = HestonParam()
        param.update(theta=theta)
        return param.mean_v * (param.eta / param.kappa / self.interval)**2 \
            * (self.interval * (1 + 2 * self.coef_big_a(theta)) \
            - (1 - self.coef_big_a(theta)) * (5 + self.coef_big_a(theta)) \
            / param.kappa / 2)

    def coef_d3(self, theta):
        """Coefficient D_3 in exact discretization of volatility.

        Parameters
        ----------
        theta : (6, ) array
            Parameter vector

        Returns
        -------
        float
            Coefficient D_3

        """
        param = HestonParam()
        param.update(theta=theta)
        return param.rho * param.eta / param.kappa / self.interval**2 \
            * ((1 - self.coef_big_a(theta)) / param.kappa \
            - self.coef_big_a(theta) * self.interval)

    def coef_f3(self, theta):
        """Coefficient F_3 in exact discretization of volatility.

        Parameters
        ----------
        theta : (6, ) array
            Parameter vector

        Returns
        -------
        float
            Coefficient F_3

        """
        param = HestonParam()
        param.update(theta=theta)
        return param.mean_v * param.rho * param.eta / param.kappa \
            / self.interval**2 \
            * ((1 + self.coef_big_a(theta)) * self.interval \
            - (1 - self.coef_big_a(theta)) / param.kappa * 2)

    def coef_r1(self, theta):
        """Coefficient R_1 in exact discretization of volatility.

        Parameters
        ----------
        theta : (6, ) array
            Parameter vector

        Returns
        -------
        float
            Coefficient R_1

        """
        param = HestonParam()
        param.update(theta=theta)
        return (1 - self.coef_big_a(theta)) \
            * (self.coef_f1(theta) + self.coef_big_c(theta)**2) \
            + (self.coef_d1(theta) \
            + 2 * self.coef_big_a(theta) * self.coef_big_c(theta)) \
            * self.coef_big_c(theta)

    def coef_r2(self, theta):
        """Coefficient R_2 in exact discretization of volatility.

        Parameters
        ----------
        theta : (6, ) array
            Parameter vector

        Returns
        -------
        float
            Coefficient R_2

        """
        param = HestonParam()
        param.update(theta=theta)
        return ((1 - self.coef_big_a(theta)) \
            * (self.coef_f2(theta) \
            + self.coef_small_c(theta) / self.interval**2) \
            + (self.coef_d2(theta) \
            + 2 * self.coef_small_a(theta) * self.coef_small_c(theta) \
            / self.interval**2) * self.coef_big_c(theta)) \
            * (1 - self.coef_big_a(theta)**2) \
            + self.coef_small_a(theta)**2 / self.interval**2 \
            * self.coef_r1(theta)

    def coef_r3(self, theta):
        """Coefficient R_3 in exact discretization of volatility.

        Parameters
        ----------
        theta : (6, ) array
            Parameter vector

        Returns
        -------
        float
            Coefficient R_3

        """
        param = HestonParam()
        param.update(theta=theta)
        return (param.lmbd - .5) * self.coef_r2(theta) \
            + (1 - self.coef_big_a(theta)**2) * self.coef_d3(theta) \
            * self.coef_big_c(theta) \
            + (1 - self.coef_big_a(theta)) * (1 - self.coef_big_a(theta)**2) \
            * self.coef_f3(theta)

    def realized_depvar(self, data):
        """Array of the left-hand side variables
        in realized moment conditions.

        Parameters
        ----------
        data : (2, nobs) array
            Returns and realized variance

        Returns
        -------
        (nobs, 3*4) array
            Dependend variables

        """
        ret, rvar = data
        var = np.vstack([ret, rvar, rvar**2, ret * rvar])
        return lagmat(var.T, maxlag=2, original=True)

    def realized_const(self, theta):
        """Intercept in the realized moment conditions.

        Parameters
        ----------
        theta : (6, ) array
            Parameter vector

        Returns
        -------
        array
            Intercept

        """
        param = HestonParam()
        param.update(theta=theta)
        return np.array([param.riskfree, self.coef_big_c(theta),
                         self.coef_r2(theta), self.coef_r3(theta)])

    def mat_a0(self, theta):
        """Matrix A_0 in integrated moments.

        Parameters
        ----------
        theta : (6, ) array
            Parameter vector

        Returns
        -------
        (4, 4) array
            Matrix A_0

        """
        param = HestonParam()
        param.update(theta=theta)
        return np.diag([0, 0, 1, 1])

    def mat_a1(self, theta):
        """Matrix A_1 in integrated moments.

        Parameters
        ----------
        theta : (6, ) array
            Parameter vector

        Returns
        -------
        (4, 4) array
            Matrix A_1

        """
        param = HestonParam()
        param.update(theta=theta)
        temp = -self.coef_big_a(theta) * (1 + self.coef_big_a(theta))
        return np.diag([0, 1, temp, temp])

    def mat_a2(self, theta):
        """Matrix A_2 in integrated moments.

        Parameters
        ----------
        theta : (6, ) array
            Parameter vector

        Returns
        -------
        (4, 4) array
            Matrix A_2

        """
        param = HestonParam()
        param.update(theta=theta)
        temp = self.coef_big_a(theta)**3
        mat_a = np.diag([1, -self.coef_big_a(theta), temp, temp])
        mat_a[0, 1] = param.lmbd - .5
        return mat_a

    def mat_a(self, theta):
        """Matrix A in integrated moments.

        Parameters
        ----------
        theta : (6, ) array
            Parameter vector

        Returns
        -------
        (4, 3*4) array
            Matrix A

        """
        param = HestonParam()
        param.update(theta=theta)
        mat_a = (self.mat_a0(theta), self.mat_a1(theta), self.mat_a2(theta))
        return np.hstack(mat_a)

    def integrated_mom(self, theta, data=None, instrlag=1):
        """Integrated moment function.

        Parameters
        ----------
        theta : array
            Model parameters
        data : (2, nobs) array
            Returns and realized variance
        instrlag : int
            Number of lags for the instruments

        Returns
        -------
        moments : (nobs, nmoms) array
            Moment restrictions
        dmoments : (nmoms, nparams) array
            Average derivative of the moment restrictions

        """
        ret, rvar = data
        # (nobs - instrlag, 4) array
        error = self.realized_depvar(data).dot(self.mat_a(theta).T) \
            - self.realized_const(theta)

        return error


if __name__ == '__main__':
    pass
