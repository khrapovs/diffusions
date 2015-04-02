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

import warnings

import numpy as np
from statsmodels.tsa.tsatools import lagmat
import numdifftools as nd

from .generic_model import SDE
from .helper_functions import columnwise_prod

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
        theta : (nparams, ) array
            Parameter vector

        """
        [self.lmbd, self.mean_v, self.kappa, self.eta, self.rho] = theta

    def get_theta(self):
        """Return vector of model parameters.

        Returns
        -------
        (nparams, ) array
            Parameter vector

        """
        return np.array([self.lmbd, self.mean_v,
                         self.kappa, self.eta, self.rho])

    def get_bounds(self):
        """Bounds on parameters.

        Returns
        -------
        sequence of (min, max) tuples

        """
        lb = [None, 1e-5, 1e-5, 1e-5, -1]
        ub = [None, None, None, None, 1]
        return list(zip(lb, ub))


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
        theta : (nparams, ) array
            Parameter vector

        Returns
        -------
        float
            Coefficient A_h

        """
        param = HestonParam()
        param.update(theta=theta)
        h = 1
        return np.exp(-param.kappa * h)

    def coef_small_a(self, theta):
        """Coefficient a_h in exact discretization of volatility.

        Parameters
        ----------
        theta : (nparams, ) array
            Parameter vector

        Returns
        -------
        float
            Coefficient a_h

        """
        param = HestonParam()
        param.update(theta=theta)
        return (1 - self.coef_big_a(theta)) / param.kappa

    def depvar_unc_mean(self, theta):
        """Array of the left-hand side variables
        in realized moment conditions.

        Parameters
        ----------
        data : (2, nobs) array
            Returns and realized variance

        Returns
        -------
        (nobs, 3*nmoms) array
            Dependend variables

        """
        param = HestonParam()
        param.update(theta=theta)
        h = 1

        mean_ret = (param.lmbd - .5) * param.mean_v

        mean_vol = param.mean_v

        mean_vol2 = (param.mean_v * (param.eta / param.kappa)**2
                     * (1 - self.coef_small_a(theta) / h) / h
                     + param.mean_v**2)

        mean_cross = ((param.lmbd - .5) * mean_vol2
                      + param.rho * param.mean_v * param.eta / param.kappa / h
                      * (1 - self.coef_small_a(theta) / h))

        return np.array([mean_ret, mean_vol, mean_vol2, mean_cross])

    def realized_const(self, theta):
        """Intercept in the realized moment conditions.

        Parameters
        ----------
        theta : (nparams, ) array
            Parameter vector

        Returns
        -------
        (4, ) array
            Intercept

        """
        param = HestonParam()
        param.update(theta=theta)
        return ((self.mat_a0(theta) + self.mat_a1(theta) + self.mat_a2(theta))
            * self.depvar_unc_mean(theta)).sum(1)

    def mat_a0(self, theta):
        """Matrix A_0 in integrated moments.

        Parameters
        ----------
        theta : (nparams, ) array
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
        theta : (nparams, ) array
            Parameter vector

        Returns
        -------
        (4, 4) array
            Matrix A_1

        """
        param = HestonParam()
        param.update(theta=theta)
        mat_a1 = np.diag([0, 1, 0, 1])
        mat_a1[2, 2] = -self.coef_big_a(theta) * (1 + self.coef_big_a(theta))
        mat_a1[3, 2] = .5 - param.lmbd
        return mat_a1

    def mat_a2(self, theta):
        """Matrix A_2 in integrated moments.

        Parameters
        ----------
        theta : (nparams, ) array
            Parameter vector

        Returns
        -------
        (4, 4) array
            Matrix A_2

        """
        param = HestonParam()
        param.update(theta=theta)
        mat_a = np.diag([1, -self.coef_big_a(theta),
                         self.coef_big_a(theta)**3, -self.coef_big_a(theta)])
        mat_a[0, 1] = .5 - param.lmbd
        mat_a[3, 2] = (param.lmbd - .5) * self.coef_big_a(theta)
        return mat_a

    def mat_a(self, theta):
        """Matrix A in integrated moments.

        Parameters
        ----------
        theta : (nparams, ) array
            Parameter vector

        Returns
        -------
        (nmoms, 3*nmoms) array
            Matrix A

        """
        param = HestonParam()
        param.update(theta=theta)
        mat_a = (self.mat_a0(theta), self.mat_a1(theta), self.mat_a2(theta))
        return np.hstack(mat_a)

    def diff_mat_a(self, theta):
        """Derivative of Matrix A in integrated moments.

        Parameters
        ----------
        theta : (nparams, ) array
            Parameter vector

        Returns
        -------
        list of nmoms (3*nmoms, nparams) arrays
            Matrix A

        """
        param = HestonParam()
        param.update(theta=theta)
        diff = []
        for i in range(self.mat_a(theta).shape[0]):
            # (nparams, 3*nmoms)
            try:
                diff.append(nd.Jacobian(lambda x: self.mat_a(x)[i])(theta))
            except:
                print('Bad!')
        return diff

    def drealized_const(self, theta):
        """Intercept in the realized moment conditions.

        Parameters
        ----------
        theta : (nparams, ) array
            Parameter vector

        Returns
        -------
        (nmoms, nparams) array
            Intercept

        """
        try:
            return nd.Jacobian(self.realized_const)(theta)
        except:
            print('bad')

    def realized_depvar(self, data):
        """Array of the left-hand side variables
        in realized moment conditions.

        Parameters
        ----------
        data : (2, nobs) array
            Returns and realized variance

        Returns
        -------
        (nobs, 3*nmoms) array
            Dependend variables

        """
        ret, rvar = data
        var = np.vstack([ret, rvar, rvar**2, ret * rvar])
        return lagmat(var.T, maxlag=2, original='in')

    def instruments(self, data=None, instrlag=0, nobs=None):
        """Create an array of instruments.

        Parameters
        ----------
        data : (ninstr, nobs) array
            Returns and realized variance
        instrlag : int
            Number of lags for the instruments

        Returns
        -------
        (nobs, ninstr*instrlag + 1) array
            Instrument array

        """
        if data is None:
            return np.ones((nobs, 1))
        else:
            instr = lagmat(np.atleast_2d(data).T, maxlag=instrlag)
            width = ((0, 0), (1, 0))
            return np.pad(instr, width, mode='constant', constant_values=1)

    def integrated_mom(self, theta, data=None, instr_choice='const',
                       instrlag=1., **kwargs):
        """Integrated moment function.

        Parameters
        ----------
        theta : array
            Model parameters
        data : (2, nobs) array
            Returns and realized variance
        instrlag : int
            Number of lags for the instruments
        instr_choice : str {'const', 'var'}
            Choice of instruments.
                - 'const' : just a constant (unconditional moments)
                - 'var' : lags of data

        Returns
        -------
        moments : (nobs - instrlag - 2, 3 * ninstr = nmoms) array
            Moment restrictions
        dmoments : (nmoms, nparams) array
            Average derivative of the moment restrictions

        """
        ret, rvar = data
        lag = 2
        # self.realized_depvar(data): (nobs, 3*nmoms)
        depvar = self.realized_depvar(data)[lag:]
        # (nobs - lag, 4) array
        error = depvar.dot(self.mat_a(theta).T) - self.realized_const(theta)

        # self.instruments(data, instrlag=instrlag): (nobs, ninstr*instrlag+1)
        # (nobs-lag, ninstr*instrlag+1)
        if instr_choice == 'const':
            instr = self.instruments(nobs=rvar.size)[:-lag]
        else:
            instr = self.instruments(rvar, instrlag=instrlag)[:-lag]
        # (nobs - instrlag - lag, 4 * (ninstr*instrlag + 1))
        moms = columnwise_prod(error, instr)
        if moms.shape[1] <= 5:
            warnings.warn("Not enough degrees of freedom!")

        # (nmoms, nparams)
        dconst = self.drealized_const(theta)
        diff_mat_a = self.diff_mat_a(theta)

        dmoms = []
        for i in range(instr.shape[1]):
            for mat_a, mat_c in zip(diff_mat_a, dconst):
                left = (instr.T[i] * depvar.T).mean(1).dot(mat_a)
                dmoms.append(left - mat_c)
        dmoms = np.vstack(dmoms)

        return moms, dmoms


if __name__ == '__main__':
    pass
