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
        self.update_ajd()
        if not self.is_valid():
            warnings.warn('Feller condition is violated!')

    def update_ajd(self):
        """Update AJD representation.

        """
        # AJD parameters
        self.mat_k0 = [self.riskfree, self.kappa * self.mean_v]
        self.mat_k1 = [[0, self.lmbd - .5], [0, -self.kappa]]
        self.mat_h0 = np.zeros((2, 2))
        self.mat_h1 = np.zeros((2, 2, 2))
        self.mat_h1[1] = [[1, self.eta*self.rho],
                    [self.eta*self.rho, self.eta**2]]

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
        self.update_ajd()

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

    def coef_big_a(self, theta, aggh):
        """Coefficient A_h in exact discretization of volatility.

        Parameters
        ----------
        theta : (nparams, ) array
            Parameter vector
        aggh : float
            Interval length

        Returns
        -------
        float
            Coefficient A_h

        """
        param = HestonParam()
        param.update(theta=theta)
        return np.exp(-param.kappa * aggh)

    def coef_big_c(self, theta, aggh):
        """Coefficient C_h in exact discretization of volatility.

        Parameters
        ----------
        theta : (nparams, ) array
            Parameter vector
        aggh : float
            Interval length

        Returns
        -------
        float
            Coefficient C_h

        """
        param = HestonParam()
        param.update(theta=theta)
        return param.mean_v * (1 - self.coef_big_a(theta, aggh))

    def coef_small_a(self, theta, aggh):
        """Coefficient a_h in exact discretization of volatility.

        Parameters
        ----------
        theta : (nparams, ) array
            Parameter vector
        aggh : float
            Interval length

        Returns
        -------
        float
            Coefficient a_h

        """
        param = HestonParam()
        param.update(theta=theta)
        return (1 - self.coef_big_a(theta, aggh)) / param.kappa / aggh

    def coef_small_c(self, theta, aggh):
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
            Coefficient c_h

        """
        param = HestonParam()
        param.update(theta=theta)
        return param.mean_v * (1 - self.coef_small_a(theta, aggh))

    def depvar_unc_mean(self, theta, aggh):
        """Array of the left-hand side variables
        in realized moment conditions.

        Parameters
        ----------
        data : (2, nobs) array
            Returns and realized variance
        aggh : float
            Interval length

        Returns
        -------
        (nobs, 3*nmoms) array
            Dependend variables

        """
        param = HestonParam()
        param.update(theta=theta)

        mean_ret = (param.lmbd - .5) * param.mean_v

        mean_vol = param.mean_v

        mean_vol2 = ((param.eta / param.kappa)**2
                     * self.coef_small_c(theta, aggh) / aggh
                     + param.mean_v**2)

        mean_cross = ((param.lmbd - .5) * mean_vol2
                      + param.rho * param.eta / param.kappa
                      * self.coef_small_c(theta, aggh) / aggh)

        return np.array([mean_ret, mean_vol, mean_vol2, mean_cross])

    def realized_const(self, theta, aggh):
        """Intercept in the realized moment conditions.

        Parameters
        ----------
        theta : (nparams, ) array
            Parameter vector
        aggh : float
            Interval length

        Returns
        -------
        (4, ) array
            Intercept

        """
        param = HestonParam()
        param.update(theta=theta)
        return ((self.mat_a0(theta, 1) + self.mat_a1(theta, 1)
            + self.mat_a2(theta, 1))
            * self.depvar_unc_mean(theta, aggh)).sum(1)

    def mat_a0(self, theta, aggh):
        """Matrix A_0 in integrated moments.

        Parameters
        ----------
        theta : (nparams, ) array
            Parameter vector
        aggh : float
            Interval length

        Returns
        -------
        (4, 4) array
            Matrix A_0

        """
        param = HestonParam()
        param.update(theta=theta)
        return np.diag([0, 0, 1, 0]).astype(float)

    def mat_a1(self, theta, aggh):
        """Matrix A_1 in integrated moments.

        Parameters
        ----------
        theta : (nparams, ) array
            Parameter vector
        aggh : float
            Interval length

        Returns
        -------
        (4, 4) array
            Matrix A_1

        """
        param = HestonParam()
        param.update(theta=theta)
        mat_a = np.diag([0, 1, 0, 1]).astype(float)
        mat_a[2, 2] = -self.coef_big_a(theta, aggh) \
            * (1 + self.coef_big_a(theta, 1))
        mat_a[3, 2] = .5 - param.lmbd
        return mat_a

    def mat_a2(self, theta, aggh):
        """Matrix A_2 in integrated moments.

        Parameters
        ----------
        theta : (nparams, ) array
            Parameter vector
        aggh : float
            Interval length

        Returns
        -------
        (4, 4) array
            Matrix A_2

        """
        param = HestonParam()
        param.update(theta=theta)
        mat_a = np.diag([1, -self.coef_big_a(theta, 1),
                         self.coef_big_a(theta, 1)**3,
                         -self.coef_big_a(theta, 1)])
        mat_a[0, 1] = .5 - param.lmbd
        mat_a[3, 2] = (param.lmbd - .5) * self.coef_big_a(theta, 1)
        return mat_a

    def mat_a(self, theta, aggh):
        """Matrix A in integrated moments.

        Parameters
        ----------
        theta : (nparams, ) array
            Parameter vector
        aggh : float
            Interval length

        Returns
        -------
        (nmoms, 3*nmoms) array
            Matrix A

        """
        param = HestonParam()
        param.update(theta=theta)
        mat_a = (self.mat_a0(theta, 1), self.mat_a1(theta, 1),
                 self.mat_a2(theta, 1))
        return np.hstack(mat_a)

    def diff_mat_a(self, theta, aggh):
        """Derivative of Matrix A in integrated moments.

        Parameters
        ----------
        theta : (nparams, ) array
            Parameter vector
        aggh : float
            Interval length

        Returns
        -------
        list of nmoms (3*nmoms, nparams) arrays
            Matrix A

        """
        param = HestonParam()
        param.update(theta=theta)
        diff = []
        for i in range(self.mat_a(theta, aggh).shape[0]):
            # (3*nmoms, nparams)
            with np.errstate(divide='ignore'):
                fun = lambda x: self.mat_a(x, aggh)[i]
                diff.append(nd.Jacobian(fun)(theta))
        return diff

    def drealized_const(self, theta, aggh):
        """Intercept in the realized moment conditions.

        Parameters
        ----------
        theta : (nparams, ) array
            Parameter vector
        aggh : float
            Interval length

        Returns
        -------
        (nmoms, nparams) array
            Intercept

        """
        with np.errstate(divide='ignore'):
            return nd.Jacobian(lambda x: self.realized_const(x, aggh))(theta)

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

    def integrated_mom(self, theta, data=None, instr_data=None,
                       instr_choice='const', aggh=1,
                       instrlag=1., exact_jacob=False, **kwargs):
        """Integrated moment function.

        Parameters
        ----------
        theta : array
            Model parameters
        data : (2, nobs) array
            Returns and realized variance
        instr_data : (ninstr, nobs) array
            Instruments (no lags)
        instrlag : int
            Number of lags for the instruments
        instr_choice : str {'const', 'var'}
            Choice of instruments.
                - 'const' : just a constant (unconditional moments)
                - 'var' : lags of instrument data
        aggh : int
            Number of intervals (days) to aggregate over using rolling mean
        exact_jacob : bool
            Whether to use exactly derived Jacobian (True)
            or numerical approximation (False)

        Returns
        -------
        moments : (nobs - instrlag - 2, 3 * ninstr = nmoms) array
            Moment restrictions
        dmoments : (nmoms, nparams) array
            Average derivative of the moment restrictions

        """
        ret, rvar = data
        lag = 2
        self.aggh = aggh
        # self.realized_depvar(data): (nobs, 3*nmoms)
        depvar = self.realized_depvar(data)[lag:]
        # (nobs - lag, 4) array
        error = depvar.dot(self.mat_a(theta, aggh).T) \
            - self.realized_const(theta, aggh)

        # self.instruments(data, instrlag=instrlag): (nobs, ninstr*instrlag+1)
        # (nobs-lag, ninstr*instrlag+1)
        if instr_choice == 'const':
            instr = self.instruments(nobs=rvar.size)[:-lag]
        else:
            instr = self.instruments(instr_data, instrlag=instrlag)[:-lag]
        # (nobs - instrlag - lag, 4 * (ninstr*instrlag + 1))
        moms = columnwise_prod(error, instr)

        if exact_jacob:
            # (nmoms, nparams)
            dconst = self.drealized_const(theta, aggh)
            dmat_a = self.diff_mat_a(theta, aggh)

            dmoms = []
            for i in range(instr.shape[1]):
                for mat_a, mat_c in zip(dmat_a, dconst):
                    left = (instr.T[i] * depvar.T).mean(1).dot(mat_a)
                    dmoms.append(left - mat_c)
            dmoms = np.vstack(dmoms)
        else:
            dmoms = None

        return moms, dmoms


if __name__ == '__main__':
    pass
