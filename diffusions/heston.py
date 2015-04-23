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
import numdifftools as nd

from .generic_model import SDE
from .heston_param import HestonParam
from .helper_functions import columnwise_prod

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
            Coefficient A_h

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
            Coefficient C_h

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
            Coefficient a_h

        """
        return (1 - self.coef_big_a(param, aggh)) / param.kappa / aggh

    def coef_small_c(self, param, aggh):
        """Coefficient c_h in exact discretization of volatility.

        Parameters
        ----------
        param : parameter instance
            Model parameters
        aggh : float
            Interval length

        Returns
        -------
        float
            Coefficient c_h

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
        mean_ret = (param.lmbd - .5) * param.mean_v

        mean_vol = param.mean_v

        mean_vol2 = ((param.eta / param.kappa)**2
                     * self.coef_small_c(param, aggh) / aggh
                     + param.mean_v**2)

        mean_cross = ((param.lmbd - .5) * mean_vol2
                      + param.rho * param.eta / param.kappa
                      * self.coef_small_c(param, aggh) / aggh)

        return np.array([mean_ret, mean_vol, mean_vol2, mean_cross])

    def realized_const(self, param, aggh):
        """Intercept in the realized moment conditions.

        Parameters
        ----------
        param : parameter instance
            Model parameters
        aggh : float
            Interval length

        Returns
        -------
        (4, ) array
            Intercept

        """
        return ((self.mat_a0(param, 1) + self.mat_a1(param, 1)
            + self.mat_a2(param, 1))
            * self.depvar_unc_mean(param, aggh)).sum(1)

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
        return np.diag([0, 0, 1, 0]).astype(float)

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
        mat_a = np.diag([0, 1, 0, 1]).astype(float)
        mat_a[2, 2] = -self.coef_big_a(param, aggh) \
            * (1 + self.coef_big_a(param, 1))
        mat_a[3, 2] = .5 - param.lmbd
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
        mat_a = np.diag([1, -self.coef_big_a(param, 1),
                         self.coef_big_a(param, 1)**3,
                         -self.coef_big_a(param, 1)])
        mat_a[0, 1] = .5 - param.lmbd
        mat_a[3, 2] = (param.lmbd - .5) * self.coef_big_a(param, 1)
        return mat_a

    def mat_a(self, param, aggh):
        """Matrix A in integrated moments.

        Parameters
        ----------
        param : parameter instance
            Model parameters
        aggh : float
            Interval length

        Returns
        -------
        (nmoms, 3*nmoms) array
            Matrix A

        """
        mat_a = (self.mat_a0(param, 1), self.mat_a1(param, 1),
                 self.mat_a2(param, 1))
        return np.hstack(mat_a)

    def diff_mat_a(self, param, aggh):
        """Derivative of Matrix A in integrated moments.

        Parameters
        ----------
        param : parameter instance
            Model parameters
        aggh : float
            Interval length

        Returns
        -------
        list of nmoms (3*nmoms, nparams) arrays
            Matrix A

        """
        diff = []
        for i in range(self.mat_a(param, aggh).shape[0]):
            # (3*nmoms, nparams)
            with np.errstate(divide='ignore'):
                diff.append(nd.Jacobian(lambda x:
                    self.mat_a(convert(x), aggh)[i])(param.get_theta()))
        return diff

    def drealized_const(self, param, aggh):
        """Intercept in the realized moment conditions.

        Parameters
        ----------
        param : parameter instance
            Model parameters
        aggh : float
            Interval length

        Returns
        -------
        (nmoms, nparams) array
            Intercept

        """
        with np.errstate(divide='ignore'):
            return nd.Jacobian(lambda x:
                self.realized_const(convert(x), aggh))(param.get_theta())

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
        param = HestonParam()
        param.update(theta=theta)

        ret, rvar = data
        lag = 2
        self.aggh = aggh
        # self.realized_depvar(data): (nobs, 3*nmoms)
        depvar = self.realized_depvar(data)[lag:]
        # (nobs - lag, 4) array
        error = depvar.dot(self.mat_a(param, aggh).T) \
            - self.realized_const(param, aggh)

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
            dconst = self.drealized_const(param, aggh)
            dmat_a = self.diff_mat_a(param, aggh)

            dmoms = []
            for i in range(instr.shape[1]):
                for mat_a, mat_c in zip(dmat_a, dconst):
                    left = (instr.T[i] * depvar.T).mean(1).dot(mat_a)
                    dmoms.append(left - mat_c)
            dmoms = np.vstack(dmoms)
        else:
            dmoms = None

        return moms, dmoms


def convert(theta):
    """Convert parameter vector to parameter instance.

    Parameters
    ----------
    theta : array
        Model parameters

    Returns
    -------
    param : parameter instance
        Model parameters

    """
    param = HestonParam()
    param.update(theta=theta)
    return param


if __name__ == '__main__':
    pass
