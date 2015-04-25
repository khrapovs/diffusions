#!/usr/bin/env python
# -*- coding: utf-8 -*-
r"""
Central Tendency (CT) model for stochastic volatility
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The model is

.. math::
    dp_{t}&=\left(r+\left(\lambda-\frac{1}{2}\right)
        \sigma_{t}^{2}\right)dt+\sigma_{t}dW_{t}^{r},\\
    d\sigma_{t}^{2}&=\kappa_{\sigma}\left(v_{t}^{2}-\sigma_{t}^{2}\right)dt
        +\eta_{\sigma}\sigma_{t}dW_{t}^{\sigma},\\
    dv_{t}^{2}&=\kappa_{v}\left(\mu-v_{t}^{2}\right)dt+\eta_{v}v_{t}dW_{t}^{v},

with :math:`p_{t}=\log S_{t}`,
and :math:`Corr\left[dW_{s}^{r},dW_{s}^{\sigma}\right]=\rho`,
or in other words
:math:`W_{t}^{\sigma}=\rho W_{t}^{r}+\sqrt{1-\rho^{2}}W_{t}^{v}`.
Also let :math:`R\left(Y_{t}\right)=r`.

"""
from __future__ import print_function, division

import numpy as np
from statsmodels.tsa.tsatools import lagmat

from .generic_model import SDE
from .helper_functions import columnwise_prod
from .central_tendency_param import CentTendParam

__all__ = ['CentTend']


class CentTend(SDE):

    """Central Tendency model.

    Attributes
    ----------

    Methods
    -------

    """

    def __init__(self, theta_true=None):
        """Initialize the class.

        """
        super(CentTend, self).__init__(theta_true)

    def coef_big_as(self, param, aggh):
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

    def coef_big_av(self, param, aggh):
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
            Coefficient A^v_h

        """
        return np.exp(-param.kappa_v * aggh)

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
        return (1 - self.coef_big_a(param, aggh)) / param.kappa_s / aggh

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
        param = CentTendParam()
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
        param = CentTendParam()
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
        param = CentTendParam()
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
        param = CentTendParam()
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
        param = CentTendParam()
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
        param = CentTendParam()
        param.update(theta=theta)
        mat_a = (self.mat_a0(theta, 1), self.mat_a1(theta, 1),
                 self.mat_a2(theta, 1))
        return np.hstack(mat_a)

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

        return moms, None


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
    return param.mean_v * param.eta_v**2 / param.kappa_v / 2


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


def unc_var_error(param):
    """Unconditional variance of aggregated volatility error,
    :math:`V\left[\frac{1}{H}\int_{0}^{H}\epsilon_{t,s}^{\sigma}ds\right]`

    Parameters
    ----------
    param : parameter instance
        Model parameters

    Returns
    -------
    float

    """
    return param.mean_v**2 + unc_mean_sigma2(param)


if __name__ == '__main__':
    pass
