#!/usr/bin/env python
# -*- coding: utf-8 -*-
r"""
GBM model class
~~~~~~~~~~~~~~~

"""
from __future__ import print_function, division

import numpy as np
import numdifftools as nd

from statsmodels.tsa.tsatools import lagmat

from .generic_model import SDE
from .helper_functions import columnwise_prod
from .gbm_param import GBMparam

__all__ = ['GBM']


class GBM(SDE):

    """Geometric Brownian Motion.

    """

    def __init__(self, theta_true=None):
        """Initialize the class.

        Parameters
        ----------
        theta_true : parameter instance
            True parameters used for simulation of the data

        """
        super(GBM, self).__init__(theta_true)

    @staticmethod
    def drift(state, theta):
        """Drift function.

        Parameters
        ----------
        state : (nvars, nsim) array_like
            Current value of the process
        theta : parameter instance
            Model parameter

        Returns
        -------
        scalar
            Drift value

        """
        return theta.mean - theta.sigma**2/2

    @staticmethod
    def diff(state, theta):
        """Diffusion (instantaneous volatility) function.

        Parameters
        ----------
        state : (nvars, nsim) array_like
            Current value of the process
        theta : parameter instance
            Model parameter

        Returns
        -------
        scalar
            Diffusion value

        """
        return theta.sigma

    def betamat(self, theta):
        """Coefficients in linear representation of the first moment.

        Parameters
        ----------
        theta : array
            Parameters

        Returns
        -------
        float
            Constant coefficient

        """
        param = GBMparam()
        param.update(theta=theta)
        loc = float(self.exact_loc(0, param))
        return np.array([loc, 0], dtype=float)

    def gammamat(self, theta):
        """Coefficients in linear representation of the second moment.

        Parameters
        ----------
        theta : array
            Parameters

        Returns
        -------
        float
            Constant coefficient

        """
        param = GBMparam()
        param.update(theta=theta)
        loc = float(self.exact_loc(0, param))
        scale = float(self.exact_scale(0, param))
        return np.array([loc**2 + scale**2, 0], dtype=float)

    def dbetamat(self, theta):
        """Derivative of the first moment coefficients (numerical).

        Parameters
        ----------
        theta : array
            Parameters

        Returns
        -------
        array
            Derivatives of the coefficient

        """
        with np.errstate(divide='ignore'):
            return nd.Jacobian(self.betamat)(theta)

    def dgammamat(self, theta):
        """Derivative of the second moment coefficients (numerical).

        Parameters
        ----------
        theta : array
            Parameters

        Returns
        -------
        array
            Derivatives of the coefficient

        """
        with np.errstate(divide='ignore'):
            return nd.Jacobian(self.gammamat)(theta)

    def dbetamat_exact(self, theta):
        """Derivative of the first moment coefficients (exact).

        Parameters
        ----------
        theta : array
            Parameters

        Returns
        -------
        array
            Derivatives of the coefficient

        """
        mean, sigma = theta
        return np.array([[self.interval, - sigma * self.interval], [0, 0]])

    def dgammamat_exact(self, theta):
        """Derivative of the second moment coefficients (exact).

        Parameters
        ----------
        theta : array
            Parameters

        Returns
        -------
        array
            Derivatives of the coefficient

        """
        mean, sigma = theta
        return np.array([[2 * self.interval**2 * (mean - sigma**2/2),
                         2 * sigma * self.interval
                         - 2 * sigma * self.interval**2
                         * (mean - sigma**2/2)], [0, 0]])

    @staticmethod
    def realized_depvar(data):
        """Array of the left-hand side variables
        in realized moment conditions.

        Parameters
        ----------
        data : (2, nobs) array
            Returns and realized variance

        Returns
        -------
        (3, nobs) array
            Dependend variables

        """
        ret, rvar = data
        return np.vstack([ret, rvar, rvar**2])

    @staticmethod
    def realized_const(theta):
        """Intercept in the realized moment conditions.

        Parameters
        ----------
        theta : array
            Parameters

        Returns
        -------
        array
            Intercept

        """
        mean, sigma = theta
        return np.array([mean - sigma**2/2, sigma**2, sigma**4])

    def drealized_const(self, theta):
        """Derivative of the intercept in the realized moment conditions.

        Parameters
        ----------
        theta : array
            Parameters

        Returns
        -------
        (nparams, nintercepts) array
            Derivatives of the coefficient

        """
        with np.errstate(divide='ignore'):
            return nd.Jacobian(self.realized_const)(theta)

    @staticmethod
    def instruments(data, instrlag=1):
        """Create an array of instruments.

        Parameters
        ----------
        data : (2, nobs) array
            Returns and realized variance
        instrlag : int
            Number of lags for the instruments

        Returns
        -------
        (ninstr, nobs - instrlag) array
            Derivatives of the coefficient

        """
        return np.vstack([np.ones_like(data[0]),
                          lagmat(data.T, maxlag=instrlag).T])[:, instrlag:]

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
        # (nobs - instrlag, 3) array
        error = (self.realized_depvar(data).T[instrlag:]
            - self.realized_const(theta))
        # (nobs - instrlag, ninstr)
        instr = self.instruments(data, instrlag=instrlag).T
        # (nobs - instrlag, 3 * ninstr = nmoms)
        moms = columnwise_prod(error, instr)
        # (nintercepts, nparams)
        dmoms = -self.drealized_const(theta)
        dmoments = []
        for minstr in instr.mean(0):
            dmoments.append(dmoms * minstr)
        dmoments = np.vstack(dmoments)

        return moms, dmoments

    def momcond(self, theta, data=None, instrlag=1):
        """Moment function.

        Parameters
        ----------
        theta : array
            Model parameters
        data : array
            Whatever data is necessary to compute moment function
        instrlag : int
            Number of lags for the instruments

        Returns
        -------
        moments : (nobs, nmoms) array
            Moment restrictions
        dmoments : (nmoms, nparams) array
            Average derivative of the moment restrictions

        """
        datalag = 1
        lagdata = lagmat(data, maxlag=datalag)[datalag:]
        nobs = lagdata.shape[0]
        datamat = np.hstack([np.ones((nobs, 1)), lagdata])

        # Coefficients in the first moment (mean)
        linearcoef = [self.betamat(theta), self.gammamat(theta)]
        # Coefficients in the second moment (variance)
        dlinearcoef = [self.dbetamat(theta), self.dgammamat(theta)]

        modelerror = []
        for i in range(len(linearcoef)):
            # Difference between data and model prediction
            error = data[datalag:]**(i+1) - datamat.dot(linearcoef[i])
            modelerror.append(error)
        modelerror = np.vstack(modelerror)

        instruments = np.hstack([np.ones((nobs, 1)),
                                 lagmat(data[:-datalag], maxlag=instrlag)]).T

        mom, dmom = [], []
        for instr in instruments:
            mom.append(modelerror * instr)
            meandata = (datamat.T * instr).mean(1)
            dtheta = []
            for coef in dlinearcoef:
                dtheta.append(meandata.dot(coef))
            dtheta = -np.vstack(dtheta)
            dmom.append(dtheta)

        mom = np.vstack(mom).T
        dmom = np.vstack(dmom)

        return mom, dmom
