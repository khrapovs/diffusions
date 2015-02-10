#!/usr/bin/env python
# -*- coding: utf-8 -*-
r"""
Geometric Brownian Motion (GBM) model
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Suppose that :math:`S_{t}` evolves according to

.. math::
    \frac{dS_{t}}{S_{t}}=\mu dt+\sigma dW_{t}.

In logs:

.. math::
    d\log S_{t}=\left(\mu-\frac{1}{2}\sigma^{2}\right)dt+\sigma dW_{t}.

After integration on the interval :math:`\left[t,t+h\right]`:

.. math::
    r_{t,h}=\log\frac{S_{t+h}}{S_{t}}
        =\left(\mu-\frac{1}{2}\sigma^{2}\right)h
        +\sigma\sqrt{h}\varepsilon_{t+h},

where :math:`\varepsilon_{t}\sim N\left(0,1\right)`.

"""
from __future__ import print_function, division

import numpy as np
import numdifftools as nd

from statsmodels.tsa.tsatools import lagmat

from .generic_model import SDE

__all__ = ['GBM', 'GBMparam']


class GBMparam(object):

    """Parameter storage for GBM model.

    Attributes
    ----------
    mean : float
        Mean of the process
    sigma : float
        Instantaneous standard deviation

    """

    def __init__(self, mean=0, sigma=.2):
        """Initialize class.

        Parameters
        ----------
        mean : float
            Mean of the process
        sigma : float
            Instantaneous standard deviation

        """
        self.mean = mean
        self.sigma = sigma
        # Vector of parameters
        self.theta = [mean, sigma]
        # AJD parameters
        self.mat_k0 = mean - sigma**2/2
        self.mat_k1 = 0
        self.mat_h0 = sigma**2
        self.mat_h1 = 0


class GBM(SDE):

    """Geometric Brownian Motion.

    Attributes
    ----------

    Methods
    -------

    """

    def __init__(self, theta_true=None):
        """Initialize the class.

        """
        super(GBM, self).__init__(theta_true)

    def drift(self, state, theta):
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

    def diff(self, state, theta):
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
        theta = GBMparam(mean=theta[0], sigma=theta[1])
        loc = float(self.exact_loc(0, theta))
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
        theta = GBMparam(mean=theta[0], sigma=theta[1])
        loc = float(self.exact_loc(0, theta))
        scale = float(self.exact_scale(0, theta))
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


if __name__ == '__main__':
    pass
