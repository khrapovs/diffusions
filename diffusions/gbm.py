#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Geometric Brownian Motion

"""
from __future__ import print_function, division

import numpy as np
import numdifftools as nd

from .generic_model import SDE

__all__ = ['GBM']


class GBMparam(object):

    """Parameter storage for GBM model.

    """

    def __init__(self, mean=0, sigma=.2):
        """Initialize class.

        Parameters
        ----------
        sigma : float

        """
        self.mean = mean
        self.sigma = sigma
        self.theta = [mean, sigma]


class GBM(SDE):

    r"""Geometric Brownian Motion.

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

    Attributes
    ----------

    Methods
    -------

    """

    def __init__(self, theta_true=None):
        """Initialize the class.

        """
        super().__init__(theta_true)

    def drift(self, x, theta):
        """Drift function.

        Parameters
        ----------
        state : array
            Current state of the process
        theta : GBMparam instance
            Parameter object

        Returns
        -------
        scalar
            Drift value

        """
        return theta.mean - theta.sigma**2/2

    def diff(self, x, theta):
        """Diffusion (instantaneous volatility) function.

        Parameters
        ----------
        state : array
            Current state of the process
        theta : GBMparam instance
            Parameter object

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
        theta : GBMparam instance
            Parameter object

        Returns
        -------
        list
            List of coefficients

        """
        loc = self.exact_loc(0, theta)
        if not isinstance(loc, float):
            raise ValueError('Location and scale should be scalars!')
        return [loc, 0]

    def gammamat(self, theta):
        """Coefficients in linear representation of the second moment.

        Parameters
        ----------
        theta : GBMparam instance
            Parameter object

        Returns
        -------
        list
            List of coefficients

        """
        loc = self.exact_loc(0, theta)
        scale = self.exact_scale(0, theta)
        if not isinstance(scale, float):
            raise ValueError('Location and scale should be scalars!')
        return [loc**2 + scale**2, 0]

    def momcond(self, theta, data=None):
        """Moment function.

        Parameters
        ----------
        theta : array
            Model parameters
        data : array
            Whatever data is necessary to compute moment function

        Returns
        -------
        moments : array, nobs x nmoms
            Moment restrictions
        dmoments : array, nmoms x nparams
            Average derivative of the moment restrictions

        """
        mean, sigma = theta
        theta = GBMparam(mean=mean, sigma=sigma)

        mat_data = np.vstack([np.ones_like(data[1:]), data[1:]]).T

        errors = np.vstack([data[1:] - mat_data.dot(self.betamat(theta)),
                            data[1:]**2 - mat_data.dot(self.gammamat(theta))])
        instruments = np.vstack([np.ones_like(data[:-1]), data[:-1]])

        dmean = np.array([-self.interval,
                          -2 * self.interval**2 * (mean - sigma**2/2)])
        dsigma = np.array([sigma * self.interval,
                           -2 * sigma * self.interval
                           + 2 * sigma * self.interval**2
                           * (mean - sigma**2/2)])
        # nmoms x nparam
        dtheta = np.vstack([dmean, dsigma]).T

        mom, dmom = [], []
        for instr in instruments:
            mom.append(errors * instr)
            dmom.append(instr.mean() * dtheta)
        mom = np.vstack(mom)
        dmom = np.vstack(dmom)

        return mom.T, dmom


if __name__ == '__main__':
    pass
