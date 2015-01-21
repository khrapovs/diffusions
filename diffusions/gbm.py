#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Geometric Brownian Motion

"""
from __future__ import print_function, division

import numpy as np

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
        return theta.mean - theta.sigma**2/2

    def diff(self, x, theta):
        return theta.sigma

    def momcond(self, theta, data=None):
        """Moment function.

        Parameters
        ----------
        theta : array
            Model parameters

        Returns
        -------
        moments : array, nobs x nmoms
            Moment restrictions
        dmoments : array, nmoms x nparams
            Average derivative of the moment restrictions

        """
        mean, sigma = theta
        theta = GBMparam(mean=mean, sigma=sigma)

        loc = self.exact_loc(data[:-1], theta)
        scale = self.exact_scale(data[:-1], theta)

        errors = np.vstack([data[1:] - loc,
                            data[1:]**2 - loc**2 - scale**2])
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
