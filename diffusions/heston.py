#!/usr/bin/env python
# -*- coding: utf-8 -*-
r"""
Heston model for stochastic volatility
======================================

The model is

.. math::
    dp_{t}&=\left(r-\frac{1}{2}\sigma_{t}^{2}\right)dt+\sigma_{t}dW_{t}^{r},
    d\sigma_{t}^{2}&=\kappa\left(\mu-\sigma_{t}^{2}\right)dt
    +\eta\sigma_{t}dW_{t}^{\sigma},

with :math:`p_{t}=\log S_{t}`,
and :math:`Corr\left[dW_{s}^{r},dW_{s}^{\sigma}\right]=\rho`,
or in other words
:math:`W_{t}^{\sigma}=\rho W_{t}^{r}+\sqrt{1-\rho^{2}}W_{t}^{v}`.
Also let :math:`R\left(Y_{t}\right)=r`.

"""
from __future__ import print_function, division

import numpy as np

from .generic_model import SDE

__all__ = ['Heston', 'HestonParam']


class HestonParam(object):

    """Parameter storage for Heston model.

    Attributes
    ----------
    mean : float
        Mean of the process
    kappa : float
        Mean reversion speed
    sigma : float
        Instantaneous standard deviation

    """

    def __init__(self, mean_r=.01, mean_v=.5, kappa=1.5, sigma=.1, rho=-.5):
        """Initialize class.

        Parameters
        ----------
        riskfree : float
            Instantaneous rate of return
        mean : float
            Mean of the volatility process
        kappa : float
            Mean reversion speed
        sigma : float
            Instantaneous standard deviation of volatility
        rho : float
            Correlation

        """
        self.mean_r = mean_r
        self.mean_v = mean_v
        self.kappa = kappa
        self.sigma = sigma
        # Vector of parameters
        self.theta = [mean_r, mean_v, kappa, sigma, rho]
        # AJD parameters
        mat_k0 = [mean_r, kappa * mean_v]
        mat_k1 = [[0, -.5], [0, -kappa]]
        mat_h0 = np.zeros((2, 2))
        mat_h1 = [np.zeros((2, 2)), [[1, sigma*rho], [sigma*rho, sigma**2]]]
        self.mat_k0 = np.atleast_1d(mat_k0)
        self.mat_k1 = np.atleast_2d(mat_k1)
        self.mat_h0 = np.atleast_2d(mat_h0)
        self.mat_h1 = np.atleast_3d(mat_h1)

    def is_valid(self):
        """Check Feller condition."""
        return 2 * self.kappa * self.mean_v - self.sigma**2 > 0


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
        return theta.kappa * (theta.mean - state)

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


if __name__ == '__main__':
    pass
