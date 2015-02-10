#!/usr/bin/env python
# -*- coding: utf-8 -*-
r"""
Heston model for stochastic volatility
======================================

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

    def __init__(self, mean_r=.01, mean_v=.5, kappa=1.5, eta=.1, rho=-.5):
        """Initialize class.

        Parameters
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
        self.mean_r = mean_r
        self.mean_v = mean_v
        self.kappa = kappa
        self.eta = eta
        # Vector of parameters
        self.theta = [mean_r, mean_v, kappa, eta, rho]
        # AJD parameters
        self.mat_k0 = [mean_r, kappa * mean_v]
        self.mat_k1 = [[0, -.5], [0, -kappa]]
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
        return theta.eta


if __name__ == '__main__':
    pass
