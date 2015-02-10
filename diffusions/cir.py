#!/usr/bin/env python
# -*- coding: utf-8 -*-
r"""
Cox-Ingersoll-Ross (CIR) model
==============================

Suppose that :math:`r_{t}` evolves according to

.. math::
    dr_{t}=\kappa\left(\mu-r_{t}\right)dt+\eta\sqrt{r_{t}}dW_{t}.

Feller condition for positivity of the process is
:math:`\kappa\mu>\frac{1}{2}\eta^{2}`.

"""
from __future__ import print_function, division

import numpy as np

from .generic_model import SDE

__all__ = ['CIR', 'CIRparam']


class CIRparam(object):

    """Parameter storage for CIR model.

    Attributes
    ----------
    mean : float
        Mean of the process
    kappa : float
        Mean reversion speed
    eta : float
        Instantaneous standard deviation

    """

    def __init__(self, mean=.5, kappa=1.5, eta=.1):
        """Initialize class.

        Parameters
        ----------
        mean : float
            Mean of the process
        kappa : float
            Mean reversion speed
        eta : float
            Instantaneous standard deviation

        """
        self.mean = mean
        self.kappa = kappa
        self.eta = eta
        # Vector of parameters
        self.theta = [mean, kappa, eta]
        # AJD parameters
        self.mat_k0 = kappa * mean
        self.mat_k1 = -kappa
        self.mat_h0 = 0
        self.mat_h1 = eta**2

    def is_valid(self):
        """Check Feller condition.

        Returns
        -------
        bool
            True for valid parameters, False for invalid

        """
        return 2 * self.kappa * self.mean - self.eta**2 > 0


class CIR(SDE):

    """Cox-Ingersoll-Ross (CIR) model.

    Attributes
    ----------

    Methods
    -------

    """

    def __init__(self, theta_true=None):
        """Initialize the class.

        """
        super(CIR, self).__init__(theta_true)

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
        return theta.eta * state**.5


if __name__ == '__main__':
    pass
