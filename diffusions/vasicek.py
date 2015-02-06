#!/usr/bin/env python
# -*- coding: utf-8 -*-
r"""
Vasicek model for interest rates
================================

Suppose that :math:`r_{t}` evolves according to

.. math::
    dr_{t}=\kappa\left(\mu-r_{t}\right)dt+\sigma dW_{t}.

"""
from __future__ import print_function, division

import numpy as np
import numdifftools as nd

from statsmodels.tsa.tsatools import lagmat

from .generic_model import SDE

__all__ = ['Vasicek', 'VasicekParam']


class VasicekParam(object):

    """Parameter storage for Vasicek model.

    Attributes
    ----------

    """

    def __init__(self, mean=.5, kappa=1.5, sigma=.1):
        """Initialize class.

        Parameters
        ----------
        sigma : float

        """
        self.mean = mean
        self.kappa = kappa
        self.sigma = sigma
        # Vector of parameters
        self.theta = [mean, kappa, sigma]
        # AJD parameters
        mat_k0 = kappa * mean
        mat_k1 = -kappa
        mat_h0 = sigma**2
        mat_h1 = 0
        self.mat_k0 = np.atleast_1d(mat_k0)
        self.mat_k1 = np.atleast_2d(mat_k1)
        self.mat_h0 = np.atleast_2d(mat_h0)
        self.mat_h1 = np.atleast_3d(mat_h1)


class Vasicek(SDE):

    """Vasicek model.

    Attributes
    ----------

    Methods
    -------

    """

    def __init__(self, theta_true=None):
        """Initialize the class.

        """
        super(Vasicek, self).__init__(theta_true)

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
