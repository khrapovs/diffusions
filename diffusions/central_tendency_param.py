#!/usr/bin/env python
# -*- coding: utf-8 -*-
r"""
Parameter class for Central Tendency (CT) model
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

"""
from __future__ import print_function, division

import warnings

import numpy as np

__all__ = ['CentTendParam']


class CentTendParam(object):

    """Parameter storage for CT model.

    Attributes
    ----------
    mean_r : float
        Instantaneous rate of return
    mean_v : float
        Mean of the volatility process
    kappa_s : float
        Mean reversion speed for volatility
    kappa_v : float
        Mean reversion speed for central tendency
    eta_s : float
        Instantaneous standard deviation of volatility
    eta_v : float
        Instantaneous standard deviation of central tendency
    rho : float
        Correlation

    """

    def __init__(self, riskfree=.0, lmbd = .1,
                 mean_v=.5, kappa_s=1.5, kappa_v=.5,
                 eta_s=.1, eta_v=.01, rho=-.5):
        """Initialize class.

        Parameters
        ----------
        riskfree : float
            Risk-free rate of return
        lmbd : float
            Equity risk premium
        mean_v : float
            Mean of the volatility process
        kappa : float
            Mean reversion speed
        eta : float
            Instantaneous standard deviation of volatility
        rho : float
            Correlation

        """
        self.riskfree = riskfree
        self.lmbd = lmbd
        self.mean_v = mean_v
        self.kappa_s = kappa_s
        self.kappa_v = kappa_v
        self.eta_s = eta_s
        self.eta_v = eta_v
        self.rho = rho
        self.update_ajd()
        if not self.is_valid():
            warnings.warn('Feller condition is violated!')

    def update_ajd(self):
        """Update AJD representation.

        """
        # AJD parameters
        self.mat_k0 = [self.riskfree, 0., self.kappa_v * self.mean_v]
        self.mat_k1 = [[0, self.lmbd - .5, 0],
                       [0, -self.kappa_s, self.kappa_s],
                       [0, 0, -self.kappa_v]]
        self.mat_h0 = np.zeros((3, 3))
        self.mat_h1 = np.zeros((3, 3, 3))
        self.mat_h1[1, 0] = [1, self.eta_s*self.rho, 0]
        self.mat_h1[1, 1] = [self.eta_s*self.rho, self.eta_s**2, 0]
        self.mat_h1[2, 2, 2] = self.eta_v**2

    def is_valid(self):
        """Check Feller condition.

        Returns
        -------
        bool
            True for valid parameters, False for invalid

        """
        return 2 * self.kappa_v * self.mean_v - self.eta_v**2 > 0

    def update(self, theta):
        """Update attributes from parameter vector.

        Parameters
        ----------
        theta : (nparams, ) array
            Parameter vector

        """
        [self.lmbd, self.mean_v, self.kappa_s, self.kappa_v,
         self.eta_s, self.eta_v, self.rho] = theta
        self.update_ajd()

    def get_theta(self):
        """Return vector of model parameters.

        Returns
        -------
        (nparams, ) array
            Parameter vector

        """
        return np.array([self.lmbd, self.mean_v, self.kappa_s, self.kappa_v,
                         self.eta_s, self.eta_v, self.rho])

    def get_bounds(self):
        """Bounds on parameters.

        Returns
        -------
        sequence of (min, max) tuples

        """
        lb = [None, 1e-5, 1e-5, 1e-5, 1e-5, 1e-5, -1]
        ub = [None, None, None, None, None, None, 1]
        return list(zip(lb, ub))


if __name__ == '__main__':
    pass
