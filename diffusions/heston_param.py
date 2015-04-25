#!/usr/bin/env python
# -*- coding: utf-8 -*-
r"""
Parameter class for Heston model
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

"""
from __future__ import print_function, division

import warnings

import numpy as np

__all__ = ['HestonParam']


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

    def __init__(self, riskfree=.0, mean_v=.5, kappa=1.5, eta=.1,
                 lmbd = .1, rho=-.5):
        """Initialize class.

        Parameters
        ----------
        riskfree : float
            Risk-free rate of return
        mean_v : float
            Mean of the volatility process
        kappa : float
            Mean reversion speed
        eta : float
            Instantaneous standard deviation of volatility
        lmbd : float
            Equity risk premium
        rho : float
            Correlation

        """
        self.riskfree = riskfree
        self.mean_v = mean_v
        self.kappa = kappa
        self.eta = eta
        self.lmbd = lmbd
        self.rho = rho
        self.update_ajd()
        if not self.is_valid():
            warnings.warn('Feller condition is violated!')

    def update_ajd(self):
        """Update AJD representation.

        """
        # AJD parameters
        self.mat_k0 = [self.riskfree, self.kappa * self.mean_v]
        self.mat_k1 = [[0, self.lmbd - .5], [0, -self.kappa]]
        self.mat_h0 = np.zeros((2, 2))
        self.mat_h1 = np.zeros((2, 2, 2))
        self.mat_h1[1] = [[1, self.eta*self.rho],
                    [self.eta*self.rho, self.eta**2]]

    def is_valid(self):
        """Check Feller condition.

        Returns
        -------
        bool
            True for valid parameters, False for invalid

        """
        return 2 * self.kappa * self.mean_v - self.eta**2 > 0

    def update(self, theta, subset='all'):
        """Update attributes from parameter vector.

        Parameters
        ----------
        theta : (nparams, ) array
            Parameter vector
        subset : str
            Which parameters to update. Belongs to ['all', 'vol']

        """
        if subset == 'all':
            [self.mean_v, self.kappa, self.eta, self.lmbd, self.rho] = theta
        elif subset == 'vol':
            [self.mean_v, self.kappa, self.eta] = theta
        else:
            raise ValueError(subset + ' keyword variable is not supported!')
        self.update_ajd()

    def get_theta(self, subset='all'):
        """Return vector of model parameters.

        Parameters
        ----------
        subset : str
            Which parameters to update. Belongs to ['all', 'vol']

        Returns
        -------
        (nparams, ) array
            Parameter vector

        """
        theta = np.array([self.mean_v, self.kappa, self.eta,
                          self.lmbd, self.rho])
        if subset == 'all':
            return theta
        elif subset == 'vol':
            return theta[:3]
        else:
            raise ValueError(subset + ' keyword variable is not supported!')

    def get_bounds(self, subset='all'):
        """Bounds on parameters.

        Parameters
        ----------
        subset : str
            Which parameters to update. Belongs to ['all', 'vol']

        Returns
        -------
        sequence of (min, max) tuples

        """
        lb = [1e-5, 1e-5, 1e-5, None, -1]
        ub = [None, None, None, None, 1]
        if subset == 'all':
            return list(zip(lb, ub))
        elif subset == 'vol':
            return list(zip(lb, ub))[:3]
        else:
            raise ValueError(subset + ' keyword variable is not supported!')
