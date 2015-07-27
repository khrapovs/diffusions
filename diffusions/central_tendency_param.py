#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
CT parameter class
~~~~~~~~~~~~~~~~~~

"""
from __future__ import print_function, division

import warnings

import numpy as np

__all__ = ['CentTendParam']


class CentTendParam(object):

    """Parameter storage for CT model.

    Attributes
    ----------
    riskfree : float
        Risk-free rate of return
    mean_v : float
        Mean of the volatility process
    kappa_s : float
        Mean reversion speed of volatility
    kappa_y : float
        Mean reversion speed of central tendency
    eta_s : float
        Instantaneous standard deviation of volatility
    eta_y : float
        Instantaneous standard deviation of central tendency
    lmbd : float
        Equity risk premium
    rho : float
        Correlation

    """

    def __init__(self, riskfree=.0, lmbd=.1, lmbd_s=.0, lmbd_y=.0,
                 mean_v=.5, kappa_s=1.5, kappa_y=.5,
                 eta_s=.1, eta_y=.01, rho=-.5, measure='P'):
        """Initialize class.

        Parameters
        ----------
        riskfree : float
            Risk-free rate of return
        mean_v : float
            Mean of the volatility process
        kappa_s : float
            Mean reversion speed of volatility
        kappa_y : float
            Mean reversion speed of central tendency
        eta_s : float
            Instantaneous standard deviation of volatility
        eta_y : float
            Instantaneous standard deviation of central tendency
        lmbd : float
            Equity risk price
        lmbd_s : float
            Volatility risk price
        lmbd_y : float
            Central tendency risk price
        rho : float
            Correlation
        measure : str
            Either physical measure (P), or risk-neutral (Q)

        """
        self.riskfree = riskfree
        self.kappa_s = kappa_s
        self.kappa_y = kappa_y
        self.mean_v = mean_v
        self.lmbd = lmbd
        self.lmbd_s = lmbd_s
        self.lmbd_y = lmbd_y
        self.eta_y = eta_y
        self.eta_s = eta_s
        self.rho = rho
        self.scale = 1
        if measure == 'Q':
            self.convert_to_q()
        self.update_ajd()
        if not self.is_valid():
            warnings.warn('Feller condition is violated!')

    def convert_to_q(self):
        """Convert parameters to risk-neutral version.

        """
        kappa_sp = self.kappa_s
        kappa_yp = self.kappa_y
        self.kappa_s = self.kappa_s - self.lmbd_s * self.eta_s
        self.kappa_y = self.kappa_y - self.lmbd_y * self.eta_y
        self.scale = kappa_sp / self.kappa_s
        self.mean_v *= (kappa_yp / self.kappa_y * self.scale)
        self.lmbd = 0
        self.eta_y *= (self.scale**.5)

    def update_ajd(self):
        """Update AJD representation.

        """
        # AJD parameters
        self.mat_k0 = [self.riskfree, 0., self.kappa_y * self.mean_v]
        self.mat_k1 = [[0, self.lmbd - .5, 0],
                       [0, -self.kappa_s, self.kappa_s],
                       [0, 0, -self.kappa_y]]
        self.mat_h0 = np.zeros((3, 3))
        self.mat_h1 = np.zeros((3, 3, 3))
        self.mat_h1[1, 0] = [1, self.eta_s*self.rho, 0]
        self.mat_h1[1, 1] = [self.eta_s*self.rho, self.eta_s**2, 0]
        self.mat_h1[2, 2, 2] = self.eta_y**2

    def is_valid(self):
        """Check Feller condition.

        Returns
        -------
        bool
            True for valid parameters, False for invalid

        """
        return 2 * self.kappa_y * self.mean_v - self.eta_y**2 > 0

    def update(self, theta, subset='all', measure='P'):
        """Update attributes from parameter vector.

        Parameters
        ----------
        theta : (nparams, ) array
            Parameter vector
        subset : str
            Which parameters to update. Belongs to ['all', 'vol']

        """
        if subset == 'all':
            [self.mean_v, self.kappa_s, self.kappa_y,
                 self.eta_s, self.eta_y, self.lmbd, self.rho] = theta
        elif subset == 'vol':
            [self.mean_v, self.kappa_s, self.kappa_y,
                 self.eta_s, self.eta_y] = theta
        else:
            raise ValueError(subset + ' keyword variable is not supported!')
        if measure == 'Q':
            self.convert_to_q()
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
        theta = np.array([self.mean_v, self.kappa_s, self.kappa_y,
                          self.eta_s, self.eta_y, self.lmbd, self.rho])
        if subset == 'all':
            return theta
        elif subset == 'vol':
            return theta[:5]
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
        lb = [1e-5, 1e-5, 1e-5, 1e-5, 1e-5, None, -1]
        ub = [None, None, None, None, None, None, 1]
        bounds = list(zip(lb, ub))
        if subset == 'all':
            return bounds
        elif subset == 'vol':
            return bounds[:5]
        else:
            raise ValueError(subset + ' keyword variable is not supported!')


if __name__ == '__main__':
    pass
