#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
CT parameter class
~~~~~~~~~~~~~~~~~~

"""
from __future__ import print_function, division

import warnings

import numpy as np

from .generic_param import GenericParam

__all__ = ['CentTendParam']


class CentTendParam(GenericParam):

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
            Under which measure:
                - 'P' : physical measure
                - 'Q' : risk-neutral

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

    def get_model_name(self):
        """Return model name.

        Returns
        -------
        str
            Parameter vector

        """
        return 'CT'

    def get_names(self, subset='all'):
        """Return parameter names.

        Parameters
        ----------
        subset : str
            Which parameter names to return. Belongs to
                - 'all' : all parameters, including those related to returns
                - 'vol' : only those related to volatility

        Returns
        -------
        list of str
            Parameter names

        """
        names = ['mean_v', 'kappa', 'eta', 'lmbd', 'rho']
        if subset == 'all':
            return names
        elif subset == 'vol':
            return names[:3]
        else:
            raise ValueError(subset + ' keyword variable is not supported!')

    @classmethod
    def from_theta(cls, theta, measure='P'):
        """Initialize parameters from parameter vector.

        Parameters
        ----------
        theta : (nparams, ) array
            Parameter vector
        measure : str
            Under which measure:
                - 'P' : physical measure
                - 'Q' : risk-neutral

        """
        param = cls(riskfree=theta[0], mean_v=theta[1], kappa_s=theta[2],
                    kappa_v=theta[3], eta_s=theta[4], eta_y=theta[5],
                    lmbd=theta[6], lmbd_s=theta[7], lmbd_y=theta[8],
                    rho=theta[9], measure=measure)

        return param

    def update(self, theta, subset='all', measure='P'):
        """Update attributes from parameter vector.

        Parameters
        ----------
        theta : (nparams, ) array
            Parameter vector
        subset : str
            Which parameters to update. Belongs to
                - 'all' : all parameters, including those related to returns
                - 'vol' : only those related to volatility

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
            Which parameters to return. Belongs to
                - 'all' : all parameters, including those related to returns
                - 'vol' : only those related to volatility

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
            Which parameter bounds to return. Belongs to
                - 'all' : all parameters, including those related to returns
                - 'vol' : only those related to volatility

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
