#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Heston parameter class
~~~~~~~~~~~~~~~~~~~~~~

"""
from __future__ import print_function, division

import warnings

import numpy as np

from .generic_param import GenericParam

__all__ = ['HestonParam']


class HestonParam(GenericParam):

    """Parameter storage for Heston model.

    Attributes
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
        Equity risk price
    lmbd_v : float
        Volatility risk price
    rho : float
        Correlation
    measure : str
        Under which measure (P or Q)

    Methods
    -------
    is_valid
        Check Feller condition
    convert_to_q
        Convert parameters to risk-neutral version

    """

    def __init__(self, riskfree=.0, mean_v=.5, kappa=1.5, eta=.1,
                 lmbd=.1, lmbd_v=.0, rho=-.5, measure='P'):
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
            Equity risk price
        lmbd_v : float
            Volatility risk price
        rho : float
            Correlation
        measure : str
            Under which measure:
                - 'P' : physical measure
                - 'Q' : risk-neutral

        """
        self.riskfree = riskfree
        self.kappa = kappa
        self.mean_v = mean_v
        self.lmbd = lmbd
        self.lmbd_v = lmbd_v
        self.eta = eta
        self.rho = rho
        self.measure = 'P'
        if measure == 'Q':
            self.convert_to_q()
        self.update_ajd()
        if not self.is_valid():
            warnings.warn('Feller condition is violated!')

    def convert_to_q(self):
        """Convert parameters to risk-neutral version.

        """
        if self.measure == 'Q':
            warnings.warn('Parameters are already converted to Q!')
        else:
            kappa_p = self.kappa
            self.kappa = kappa_p - self.lmbd_v * self.eta
            self.mean_v *= (kappa_p / self.kappa)
            self.lmbd = .0
            self.measure = 'Q'
            self.update_ajd()

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

    def get_model_name(self):
        """Return model name.

        Returns
        -------
        str
            Parameter vector

        """
        return 'Heston'

    def get_names(self, subset='all'):
        """Return parameter names.

        Parameters
        ----------
        subset : str
            Which parameters to return. Belongs to
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

    def is_valid(self):
        """Check Feller condition.

        Returns
        -------
        bool
            True for valid parameters, False for invalid

        """
        return 2 * self.kappa * self.mean_v - self.eta**2 > 0

    @classmethod
    def from_theta(cls, theta, measure='P'):
        """Initialize parameters from parameter vector.

        Parameters
        ----------
        theta : (nparams, ) array
            Parameter vector
        measure : str
            measure : str
            Under which measure:
                - 'P' : physical measure
                - 'Q' : risk-neutral

        """
        return cls(riskfree=theta[0], mean_v=theta[1], kappa=theta[2],
                   eta=theta[3], lmbd=theta[4], lmbd_v=theta[5],
                   rho=theta[6], measure=measure)

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
        measure : str
            Under which measure:
                - 'P' : physical measure
                - 'Q' : risk-neutral

        """
        if subset == 'all':
            [self.mean_v, self.kappa, self.eta, self.lmbd, self.rho] = theta
        elif subset == 'vol':
            [self.mean_v, self.kappa, self.eta] = theta
        else:
            raise ValueError(subset + ' keyword variable is not supported!')
        self.measure = 'P'
        if measure == 'Q':
            self.convert_to_q()
        self.update_ajd()

    def get_theta(self, subset='all'):
        """Return vector of model parameters.

        Parameters
        ----------
        subset : str
            Which parameters to update. Belongs to
                - 'all' : all parameters, including those related to returns
                - 'vol' : only those related to volatility

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
            Which parameters to update. Belongs to
                - 'all' : all parameters, including those related to returns
                - 'vol' : only those related to volatility

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
