#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
CT parameter class
~~~~~~~~~~~~~~~~~~

"""
from __future__ import print_function, division

import warnings

import numpy as np

from .param_generic import GenericParam

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
    measure : str
        Under which measure (P or Q)

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
        self.measure = 'P'
        if measure == 'Q':
            self.convert_to_q()
        self.update_ajd()

    @staticmethod
    def get_model_name():
        """Return model name.

        Returns
        -------
        str
            Parameter vector

        """
        return 'Central Tendency'

    @staticmethod
    def get_names(subset='all', measure='PQ'):
        """Return parameter names.

        Parameters
        ----------
        subset : str

            Which parameters to return. Belongs to
                - 'all' : all parameters, including those related to returns
                - 'vol' : only those related to volatility
        measure : str

            Under which measure:
                - 'P' : physical measure
                - 'Q' : risk-neutral
                - 'PQ' : both

        Returns
        -------
        list of str
            Parameter names

        """
        names = ['mean_v', 'kappa_s', 'kappa_y', 'eta_s', 'eta_y',
                 'rho', 'lmbd', 'lmbd_s', 'lmbd_y']

        if subset == 'all' and measure == 'PQ':
            return names
        elif subset == 'all' and measure in ('P', 'Q'):
            return names[:-2]
        elif subset == 'vol' and measure == 'PQ':
            return names[:5] + names[-2:]
        elif subset == 'vol' and measure in ('P', 'Q'):
            return names[:5]
        else:
            raise NotImplementedError('Keyword variable is not supported!')

    def convert_to_q(self):
        """Convert parameters to risk-neutral version.

        """
        if self.measure == 'Q':
            warnings.warn('Parameters are already converted to Q!')
        else:
            kappa_sp = self.kappa_s
            kappa_yp = self.kappa_y
            self.kappa_s = self.kappa_s - self.lmbd_s * self.eta_s
            self.kappa_y = self.kappa_y - self.lmbd_y * self.eta_y
            self.scale = kappa_sp / self.kappa_s
            self.mean_v *= (kappa_yp / self.kappa_y * self.scale)
            self.lmbd = 0
            self.eta_y *= (self.scale**.5)
            self.measure = 'Q'
            self.update_ajd()

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

    def feller(self):
        """Check Feller condition.

        Returns
        -------
        bool
            True for valid parameters, False for invalid

        """
        return 2 * self.kappa_y * self.mean_v - self.eta_y**2 > 0

    def is_valid(self):
        """Check validity of parameters.

        Returns
        -------
        bool
            True for valid parameters, False for invalid

        """
        posit1 = (self.mean_v > 0) & (self.kappa_y > 0) & (self.eta_y > 0)
        posit2 = (self.kappa_s > 0) & (self.eta_s > 0)
        return posit1 & posit2 & self.feller()

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
        return cls(riskfree=theta[0], mean_v=theta[1], kappa_s=theta[2],
                   kappa_y=theta[3], eta_s=theta[4], eta_y=theta[5],
                   rho=theta[6], lmbd=theta[7], lmbd_s=theta[8],
                   lmbd_y=theta[9], measure=measure)

    def update(self, theta, subset='all', measure='PQ'):
        """Update attributes from parameter vector.

        Parameters
        ----------
        theta : (nparams, ) array
            Parameter vector
        subset : str
            Which parameters to update

            Belongs to
                - 'all' : all parameters, including those related to returns
                - 'vol' : only those related to volatility

        measure : str

            Under which measure:
                - 'P' : physical measure
                - 'Q' : risk-neutral
                - 'PQ' : both

        """
        [self.mean_v, self.kappa_s, self.kappa_y,
             self.eta_s, self.eta_y] = theta[:5]

        if subset == 'all' and measure == 'PQ':
            [self.rho, self.lmbd, self.lmbd_s, self.lmbd_y] = theta[5:]
        elif subset == 'all' and measure in ('P', 'Q'):
            [self.rho, self.lmbd] = theta[5:7]
        elif subset == 'vol' and measure == 'PQ':
            [self.lmbd_s, self.lmbd_y] = theta[-2:]
        elif subset == 'vol' and measure in ('P', 'Q'):
            pass
        else:
            raise NotImplementedError('Keyword variable is not supported!')

        self.measure = 'P'
        if measure == 'Q':
            self.convert_to_q()
        self.update_ajd()

    def get_theta(self, subset='all', measure='PQ'):
        """Return vector of model parameters.

        Parameters
        ----------
        subset : str
            Which parameters to return

            Belongs to
                - 'all' : all parameters, including those related to returns
                - 'vol' : only those related to volatility

        measure : str

            Under which measure:
                - 'P' : physical measure
                - 'Q' : risk-neutral
                - 'PQ' : both

        Returns
        -------
        (nparams, ) array
            Parameter vector

        """
        theta = np.array([self.mean_v, self.kappa_s, self.kappa_y,
                          self.eta_s, self.eta_y, self.rho,
                          self.lmbd, self.lmbd_s, self.lmbd_y])
        if subset == 'all' and measure == 'PQ':
            return theta
        elif subset == 'all' and measure in ('P', 'Q'):
            return theta[:-2]
        elif subset == 'vol' and measure == 'PQ':
            return np.concatenate((theta[:5], theta[-2:]))
        elif subset == 'vol' and measure in ('P', 'Q'):
            return theta[:5]
        else:
            raise NotImplementedError('Keyword variable is not supported!')

    def get_bounds(self, subset='all', measure='PQ'):
        """Bounds on parameters.

        Parameters
        ----------
        subset : str
            Which parameters to update

            Belongs to
                - 'all' : all parameters, including those related to returns
                - 'vol' : only those related to volatility

        measure : str

            Under which measure:
                - 'P' : physical measure
                - 'Q' : risk-neutral
                - 'PQ' : both

        Returns
        -------
        sequence of (min, max) tuples

        """
        lb = [1e-5, 1e-5, 1e-5, 1e-5, 1e-5, -1, None, None, None]
        ub = [None, None, None, None, None, 1, None, None, None]
        bounds = list(zip(lb, ub))

        if subset == 'all' and measure == 'PQ':
            return bounds
        elif subset == 'all' and measure in ('P', 'Q'):
            return bounds[:-2]
        elif subset == 'vol' and measure == 'PQ':
            return bounds[:5] + bounds[-2:]
        elif subset == 'vol' and measure in ('P', 'Q'):
            return bounds[:5]
        else:
            raise NotImplementedError('Keyword variable is not supported!')
