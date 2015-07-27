#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Vasicek parameter class
~~~~~~~~~~~~~~~~~~~~~~~

"""
from __future__ import print_function, division

import numpy as np

__all__ = ['VasicekParam']


class VasicekParam(object):

    """Parameter storage for Vasicek model.

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
        self.update_ajd()

    def update_ajd(self):
        """Update AJD representation.

        """
        # AJD parameters
        self.mat_k0 = self.kappa * self.mean
        self.mat_k1 = -self.kappa
        self.mat_h0 = self.eta**2
        self.mat_h1 = 0

    def get_theta(self):
        """Return vector of parameters.

        Returns
        -------
        (3, ) array
            Parameter vector

        """
        return np.array([self.mean, self.kappa, self.eta])

    def update(self, theta):
        """Update attributes from parameter vector.

        Parameters
        ----------
        theta : (nparams, ) array
            Parameter vector

        """
        [self.mean, self.kappa, self.eta] = theta
        self.update_ajd()
