#!/usr/bin/env python
# -*- coding: utf-8 -*-
r"""
Parameter class for Geometric Brownian Motion (GBM)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

"""
from __future__ import print_function, division

import numpy as np

__all__ = ['GBMparam']


class GBMparam(object):

    """Parameter storage for GBM model.

    Attributes
    ----------
    mean : float
        Mean of the process
    sigma : float
        Instantaneous standard deviation

    """

    def __init__(self, mean=0, sigma=.2):
        """Initialize class.

        Parameters
        ----------
        mean : float
            Mean of the process
        sigma : float
            Instantaneous standard deviation

        """
        self.mean = mean
        self.sigma = sigma
        self.update_ajd()

    def update_ajd(self):
        """Update AJD representation.

        """
        # AJD parameters
        self.mat_k0 = self.mean - self.sigma**2/2
        self.mat_k1 = 0.
        self.mat_h0 = self.sigma**2
        self.mat_h1 = 0.

    def get_theta(self):
        """Return vector of parameters.

        Returns
        -------
        (2, ) array
            Parameter vector

        """
        return np.array([self.mean, self.sigma])

    def update(self, theta):
        """Update attributes from parameter vector.

        Parameters
        ----------
        theta : (nparams, ) array
            Parameter vector

        """
        [self.mean, self.sigma] = theta
        self.update_ajd()
