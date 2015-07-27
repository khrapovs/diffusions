#!/usr/bin/env python
# -*- coding: utf-8 -*-
r"""
Parameter class for Geometric Brownian Motion (GBM)
---------------------------------------------------

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

    @classmethod
    def from_theta(cls, theta):
        """Update attributes from parameter vector.

        Parameters
        ----------
        theta : (nparams, ) array
            Parameter vector

        """
        param = cls(mean=theta[0], sigma=theta[1])
        param.update_ajd()
        return param

    def get_names(self):
        """Return parameter names.

        Returns
        -------
        (2, ) array
            Parameter vector

        """
        return ['mean', 'sigma']

    def get_theta(self):
        """Return vector of parameters.

        Returns
        -------
        (2, ) array
            Parameter vector

        """
        return np.array([self.mean, self.sigma])

    def __str__(self):
        """String representation.

        """
        show = 'GBM parameters:\n'
        for name, param in zip(self.get_names(), self.get_theta()):
            show += name + ' = ' + str(param) + ', '
        return show[:-2]

    def __repr__(self):
        """String representation.

        """
        return self.__str__()
