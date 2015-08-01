#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
GBM parameter class
~~~~~~~~~~~~~~~~~~~

"""
from __future__ import print_function, division

import numpy as np

from .param_generic import GenericParam

__all__ = ['GBMparam']


class GBMparam(GenericParam):

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

    def is_valid(self):
        """Check validity of parameters.

        Returns
        -------
        bool
            True for valid parameters, False for invalid

        """
        return self.sigma > 0

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
        """Initialize parameters from parameter vector.

        Parameters
        ----------
        theta : (nparams, ) array
            Parameter vector

        """
        param = cls(mean=theta[0], sigma=theta[1])
        param.update_ajd()
        return param

    def update(self, theta):
        """Update attributes from parameter vector.

        Parameters
        ----------
        theta : (nparams, ) array
            Parameter vector

        """
        self.mean, self.sigma = theta
        self.update_ajd()

    def get_model_name(self):
        """Return model name.

        Returns
        -------
        str
            Parameter vector

        """
        return 'GBM'

    def get_names(self):
        """Return parameter names.

        Returns
        -------
        (2, ) list of str
            Parameter names

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
