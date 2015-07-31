#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Vasicek parameter class
~~~~~~~~~~~~~~~~~~~~~~~

"""
from __future__ import print_function, division

import numpy as np

from .generic_param import GenericParam

__all__ = ['VasicekParam']


class VasicekParam(GenericParam):

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

    def is_valid(self):
        """Check validity of parameters.

        Returns
        -------
        bool
            True for valid parameters, False for invalid

        """
        return (self.kappa > 0) & (self.eta > 0)

    def update_ajd(self):
        """Update AJD representation.

        """
        # AJD parameters
        self.mat_k0 = self.kappa * self.mean
        self.mat_k1 = -self.kappa
        self.mat_h0 = self.eta**2
        self.mat_h1 = 0

    @classmethod
    def from_theta(cls, theta):
        """Initialize parameters from parameter vector.

        Parameters
        ----------
        theta : (nparams, ) array
            Parameter vector

        """
        param = cls(mean=theta[0], kappa=theta[1], eta=theta[2])
        param.update_ajd()
        return param

    def update(self, theta):
        """Update attributes from parameter vector.

        Parameters
        ----------
        theta : (nparams, ) array
            Parameter vector

        """
        self.mean, self.kappa, self.eta = theta
        self.update_ajd()

    def get_model_name(self):
        """Return model name.

        Returns
        -------
        str
            Parameter vector

        """
        return 'Vasicek'

    def get_names(self):
        """Return parameter names.

        Returns
        -------
        (3, ) list of str
            Parameter names

        """
        return ['mean', 'kappa', 'eta']

    def get_theta(self):
        """Return vector of parameters.

        Returns
        -------
        (3, ) array
            Parameter vector

        """
        return np.array([self.mean, self.kappa, self.eta])
