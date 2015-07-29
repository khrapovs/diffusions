#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Generic parameter class
~~~~~~~~~~~~~~~~~~~~~~~

"""
from __future__ import print_function, division

__all__ = ['GenericParam']


class GenericParam(object):

    """Generic parameter storage. Must be overriden.

    Attributes
    ----------

    """

    def __init__(self):
        """Initialize class.

        """
        pass

    def update_ajd(self):
        """Update AJD representation.

        """
        raise NotImplementedError('Must be overridden')

    @classmethod
    def from_theta(cls, theta):
        """Initialize parameters from parameter vector.

        Parameters
        ----------
        theta : (nparams, ) array
            Parameter vector

        """
        raise NotImplementedError('Must be overridden')

    def update(self, theta, subset='all', measure='P'):
        """Update attributes from parameter vector.

        Parameters
        ----------
        theta : (nparams, ) array
            Parameter vector
        subset : str
            Which parameters to update. Belongs to ['all', 'vol']
        measure : str
            Either physical measure (P), or risk-neutral (Q)

        """
        raise NotImplementedError('Must be overridden')

    def get_model_name(self):
        """Return model name.

        Returns
        -------
        str
            Parameter vector

        """
        raise NotImplementedError('Must be overridden')

    def get_names(self):
        """Return parameter names.

        Returns
        -------
        list of str
            Parameter names

        """
        raise NotImplementedError('Must be overridden')

    def get_theta(self):
        """Return vector of parameters.

        Returns
        -------
        array
            Parameter vector

        """
        raise NotImplementedError('Must be overridden')

    def __str__(self):
        """String representation.

        """
        show = self.get_model_name() + ' parameters:\n'
        for name, param in zip(self.get_names(), self.get_theta()):
            show += name + ' = ' + str(param) + ', '
        return show[:-2]

    def __repr__(self):
        """String representation.

        """
        return self.__str__()
