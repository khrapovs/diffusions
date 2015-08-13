#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Generic parameter class
~~~~~~~~~~~~~~~~~~~~~~~

"""
from __future__ import print_function, division

import pandas as pd

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

    def is_valid(self):
        """Check whether parameters are valid.

        Returns
        -------
        bool
            True for valid parameters, False for invalid

        """
        return True

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

    def get_bounds(self):
        """Get parameter bounds.

        Returns
        -------
        list of tuples
            Parameter bounds

        """
        return None

    def get_constraints(self):
        """Get parameter constraints.

        Returns
        -------
        dict or sequence of dict
            Equality and inequality constraints. See scipy.optimize.minimize

        """
        return ()

    def __str__(self):
        """String representation.

        """
        show = self.get_model_name() + ' parameters under ' + self.measure
        if self.is_valid():
            show += ' (valid)'
        else:
            show += ' (not valid)'
        show += ':\n'
        table = pd.DataFrame({'theta': self.get_theta()},
                             index=self.get_names())
        tb_str = table.to_string(float_format=lambda x: '%.4f' % x)
        width = len(tb_str) // (table.shape[0] + 1)
        show += width * '-' + '\n'
        show += tb_str
        show += '\n' + width * '-'
        return show

    def __repr__(self):
        """String representation.

        """
        return self.__str__()
