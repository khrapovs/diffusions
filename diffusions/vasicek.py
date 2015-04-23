#!/usr/bin/env python
# -*- coding: utf-8 -*-
r"""
Vasicek model for interest rates
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Suppose that :math:`r_{t}` evolves according to

.. math::
    dr_{t}=\kappa\left(\mu-r_{t}\right)dt+\eta dW_{t}.

"""
from __future__ import print_function, division

from .generic_model import SDE

__all__ = ['Vasicek']


class Vasicek(SDE):

    """Vasicek model.

    Attributes
    ----------

    Methods
    -------

    """

    def __init__(self, theta_true=None):
        """Initialize the class.

        """
        super(Vasicek, self).__init__(theta_true)

    def drift(self, state, theta):
        """Drift function.

        Parameters
        ----------
        state : (nvars, nsim) array_like
            Current value of the process
        theta : parameter instance
            Model parameter

        Returns
        -------
        scalar
            Drift value

        """
        return theta.kappa * (theta.mean - state)

    def diff(self, state, theta):
        """Diffusion (instantaneous volatility) function.

        Parameters
        ----------
        state : (nvars, nsim) array_like
            Current value of the process
        theta : parameter instance
            Model parameter

        Returns
        -------
        scalar
            Diffusion value

        """
        return theta.eta


if __name__ == '__main__':
    pass
