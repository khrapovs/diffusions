#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
CIR model class
~~~~~~~~~~~~~~~

"""
from __future__ import print_function, division

from .model_generic import SDE

__all__ = ['CIR']


class CIR(SDE):

    """Cox-Ingersoll-Ross (CIR) model.

    """

    def __init__(self, param=None):
        """Initialize the class.

        Parameters
        ----------
        param : parameter instance
            True parameters used for simulation of the data

        """
        super(CIR, self).__init__(param)

    @staticmethod
    def drift(state, theta):
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

    @staticmethod
    def diff(state, theta):
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
        return theta.eta * state**.5
