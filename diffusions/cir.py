#!/usr/bin/env python
# -*- coding: utf-8 -*-
r"""
Cox-Ingersoll-Ross (CIR) model
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Suppose that :math:`r_{t}` evolves according to

.. math::
    dr_{t}=\kappa\left(\mu-r_{t}\right)dt+\eta\sqrt{r_{t}}dW_{t}.

Feller condition for positivity of the process is
:math:`\kappa\mu>\frac{1}{2}\eta^{2}`.

"""
from __future__ import print_function, division

from .generic_model import SDE

__all__ = ['CIR']


class CIR(SDE):

    """Cox-Ingersoll-Ross (CIR) model.

    Attributes
    ----------

    Methods
    -------

    """

    def __init__(self, theta_true=None):
        """Initialize the class.

        """
        super(CIR, self).__init__(theta_true)

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


if __name__ == '__main__':
    pass
