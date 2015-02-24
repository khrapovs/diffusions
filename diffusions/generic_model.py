#!/usr/bin/env python
# -*- coding: utf-8 -*-
r"""
Introduction
============

Affine Diffusion
~~~~~~~~~~~~~~~~

A jump-diffusion process is a Markov process solving the stochastic
differential equationd

.. math::
    Y_{t}=\mu\left(Y_{t},\theta_{0}\right)dt
        +\sigma\left(Y_{t},\theta_{0}\right)dW_{t}.

Discounting is an integral part of pricing so we first introduce a discount-
rate function :math:`R:D\to\mathbb{R}` that is an affine function of the
state

.. math::
    R\left(Y\right)=\rho_{0}+\rho_{1}\cdot Y,

for :math:`\rho=\left(\rho_{0},\rho_{1}\right)\in\mathbb{R}
\times\mathbb{R}^{N}`.
The affine dependence of the drift and diffusion coefficients of :math:`Y` are
determined by coefficients :math:`\left(K,H\right)` defined by:

:math:`\mu\left(Y\right)=K_{0}+K_{1}Y`,
for :math:`K=\left(K_{0},K_{1}\right)
\in\mathbb{R}^{N}\times\mathbb{R}^{N\times N}`,

and

:math:`\left[\sigma\left(Y\right)\sigma\left(Y\right)^{\prime}\right]_{ij}
=\left[H_{0}\right]_{ij}+\left[H_{1}\right]_{ij}\cdot Y`,
for :math:`H=\left(H_{0},H_{1}\right)\in\mathbb{R}^{N\times N}
\times\mathbb{R}^{N\times N\times N}`.

Here

.. math::
    \left[H_{1}\right]_{ij}\cdot Y=\sum_{k=1}^{N}\left[H_{1}\right]_{ijk}Y_{k}.


A characteristic :math:`\chi=\left(K,H,\rho\right)`
captures both the distribution
of :math:`Y` as well as the effects of any discounting.


Discretization and moments
~~~~~~~~~~~~~~~~~~~~~~~~~~

Given the generic continuous-time diffusion model

.. math::
    dY_{t}=\mu\left(Y_{t},\theta_{0}\right)dt
        +\sigma\left(Y_{t},\theta_{0}\right)dW_{t}

we can discretize it as

.. math::
    Y_{t+h}\approx Y_{t}+\mu\left(Y_{t},\theta_{0}\right)h
        +\sigma\left(Y_{t},\theta_{0}\right)\sqrt{h}\varepsilon_{t}.

To be more precise, we can integrate the diffusion on the interval
:math:`\left[t,t+h\right]`:

.. math::

    y_{t,t+h}=Y_{t+h}-Y_{t}=\int_{t}^{t+h}dY_{s}
        =\int_{t}^{t+h}\mu\left(Y_{s},\theta_{0}\right)ds
        +\int_{t}^{t+h}\sigma\left(Y_{s},\theta_{0}\right)dW_{s}.

Conditional mean and variance are thus

.. math::

    E_{t}\left[y_{t,t+h}\right] &=
    E_{t}\left[\int_{t}^{t+h}\mu\left(Y_{s},\theta_{0}\right)ds\right],\\
    V_{t}\left[y_{t,t+h}\right] &=
    E_{t}\left[\int_{t}^{t+h}\sigma^{2}
    \left(Y_{s},\theta_{0}\right)ds\right].

Hence, the moment function is

.. math::

    g\left(y_{t,t+h};\theta\right)=\left[\begin{array}{c}
    y_{t,t+h}-\int_{t}^{t+h}\mu\left(Y_{s},\theta_{0}\right)ds\\
    y_{t,t+h}^{2}-\int_{t}^{t+h}\sigma^{2}\left(Y_{s},\theta_{0}\right)ds
    -\left(\int_{t}^{t+h}\mu\left(Y_{s},\theta_{0}\right)ds\right)^{2}
    \end{array}\right],

with

.. math::

    E_{t}\left[g\left(y_{t,t+h};\theta\right)\right]=0.

"""
from __future__ import print_function, division

import numpy as np

from ..mygmm import GMM
from .helper_functions import nice_errors, ajd_drift, ajd_diff

__all__ = ['SDE']


class SDE(object):

    """Generic Model.

    Attributes
    ----------
    theta_true : parameter instance
        True parameters used for simulation of the data

    Methods
    -------

    """

    def __init__(self, theta_true=None):
        """Initialize the class.

        Parameters
        ----------
        theta_true : parameter instance
            True parameters used for simulation of the data

        """
        self.interval = None
        self.nobs = None
        self.theta_true = theta_true

    def euler_loc(self, state, theta):
        """Euler location.

        Parameters
        ----------
        state : (nvars, nsim) array_like
            Current value of the process
        theta : parameter instance
            Model parameter

        Returns
        -------
        (nvars, nsim) array_like

        """
        return ajd_drift(state, theta) * self.interval

    def euler_scale(self, state, theta):
        """Euler scale.

        Parameters
        ----------
        state : (nvars, nsim) array_like
            Current value of the process
        theta : parameter instance
            Model parameter

        Returns
        -------
        (nvars, nvars, nsim) array_like

        """
        return ajd_diff(state, theta) * self.interval**.5

    def exact_loc(self, state, theta):
        """Eaxct location.

        Parameters
        ----------
        state : (nvars, nsim) array_like
            Current value of the process
        theta : parameter instance
            Model parameter

        Returns
        -------
        (nvars, nsim) array_like

        """
        return self.euler_loc(state, theta)

    def exact_scale(self, state, theta):
        """Exact scale.

        Parameters
        ----------
        state : (nvars, nsim) array_like
            Current value of the process
        theta : parameter instance
            Model parameter

        Returns
        -------
        (nvars, nvars, nsim) array_like

        """
        return self.euler_scale(state, theta)

    def update(self, state, error):
        """Euler update function.

        Parameters
        ----------
        state : (nsim, nvars) array_like
            Current value of the process
        error : (nsim, nvars) array_like
            Random shocks

        Returns
        -------
        (nsim, nvars) array
            Update of the process value. Same shape as the input.

        """
        # (nsim, nvars) array_like
        loc = self.euler_loc(state, self.theta_true)
        # (nsim, nvars, nvars) array_like
        scale = self.euler_scale(state, self.theta_true)

        new_state = loc / self.ndiscr \
            + (np.transpose(scale, axes=[1, 2, 0]) * error.T).sum(1).T \
            / self.ndiscr**.5

        # Equivalent operation through the loop:
#        new_state = loc / self.ndiscr
#        for i in range(error.shape[0]):
#            new_state[i] += (scale[i] * error[i]).sum(1) / self.ndiscr**.5

        return new_state

    def simulate(self, start, interval, ndiscr, nobs, nsim, diff=None):
        """Simulate observations from the model.

        Parameters
        ----------
        start : array_like
            Starting value for simulation
        interval : float
            Interval length
        ndiscr : int
            Number of Euler discretization points inside unit interval
        nobs : int
            Number of points to simulate in one series
        nsim : int
            Number of time series to simulate
        diff : int
            Dimensions which should be differentiated,
            i.e. return = price[1:] - price[:-1]

        Returns
        -------
        paths : (nobs, 2*nsim, nvars) array
            Simulated data

        """
        self.interval = interval
        self.nobs = nobs
        self.ndiscr = ndiscr
        nvars = np.size(start)
        npoints = nobs * ndiscr

        self.errors = np.random.normal(size=(npoints, nsim, nvars))

        # Standardize the errors
        self.errors = nice_errors(self.errors, 1)
        nsim = self.errors.shape[1]

        paths = start * np.ones((npoints + 1, nsim, nvars))

        for i in range(npoints):
            # (nsim, nvars)
            paths[i+1] = paths[i] + self.update(paths[i], self.errors[i])

        # (nobs+1, nsim, nvars)
        paths = paths[::ndiscr]
        if diff is not None:
            paths[1:, :, diff] = paths[1:, :, diff] - paths[:-1, :, diff]
        return paths[1:]

    def sim_realized(self, start, interval=1/80, ndiscr=10,
                     nperiods=500, nsim=1, diff=None):
        """Simulate observations from the model.

        Parameters
        ----------
        start : array_like
            Starting value for simulation
        interval : float
            Interval length
        ndiscr : int
            Number of Euler discretization points inside unit interval
        nperiods : int
            Number of points to simulate in one series
        nsim : int
            Number of time series to simulate
        diff : int
            Dimensions which should be differentiated,
            i.e. return = price[1:] - price[:-1]

        Returns
        -------
        returns : (nperiods, ) array
            Simulated returns
        realized_vol : (nperiods, ) array
            Simulated realized volatility

        """
        nobs = int(nperiods / interval)
        paths = self.simulate(start, interval, ndiscr, nobs, nsim, diff)
        returns = paths[:, 0, 0].reshape((nperiods, int(nobs / nperiods)))
        rvol = returns.var(1)
        returns = returns.sum(1)
        return returns, rvol

    def gmmest(self, theta_start, **kwargs):
        """Estimate model parameters using GMM.

        Parameters
        ----------
        theta_start : parameter instance
            Initial parameter values for estimation.
            Object with mandatory attribute 'theta'.

        """
        estimator = GMM(self.momcond)
        return estimator.gmmest(theta_start.theta, **kwargs)


if __name__ == '__main__':
    pass
