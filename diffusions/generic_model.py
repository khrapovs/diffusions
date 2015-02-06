#!/usr/bin/env python
# -*- coding: utf-8 -*-
r"""
Generic Model
=============

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

from mygmm import GMM
from .helper_functions import nice_errors, ajd_drift, ajd_diff

__all__ = ['SDE']


class SDE(object):

    """Generic Model.

    Attributes
    ----------
    paths : (nvars, nobs, nsim) array
        Simulated observations

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
        self.paths = None
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
        state : (nvars, nsim) array_like
            Current value of the process
        error : (nvars, nsim) array_like
            Random shocks

        Returns
        -------
        (nvars, nsim) array
            Update of the process value. Same shape as the input.

        """
        # (nvars, nsim) array_like
        loc = self.euler_loc(state, self.theta_true)
        # (nvars, nvars, nsim) array_like
        scale = self.euler_scale(state, self.theta_true)

        return loc / self.ndiscr + (scale * error).sum(1) / self.ndiscr**.5

    def simulate(self, start, interval, ndiscr, nobs, nsim):
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

        Returns
        -------
        paths : (npoints+1, nvars, nsim*2) array
            Simulated data

        """
        self.interval = interval
        self.nobs = nobs
        self.ndiscr = ndiscr
        nvars = np.size(start)
        npoints = nobs * ndiscr

        self.errors = np.random.normal(size=(npoints, nvars, nsim))
        # Standardize the errors
        self.errors = nice_errors(self.errors, -1)
        nsim *= 2

        paths = np.ones((npoints + 1, nvars, nsim)) * start

        for i in range(npoints):
            paths[i+1] = paths[i] + self.update(paths[i], self.errors[i])

        paths = paths[::ndiscr]
        # Assuming that paths are log prices, then covert to log returns
        #paths = paths[1:] - paths[:-1]
        if nsim > 1:
            return paths
        else:
            return paths.flatten()

    def gmmest(self, theta_start, **kwargs):
        """Estimate model parameters using GMM.

        """
        estimator = GMM(self.momcond)
        return estimator.gmmest(theta_start.theta, **kwargs)


if __name__ == '__main__':
    pass
