#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Generic Model

"""
from __future__ import print_function, division

import numpy as np
import matplotlib.pylab as plt
import seaborn as sns

from mygmm import GMM

__all__ = ['SDE', 'plot_trajectories', 'plot_final_distr']


class SDE(object):

    r"""Generic Model.

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


    Attributes
    ----------

    Methods
    -------

    """

    def __init__(self, theta_true=None):
        self.paths = None
        self.eps = None
        self.interval = None
        self.nobs = None
        self.theta_true = theta_true

    def euler_loc(self, x, theta):
        return self.drift(x, theta) * self.interval

    def euler_scale(self, x, theta):
        return self.diff(x, theta) * self.interval**.5

    def exact_loc(self, x, theta):
        return self.euler_loc(x, theta)

    def exact_scale(self, x, theta):
        return self.euler_scale(x, theta)

    def update(self, state, error):
        """Euler update function.

        Parameters
        ----------
        state : array
            Current value of the process
        error : array
            Random shocks

        Returns
        -------
        array
            Update of the process value. Same shape as the input.

        """
        return self.euler_loc(state, self.theta_true)/self.ndiscr \
            + self.euler_scale(state, self.theta_true)/self.ndiscr**.5 * error

    def simulate(self, start, interval, ndiscr, nobs, nsim):
        """Simulate observations from the model.

        Parameters
        ----------
        start : array_like
            Starting value for simulation
        interval : float
            Interval length
        ndiscr : int
            Number of discretization points inside unit interval
        nobs : int
            Number of points to simulate in one series
        nsim : int
            Number of time series to simulate

        """
        self.interval = interval
        self.nobs = nobs
        self.ndiscr = ndiscr

        npoints = nobs * ndiscr
        self.errors = np.random.normal(size=(npoints, nsim))
        # Normalize the errors
        self.errors -= self.errors.mean(0)
        self.errors /= self.errors.std(0)

        paths = np.ones((npoints + 1, nsim)) * start

        for i in range(npoints):
            paths[i+1] = paths[i] + self.update(paths[i], self.errors[i])

        paths = paths[::ndiscr]
        # Assuming that paths are log prices, then covert to log returns
        paths = paths[1:] - paths[:-1]
        if nsim > 1:
            self.paths = paths
        else:
            self.paths = paths.flatten()

    def gmmest(self, theta_start, **kwargs):
        """Estimate model parameters using GMM.

        """
        estimator = GMM(self.momcond)
        return estimator.gmmest(theta_start.theta, **kwargs)


def plot_trajectories(paths, interval):
    """Plot process realizations.

    Parameters
    ----------
    paths : array


    """
    x = np.arange(0, interval * paths.shape[0], interval)
    plt.plot(x, paths)
    plt.xlabel('$t$')
    plt.ylabel('$x_t$')
    plt.show()


def plot_final_distr(paths):
    sns.kdeplot(paths[-1])
    plt.xlabel('x')
    plt.ylabel('f')
    plt.show()


if __name__ == '__main__':
    pass
