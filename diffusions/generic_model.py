#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Generic Model

"""
from __future__ import print_function, division

import numpy as np
import matplotlib.pylab as plt
import seaborn as sns

__all__ = ['SDE']


class SDE(object):

    r"""Generic Model.

    Given the generic continuous-time diffusion model

    .. math::
        dY_{t}=\mu\left(Y_{t},\theta_{0}\right)dt
            +\sigma\left(Y_{t},\theta_{0}\right)dW_{t}

    we can discretize it as

    .. math::
        Y_{t}\approx Y_{t}+\mu\left(Y_{t-h},\theta_{0}\right)h
            +\sigma\left(Y_{t-h},\theta_{0}\right)\sqrt{h}\varepsilon_{t}.

    Attributes
    ----------

    Methods
    -------

    """

    def __init__(self, theta_true=None):
        self.paths = None
        self.eps = None
        self.interval = None
        self.nperiods = None
        self.theta_true = theta_true

    def euler_loc(self, x, theta):
        return self.drift(x, theta) * self.interval

    def euler_scale(self, x, theta):
        return self.diff(x, theta) * self.interval**.5

    def sim(self, state, error):
        """Euler update function for return equation.

        """
        # Number of discretization intervals
        ndiscr = error.shape[0]
        return self.euler_loc(state, self.theta_true) / ndiscr \
            + self.euler_scale(state, self.theta_true) / ndiscr**.5 * error

    def simulate(self, start, interval, ndiscr, nperiods, nsim):
        """Simulate observations from the model.

        Parameters
        ----------
        start : array_like
            Starting value for simulation
        interval : float
            Interval length
        ndiscr : int
            Number of discretization points inside unit interval
        nperiods : int
            Number of points to simulate in one series
        nsim : int
            Number of time series to simulate

        """
        self.interval = interval
        self.nperiods = nperiods
        size = (nperiods, ndiscr, nsim)
        self.eps = np.random.normal(size=size, scale=interval**.5)
        x = np.ones((nperiods, nsim)) * start

        for n in range(nperiods-1):
            x[n+1] = reduce(self.sim, self.eps[n], x[n])

        if nsim > 1:
            self.paths = x
        else:
            self.paths = x.flatten()

    def plot_trajectories(self, num):
        if self.paths is None:
            ValueError('Simulate data first!')
        else:
            x = np.arange(0, self.interval * self.nperiods, self.interval)
            plt.plot(x, self.paths[:, :num])
            plt.xlabel('$t$')
            plt.ylabel('$x_t$')
            plt.show()

    def plot_final_distr(self):
        if self.paths is None:
            ValueError('Simulate data first!')
        else:
            data = self.paths[-1]
            sns.kdeplot(data)
            plt.xlabel('x')
            plt.ylabel('f')
            plt.show()


def reduce(sim, eps, x):
    for e in eps:
        x = sim(x, e)
    return x


if __name__ == '__main__':
    pass
