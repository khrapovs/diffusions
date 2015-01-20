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

    def moment(self, theta, x):
        loc = self.exact_loc(x[:-1], theta)
        scale = self.exact_scale(x[:-1], theta)
        X1 = x[1:] - loc
        X2 = x[1:]**2 - loc**2 - scale**2
        X = np.vstack([X1, X2])
        Z = np.vstack([np.ones_like(x[:-1]), x[:-1]])
        new_shape = (X.shape[0] * Z.shape[0], X.shape[1])
        g = np.reshape(X[:, np.newaxis, :] * Z[np.newaxis, :, :], new_shape)
        return np.mat(g)

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
