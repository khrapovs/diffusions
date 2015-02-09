#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Helper functions

"""
from __future__ import print_function, division

import numpy as np
import matplotlib.pylab as plt
import seaborn as sns

__all__ = ['nice_errors', 'plot_trajectories', 'plot_final_distr',
           'ajd_drift', 'ajd_diff']


def ajd_drift(state, theta):
    """Instantaneous mean.

    Parameters
    ----------
    state : (nsim, nvars) array_like
        Current value of the process
    theta : parameter instance
        Model parameter

    Returns
    -------
    (nsim, nvars) array_like
        Value of the drift

    """
    state = np.atleast_2d(state)
    return theta.mat_k0 + state.dot(np.transpose(theta.mat_k1))


def ajd_diff(state, theta):
    """Instantaneous volatility.

    Parameters
    ----------
    state : (nsim, nvars) array_like
        Current value of the process
    theta : parameter instance
        Model parameter

    Returns
    -------
    (nsim, nvars, nvars) array_like
        Value of the diffusion

    """
    state = np.atleast_2d(state)
    mat_h1 = np.atleast_3d(theta.mat_h1)
    # (nsim, nvars, nvars)
    var = theta.mat_h0 + np.tensordot(state, mat_h1, axes=(1, 0))
    try:
        return np.linalg.cholesky(var)
    except(np.linalg.LinAlgError):
        return np.ones_like(var) * 1e10


def nice_errors(errors, sdim):
    """Normalize the errors and apply antithetic sampling.

    Parameters
    ----------
    errors : array
        Innovations to be standardized
    sdim : int
        Which dimension corresponds to simulation instances?

    Returns
    -------
    errors : array
        Standardized innovations

    """
    if errors.shape[sdim] > 10:
        errors -= errors.mean(sdim, keepdims=True)
        errors /= errors.std(sdim, keepdims=True)
    return np.concatenate((errors, -errors), axis=sdim)


def plot_trajectories(paths, interval):
    """Plot process realizations.

    Parameters
    ----------
    paths : array
        Process realizations. Shape is either (nobs,) or (nobs, nsim)
    interval : float
        Length of unit interval

    """
    x = np.arange(0, interval * paths.shape[0], interval)
    plt.plot(x, paths)
    plt.xlabel('$t$')
    plt.ylabel('$x_t$')
    plt.show()


def plot_final_distr(paths):
    """Plot marginal distribution of the process.

    Parameters
    ----------
    paths : array
        Process realizations. Shape is (nobs, nsim)

    """
    if paths.ndim != 2:
        raise ValueError('Simulate more paths!')
    sns.kdeplot(paths[-1])
    plt.xlabel('x')
    plt.ylabel('f')
    plt.show()
