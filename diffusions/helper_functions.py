#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Helper functions

"""
from __future__ import print_function, division

import numpy as np
import matplotlib.pylab as plt
import seaborn as sns

__all__ = ['nice_errors', 'ajd_drift', 'ajd_diff',
           'plot_trajectories', 'plot_final_distr', 'plot_realized',
           'columnwise_prod', 'rolling_window']


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


def plot_realized(returns, rvar):
    """Plot realized returns and volatility.

    Parameters
    ----------
    returns : array
        Returns
    rvar : array
        Realized variance

    """
    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(7, 6))
    axes[0].plot(returns, label='Returns')
    axes[1].plot(rvar**.5, label='Realized volatility')
    axes[0].legend()
    axes[1].legend()
    plt.show()


def columnwise_prod(left, right):
    """Columnwise kronker product.

    Parameters
    ----------
    left : (n, m) array
    right : (n, p) array

    Returns
    -------
    (n, m*p) array
        [left * right[:, 0], ..., left * right[:, -1]]

    Example
    -------
    >>> left = np.arange(6).reshape((3,2))
    >>> left
    array([[0, 1],
           [2, 3],
           [4, 5]])
    >>> right = np.arange(9).reshape((3,3))
    >>> right
    array([[0, 1, 2],
           [3, 4, 5],
           [6, 7, 8]])
    >>> columnwise_prod(left, right)
    array([[ 0,  0,  0,  1,  0,  2],
           [ 6,  9,  8, 12, 10, 15],
           [24, 30, 28, 35, 32, 40]])

    """
    prod = left[:, np.newaxis, :] * right[:, :, np.newaxis]
    return prod.reshape((left.shape[0], left.shape[1] * right.shape[1]))


def rolling_window(fun, mat, window=1):
    """Rolling window apply.

    Source: http://www.rigtorp.se/2011/01/01/rolling-statistics-numpy.html

    Parameters
    ----------
    fun : function
        Function to apply
    mat : array_like
        Data to transform
    window : int
        Window size
    axis : int
        Which axis to apply to

    Returns
    -------
    array

    Examples
    --------
    .. doctest::

        >>> mat = np.arange(10).reshape((2,5))
        >>> mat
        array([[0, 1, 2, 3, 4],
               [5, 6, 7, 8, 9]])
        >>> rolling_window(np.mean, mat, window=2)
        array([[ 0.5,  1.5,  2.5,  3.5],
               [ 5.5,  6.5,  7.5,  8.5]])

    """
    shape = mat.shape[:-1] + (mat.shape[-1] - window + 1, window)
    strides = mat.strides + (mat.strides[-1],)
    mat = np.lib.stride_tricks.as_strided(mat, shape=shape, strides=strides)
    return np.apply_along_axis(fun, -1, mat)
