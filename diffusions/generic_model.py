#!/usr/bin/env python
# -*- coding: utf-8 -*-
r"""
Affine Diffusion
================

A jump-diffusion process is a Markov process solving the stochastic
differential equationd

.. math::
    Y_{t}=\mu\left(Y_{t},\theta_{0}\right)dt
        +\sigma\left(Y_{t},\theta_{0}\right)dW_{t}.

A discount-rate function :math:`R:D\to\mathbb{R}` is an affine function of the
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

"""
from __future__ import print_function, division

import numpy as np

from diffusions.mygmm import GMM
from .helper_functions import (nice_errors, ajd_drift, ajd_diff,
                               rolling_window, columnwise_prod, instruments)

__all__ = ['SDE']


class SDE(object):

    """Generic Model.

    Attributes
    ----------
    theta_true : parameter instance
        True parameters used for simulation of the data

    Methods
    -------
    simulate
        Simulate observations from the model
    sim_realized
        Simulate realized returns and variance from the model
    gmmest
        Estimate model parameters using GMM
    integrated_gmm
        Estimate model parameters using Integrated GMM

    """

    def __init__(self, theta_true=None):
        """Initialize the class.

        Parameters
        ----------
        theta_true : parameter instance
            True parameters used for simulation of the data

        """
        self.interval = None
        self.ndiscr = None
        self.theta_true = theta_true
        self.errors = None

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

    def depvar_unc_mean(self, param, aggh):
        """Unconditional means of realized data.

        Parameters
        ----------
        param : parameter instance
            Model parameters
        aggh : float
            Interval length

        Returns
        -------
        array

        """
        return np.array([self.mean_vol(param, aggh),
                         self.mean_vol2(param, aggh),
                         self.mean_ret(param, aggh),
                         self.mean_cross(param, aggh)])

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

    def simulate(self, start, interval, ndiscr, nobs, nsim, diff=None,
                 new_innov=True):
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
        new_innov : bool
            Whether to generate new innovations (True),
            or use already stored (False)

        Returns
        -------
        paths : (nobs, 2*nsim, nvars) array
            Simulated data

        """
        if np.size(self.theta_true.mat_k0) != np.size(start):
            raise ValueError('Start for paths is of wrong dimension!')
        self.interval = interval
        self.ndiscr = ndiscr
        nvars = np.size(start)
        npoints = nobs * ndiscr

        if self.errors is None or new_innov:
            # Generate new errors
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

    def sim_realized(self, start, interval=1/80, ndiscr=1, aggh=1,
                     nperiods=500, nsim=1, diff=None, new_innov=True):
        """Simulate realized returns and variance from the model.

        Parameters
        ----------
        start : array_like
            Starting value for simulation
        interval : float
            Interval length for latent simulation (fraction of the day)
        ndiscr : int
            Number of Euler discretization points inside unit interval
        aggh : int
            Number of intervals (days) to aggregate over using rolling mean
        nperiods : int
            Number of points to simulate in one series (days)
        nsim : int
            Number of time series to simulate
        diff : int
            Dimensions which should be differentiated,
            i.e. return = price[1:] - price[:-1]
        new_innov : bool
            Whether to generate new innovations (True),
            or use already stored (False)

        Returns
        -------
        returns : (nperiods, ) array
            Simulated returns
        rvar : (nperiods, ) array
            Simulated realized variance

        """
        intervals = int(1 / interval)
        nobs = nperiods * intervals
        paths = self.simulate(start, interval, ndiscr, nobs, nsim, diff,
                              new_innov)
        returns = paths[:, 0, 0].reshape((nperiods, intervals))
        # Compute realized var and returns over one day
        rvar = (returns**2).sum(1)
        returns = returns.sum(1)
        # Aggregate over arbitrary number of days
        rvar = rolling_window(np.mean, rvar, window=aggh)
        returns = rolling_window(np.mean, returns, window=aggh)
        return returns, rvar

    def gmmest(self, theta_start, **kwargs):
        """Estimate model parameters using GMM.

        Parameters
        ----------
        theta_start : parameter instance
            Initial parameter values for estimation.
            Object with mandatory attribute 'theta'.

        """
        estimator = GMM(self.momcond)
        return estimator.gmmest(theta_start.get_theta(), **kwargs)

    def integrated_gmm(self, theta_start, **kwargs):
        """Estimate model parameters using Integrated GMM.

        Parameters
        ----------
        theta_start : parameter instance
            Initial parameter values for estimation.
            Object with mandatory attribute 'theta'.

        """
        estimator = GMM(self.integrated_mom)
        return estimator.gmmest(theta_start, **kwargs)

    def integrated_mom(self, theta, data=None, instr_data=None,
                       instr_choice='const', aggh=1, subset='all',
                       instrlag=1, **kwargs):
        """Integrated moment function.

        Parameters
        ----------
        theta : array
            Model parameters
        data : (2, nobs) array
            Returns and realized variance
        instr_data : (ninstr, nobs) array
            Instruments (no lags)
        instrlag : int
            Number of lags for the instruments
        instr_choice : str {'const', 'var'}
            Choice of instruments.
                - 'const' : just a constant (unconditional moments)
                - 'var' : lags of instrument data
        aggh : int
            Number of intervals (days) to aggregate over using rolling mean
        subset : str
            Which parameters to estimate. Belongs to ['all', 'vol']

        Returns
        -------
        moments : (nobs - instrlag - 2, 3 * ninstr = nmoms) array
            Moment restrictions
        dmoments : (nmoms, nparams) array
            Average derivative of the moment restrictions

        """
        param, subset_sl = self.convert(theta, subset)

        ret, rvar = data
        lag = 2
        self.aggh = aggh
        # self.realized_depvar(data): (nobs, 3*nmoms)
        depvar = self.realized_depvar(data)[lag:]
        # (nobs - lag, 4) array
        error = depvar.dot(self.mat_a(param, subset_sl).T) \
            - self.realized_const(param, aggh, subset_sl)

        # self.instruments(data, instrlag=instrlag): (nobs, ninstr*instrlag+1)
        # (nobs-lag, ninstr*instrlag+1)
        instr = instruments(instr_data, nobs=rvar.size, instrlag=instrlag,
                            instr_choice=instr_choice)[:-lag]
        # (nobs - instrlag - lag, 4 * (ninstr*instrlag + 1))
        moms = columnwise_prod(error, instr)

        return moms, None


if __name__ == '__main__':
    pass
