#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Generic model class
-------------------

"""
from __future__ import print_function, division

import numpy as np

from mygmm import GMM
from .helper_functions import (nice_errors, ajd_drift, ajd_diff,
                               rolling_window, columnwise_prod, instruments)
try:
    from .simulate import simulate
except:
    print('Failed to import cython modules. '
          + 'Temporary hack to compile documentation.')

__all__ = ['SDE']


class SDE(object):

    """Generic Model.

    Attributes
    ----------
    param : parameter instance
        True parameters used for simulation of the data

    Methods
    -------
    simulate
        Simulate observations from the model
    sim_realized
        Simulate realized returns and variance
    sim_realized_pq
        Simulate realized returns and variance under both P and Q
    gmmest
        Estimate model parameters using GMM
    integrated_gmm
        Estimate model parameters using Integrated GMM

    """

    def __init__(self, param=None):
        """Initialize the class.

        Parameters
        ----------
        param : parameter instance
            True parameters used for simulation of the data

        """
        self.nsub = None
        self.ndiscr = None
        self.param = param
        self.errors = None

    def update_theta(self, param):
        """Update model parameters.

        Parameters
        ----------
        param : parameter instance
            True parameters used for simulation of the data

        """
        self.param = param

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
        return ajd_drift(state, theta) / self.nsub

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
        return ajd_diff(state, theta) / self.nsub**.5

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
        loc = self.euler_loc(state, self.param)
        # (nsim, nvars, nvars) array_like
        scale = self.euler_scale(state, self.param)

        new_state = loc / self.ndiscr \
            + (np.transpose(scale, axes=[1, 2, 0]) * error.T).sum(1).T \
            / self.ndiscr**.5

        # Equivalent operation through the loop:
#        new_state = loc / self.ndiscr
#        for i in range(error.shape[0]):
#            new_state[i] += (scale[i] * error[i]).sum(1) / self.ndiscr**.5

        return new_state

    def simulate(self, start=None, nsub=80, ndiscr=1, nobs=500, nsim=1,
                 diff=None, new_innov=True, cython=True):
        """Simulate observations from the model.

        Parameters
        ----------
        start : array_like
            Starting value for simulation
        nsub : int
            Interval length
        ndiscr : int
            Number of Euler discretization points inside a subinterval
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
        cython : bool
            Whether to use cython-optimized simulation (True) or not (False)

        Returns
        -------
        paths : (nobs, 2*nsim, nvars) array
            Simulated data

        """
        if start is None:
            start = self.get_start()
        if np.size(self.param.mat_k0) != np.size(start):
            raise ValueError('Start for paths is of wrong dimension!')
        self.nsub = nsub
        self.ndiscr = ndiscr
        nvars = np.size(start)
        npoints = nobs * ndiscr

        if self.errors is None or new_innov:
            # Generate new errors
            self.errors = np.random.normal(size=(npoints, nsim, nvars))
            # Standardize the errors
            self.errors = nice_errors(self.errors, 1)

        if cython:
            dt = 1 / ndiscr / nsub
            paths = simulate(self.errors, np.atleast_1d(start).astype(float),
                             np.atleast_1d(self.param.mat_k0).astype(float),
                             np.atleast_2d(self.param.mat_k1).astype(float),
                             np.atleast_2d(self.param.mat_h0).astype(float),
                             np.atleast_3d(self.param.mat_h1).astype(float),
                             float(dt))
        else:
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

    def sim_realized(self, start=None, nsub=80, ndiscr=10, aggh=1,
                     nperiods=500, nsim=1, diff=None, new_innov=True,
                     cython=True):
        """Simulate realized returns and variance from the model.

        Parameters
        ----------
        start : array_like
            Starting value for simulation
        nsub : int
            Number of subintervals for latent simulation (fractions of the day)
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
        cython : bool
            Whether to use cython-optimized simulation (True) or not (False)

        Returns
        -------
        returns : (nperiods, ) array
            Simulated returns
        rvar : (nperiods, ) array
            Simulated realized variance

        """
        if start is None:
            start = self.get_start()
        nobs = nperiods * nsub
        paths = self.simulate(start, nsub=nsub, ndiscr=ndiscr,
                              nobs=nobs, nsim=nsim, diff=diff,
                              new_innov=new_innov, cython=cython)
        returns = paths[:, 0, 0].reshape((nperiods, nsub))
        # Compute realized var and returns over one day
        rvar = (returns**2).sum(1)
        returns = returns.sum(1)
        # Aggregate over arbitrary number of days
        rvar = rolling_window(np.mean, rvar, window=aggh)
        returns = rolling_window(np.mean, returns, window=aggh)
        return returns, rvar

    def sim_realized_pq(self, start_p=None, start_q=None,
                        aggh=[1, 1], **kwargs):
        """Simulate realized data from the model under both P and Q.

        Parameters
        ----------
        start_p : array_like
            Starting value for simulation under P
        start_q : array_like
            Starting value for simulation under Q
        aggh : list
            Aggregation windows for P and Q respectively
        kwargs : dict
            Anything that needs to go through sim_realized

        Returns
        -------
        data_p : tuple
            Returns and realized variance under P
        data_q : tuple
            Returns and realized variance under Q

        Notes
        -----
        For argumentsts see sim_realized

        """
        if start_p is None:
            start_p = self.get_start()
        data_p = self.sim_realized(start_p, aggh=aggh[0],
                                   new_innov=True, **kwargs)
        self.param.convert_to_q()
        if start_q is None:
            start_q = self.get_start()
        data_q = self.sim_realized(start_q, aggh=aggh[1],
                                   new_innov=False, **kwargs)
        return data_p, data_q

    def gmmest(self, theta_start, **kwargs):
        """Estimate model parameters using GMM.

        Parameters
        ----------
        theta_start : array
            Initial parameter values for estimation
        kwargs : dict
            Anything that needs to go through mygmm

        Notes
        -----
        For arguments see momcond

        """
        estimator = GMM(self.momcond)
        return estimator.gmmest(theta_start.get_theta(), **kwargs)

    def integrated_gmm(self, param_start, subset='all', measure='P',
                       names=None, bounds=None, constraints=(), **kwargs):
        """Estimate model parameters using Integrated GMM.

        Parameters
        ----------
        param_start : parameter class
            Initial parameter values for estimation
        subset : str

            Which parameters to estimate. Belongs to
                - 'all' : all parameters, including those related to returns
                - 'vol' : only those related to volatility

        measure : str

            Under which measure to estimate:
                - 'P' : physical measure
                - 'Q' : risk-neutral
                - 'PQ' : both

        names : list of str
            Parameter names
        bounds : list of tuples
            Parameter bounds
        constraints : dict or sequence of dict
            Equality and inequality constraints. See scipy.optimize.minimize
        kwargs : dict
            Anything that needs to go through mygmm

        Notes
        -----
        For arguments see integrated_mom

        """
        estimator = GMM(self.integrated_mom)
        self.param = param_start
        theta_start = self.param.get_theta(subset=subset, measure=measure)
        if names is None:
            names = self.param.get_names(subset=subset, measure=measure)
        if bounds is None:
            bounds = self.param.get_bounds(subset=subset, measure=measure)
        if constraints == ():
            constraints = self.param.get_constraints()
        return estimator.gmmest(theta_start, names=names, subset=subset,
                                measure=measure, bounds=bounds,
                                constraints=constraints, **kwargs)

    def integrated_mom(self, theta, data=None, instr_data=None,
                       instr_choice='const', aggh=1, subset='all',
                       instrlag=1, measure='P', **kwargs):
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
            Which parameters to estimate. Belongs to
                - 'all' : all parameters, including those related to returns
                - 'vol' : only those related to volatility
        measure : str
            Under which measure to estimate:
                - 'P' : physical measure
                - 'Q' : risk-neutral
                - 'PQ' : both
        kwargs : dict
            Anything that needs to go through mygmm

        Returns
        -------
        moments : (nobs - instrlag - 2, 3 * ninstr = nmoms) array
            Moment restrictions
        dmoments : (nmoms, nparams) array
            Average derivative of the moment restrictions

        """
        subset_sl = None
        if subset == 'vol':
            subset_sl = slice(2)

        self.param.update(theta=theta, subset=subset, measure=measure)
        lag = 2

        if measure == 'PQ':
            error = []
            for data_x, agg, meas in zip(data, aggh, measure):
                if meas == 'Q':
                    self.param.convert_to_q()
                depvar = self.realized_depvar(data_x)[lag:]
                # (nobs - lag, 4) array
                error.append(depvar.dot(self.mat_a(self.param, subset_sl).T) \
                    - self.realized_const(self.param, agg, subset_sl))

            error = np.hstack(error)

        else:
            depvar = self.realized_depvar(data)[lag:]
            # (nobs - lag, 4) array
            error = depvar.dot(self.mat_a(self.param, subset_sl).T) \
                - self.realized_const(self.param, aggh, subset_sl)

        nobs = error.shape[0] + lag
        # self.instruments(data, instrlag=instrlag): (nobs, ninstr*instrlag+1)
        # (nobs-lag, ninstr*instrlag+1)
        instr = instruments(instr_data, nobs=nobs, instrlag=instrlag,
                            instr_choice=instr_choice)[:-lag]
        # (nobs - instrlag - lag, 4 * (ninstr*instrlag + 1))
        moms = columnwise_prod(error, instr)

        return moms, None
