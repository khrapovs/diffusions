#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Try Geometric Brownian Motion

"""
from __future__ import print_function, division

import seaborn as sns
import numpy as np

from diffusions import GBM, GBMparam
from diffusions.helper_functions import (plot_trajectories, plot_final_distr,
                                         plot_realized, take_time)


def try_simulation():
    mean, sigma = .05, .2
    theta_true = GBMparam(mean, sigma)
    print(theta_true)

    gbm = GBM(theta_true)

    start, nperiods, nsub, ndiscr, nsim = 1, 500, 2, 10, 2
    nobs = nperiods * nsub
    paths = gbm.simulate(start, nsub=nsub, ndiscr=ndiscr, nobs=nobs,
                         nsim=nsim, diff=0)
    data = paths[:, 0, 0]

    plot_trajectories(data, nsub, 'returns')


def try_marginal():
    mean, sigma = .05, .2
    theta_true = GBMparam(mean, sigma)
    gbm = GBM(theta_true)

    start, nperiods, nsub, ndiscr, nsim = 1, 500, 2, 10, 20
    nobs = nperiods * nsub
    paths = gbm.simulate(start, nsub=nsub, ndiscr=ndiscr, nobs=nobs,
                         nsim=nsim, diff=0)
    data = paths[:, :, 0]

    plot_final_distr(data * nsub, 'returns')


def try_gmm():
    mean, sigma = 1.5, .2
    theta_true = GBMparam(mean, sigma)
    gbm = GBM(theta_true)

    start, nperiods, nsub, ndiscr, nsim = 1, 500, 2, 10, 1
    nobs = nperiods * nsub
    paths = gbm.simulate(start, nsub=nsub, ndiscr=ndiscr, nobs=nobs,
                         nsim=nsim, diff=0)
    data = paths[:, 0, 0]

    plot_trajectories(data, nsub, 'returns')

    mean, sigma = 2.5, .4
    theta_start = GBMparam(mean, sigma)
    res = gbm.gmmest(theta_start, data=data, instrlag=2)
    print(res)


def try_sim_realized():
    mean, sigma = .05, .2
    theta_true = GBMparam(mean, sigma)
    gbm = GBM(theta_true)

    start, nperiods, nsub, ndiscr, nsim = 1, 500, 80, 1, 1
    aggh = 10
    returns, rvar = gbm.sim_realized(start, nsub=nsub, ndiscr=ndiscr,
                                     aggh=aggh, nperiods=nperiods,
                                     nsim=nsim, diff=0)

    plot_realized(returns, rvar)


def try_integrated_gmm():
    mean, sigma = 1.5, .2
    theta_true = GBMparam(mean, sigma)
    gbm = GBM(theta_true)

    start, nperiods, nsub, ndiscr, nsim = 1, 500, 80, 1, 1
    aggh = 10
    returns, rvar = gbm.sim_realized(start, nsub=nsub, ndiscr=ndiscr,
                                     aggh=aggh, nperiods=nperiods,
                                     nsim=nsim, diff=0)
    data = np.vstack([returns, rvar])
    print(rvar.mean()**.5)
    plot_realized(returns, rvar)

    mean, sigma = 2.5, .4
    theta_start = GBMparam(mean, sigma)
    res = gbm.integrated_gmm(theta_start, data=data, instrlag=2)
    print(res)


if __name__ == '__main__':

    sns.set_context('notebook')
#    with take_time('Simulation'):
#        try_simulation()
#    with take_time('Marginal density'):
#        try_marginal()
#    with take_time('Simulate RV'):
#        try_sim_realized()
#    with take_time('GMM'):
#        try_gmm()
    with take_time('Integrated GMM'):
        try_integrated_gmm()
