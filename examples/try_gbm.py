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
                                         plot_realized)


def try_simulation():
    mean, sigma = .05, .2
    theta_true = GBMparam(mean, sigma)
    print(theta_true)

    gbm = GBM(theta_true)

    start, nperiods, interval, ndiscr, nsim = 1, 500, .5, 10, 2
    nobs = int(nperiods / interval)
    paths = gbm.simulate(start, interval=interval, ndiscr=ndiscr, nobs=nobs,
                         nsim=nsim, diff=0)
    data = paths[:, 0, 0]

    plot_trajectories(data, interval, 'returns')


def try_marginal():
    mean, sigma = .05, .2
    theta_true = GBMparam(mean, sigma)
    gbm = GBM(theta_true)

    start, nperiods, interval, ndiscr, nsim = 1, 500, .5, 10, 20
    nobs = int(nperiods / interval)
    paths = gbm.simulate(start, interval=interval, ndiscr=ndiscr, nobs=nobs,
                         nsim=nsim, diff=0)
    data = paths[:, :, 0]

    plot_final_distr(data/interval, 'returns')


def try_gmm():
    mean, sigma = 1.5, .2
    theta_true = GBMparam(mean, sigma)
    gbm = GBM(theta_true)

    start, nperiods, interval, ndiscr, nsim = 1, 500, .5, 10, 1
    nobs = int(nperiods / interval)
    paths = gbm.simulate(start, interval=interval, ndiscr=ndiscr, nobs=nobs,
                         nsim=nsim, diff=0)
    data = paths[:, 0, 0]

    plot_trajectories(data, interval, 'returns')

    mean, sigma = 2.5, .4
    theta_start = GBMparam(mean, sigma)
    res = gbm.gmmest(theta_start, data=data, instrlag=2)
    print(res)


def try_sim_realized():
    mean, sigma = .05, .2
    theta_true = GBMparam(mean, sigma)
    gbm = GBM(theta_true)

    start, nperiods, interval, ndiscr, nsim = 1, 500, 1/80, 1, 1
    aggh = 10
    returns, rvar = gbm.sim_realized(start, interval=interval, ndiscr=ndiscr,
                                     aggh=aggh, nperiods=nperiods,
                                     nsim=nsim, diff=0)

    plot_realized(returns, rvar)


def try_integrated_gmm():
    mean, sigma = 1.5, .2
    theta_true = GBMparam(mean, sigma)
    gbm = GBM(theta_true)

    start, nperiods, interval, ndiscr, nsim = 1, 500, 1/80, 1, 1
    aggh = 10
    returns, rvar = gbm.sim_realized(start, interval=interval, ndiscr=ndiscr,
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
#    try_simulation()
#    try_marginal()
#    try_sim_realized()
#    try_gmm()
#    try_integrated_gmm()

