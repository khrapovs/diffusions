#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Generic Model

"""
from __future__ import print_function, division

import seaborn as sns

from diffusions import GBM, GBMparam
from diffusions import plot_trajectories, plot_final_distr, plot_realized


def try_simulation():
    mean, sigma = .05, .2
    theta_true = GBMparam(mean, sigma)
    gbm = GBM(theta_true)

    start, nperiods, interval, ndiscr, nsim = 1, 500, .5, 10, 2
    nobs = int(nperiods / interval)
    paths = gbm.simulate(start, interval, ndiscr, nobs, nsim, diff=0)
    data = paths[:, 0, 0]

    plot_trajectories(data, interval)


def try_marginal():
    mean, sigma = .05, .2
    theta_true = GBMparam(mean, sigma)
    gbm = GBM(theta_true)

    start, nperiods, interval, ndiscr, nsim = 1, 500, .5, 10, 20
    nobs = int(nperiods / interval)
    paths = gbm.simulate(start, interval, ndiscr, nobs, nsim, diff=0)
    data = paths[:, :, 0]

    plot_final_distr(data/interval)


def try_gmm():
    mean, sigma = 1.5, .2
    theta_true = GBMparam(mean, sigma)
    gbm = GBM(theta_true)

    start, nperiods, interval, ndiscr, nsim = 1, 500, .5, 10, 1
    nobs = int(nperiods / interval)
    paths = gbm.simulate(start, interval, ndiscr, nobs, nsim, diff=0)
    data = paths[:, 0, 0]

    plot_trajectories(data, interval)

    mean, sigma = 2.5, .4
    theta_start = GBMparam(mean, sigma)
    res = gbm.gmmest(theta_start, data=data, instrlag=2)
    res.print_results()


def try_sim_realized():
    mean, sigma = .05, .2
    theta_true = GBMparam(mean, sigma)
    gbm = GBM(theta_true)

    start, nperiods, interval, ndiscr, nsim = 1, 500, 1/80, 1, 1
    returns, rvar = gbm.sim_realized(start, interval, ndiscr,
                                     nperiods, nsim, diff=0)

    plot_realized(returns, rvar)


if __name__ == '__main__':

    sns.set_context('notebook')
#    try_simulation()
#    try_marginal()
#    try_gmm()
    try_sim_realized()
