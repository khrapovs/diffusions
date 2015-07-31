#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Try Cox-Ingersoll-Ross Model

"""
from __future__ import print_function, division

import seaborn as sns

from diffusions import CIR, CIRparam
from diffusions.helper_functions import (plot_trajectories, plot_final_distr,
                                         plot_realized)


def try_simulation():
    mean, kappa, eta = .5, .1, .2
    theta_true = CIRparam(mean, kappa, eta)
    # 2 * kappa * mean - eta**2 > 0
    print(theta_true.is_valid())
    cir = CIR(theta_true)

    x0, nperiods, interval, ndiscr, nsim = mean, 500, .5, 10, 3
    nobs = int(nperiods / interval)
    paths = cir.simulate(x0, interval=interval, ndiscr=ndiscr,
                         nobs=nobs, nsim=nsim)
    data = paths[:, 0, 0]

    plot_trajectories(data, interval, 'rates')


def try_marginal():
    mean, kappa, eta = .5, .1, .2
    theta_true = CIRparam(mean, kappa, eta)
    # 2 * kappa * mean - eta**2 > 0
    print(theta_true.is_valid())
    cir = CIR(theta_true)

    x0, nperiods, interval, ndiscr, nsim = mean, 500, .5, 10, 20
    nobs = int(nperiods / interval)
    paths = cir.simulate(x0, interval=interval, ndiscr=ndiscr,
                         nobs=nobs, nsim=nsim)
    data = paths[:, :, 0]

    plot_final_distr(data, 'rates')


def try_sim_realized():
    mean, kappa, eta = .5, .1, .2
    theta_true = CIRparam(mean, kappa, eta)
    cir = CIR(theta_true)

    start, nperiods, interval, ndiscr, nsim = 1, 500, 1/80, 1, 1
    aggh = 10
    returns, rvar = cir.sim_realized(start, interval=interval, ndiscr=ndiscr,
                                     aggh=aggh, nperiods=nperiods,
                                     nsim=nsim, diff=0)

    plot_realized(returns, rvar)


if __name__ == '__main__':

    sns.set_context('notebook')
    try_simulation()
    try_marginal()
    try_sim_realized()
