#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Try Vasicek Model

"""
from __future__ import print_function, division

import seaborn as sns

from diffusions import Vasicek, VasicekParam
from diffusions import plot_trajectories, plot_final_distr, plot_realized


def try_simulation():
    mean, kappa, eta = .5, .1, .2
    theta_true = VasicekParam(mean, kappa, eta)
    vasicek = Vasicek(theta_true)

    x0, nperiods, interval, ndiscr, nsim = 1, 500, .5, 10, 3
    npoints = int(nperiods / interval)
    paths = vasicek.simulate(x0, interval, ndiscr, npoints, nsim)
    data = paths[:, 0, 0]

    plot_trajectories(data, interval)


def try_marginal():
    mean, kappa, eta = .5, .1, .2
    theta_true = VasicekParam(mean, kappa, eta)
    vasicek = Vasicek(theta_true)

    x0, nperiods, interval, ndiscr, nsim = mean, 500, .5, 10, 20
    nobs = int(nperiods / interval)
    paths = vasicek.simulate(x0, interval, ndiscr, nobs, nsim)
    data = paths[:, :, 0]

    plot_final_distr(data)


def try_sim_realized():
    mean, kappa, eta = .5, .1, .2
    theta_true = VasicekParam(mean, kappa, eta)
    vasicek = Vasicek(theta_true)

    start, nperiods, interval, ndiscr, nsim = 1, 500, 1/80, 1, 1
    aggh = 10
    returns, rvar = vasicek.sim_realized(start, interval=interval,
                                         ndiscr=ndiscr, aggh=aggh,
                                         nperiods=nperiods, nsim=nsim, diff=0)

    plot_realized(returns, rvar)


if __name__ == '__main__':

    sns.set_context('notebook')
#    try_marginal()
#    try_simulation()
    try_sim_realized()
