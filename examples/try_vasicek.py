#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Try Vasicek Model

"""
from __future__ import print_function, division

import seaborn as sns

from diffusions import Vasicek, VasicekParam
from diffusions.helper_functions import (plot_trajectories, plot_final_distr,
                                         plot_realized)


def try_simulation():
    mean, kappa, eta = .5, .1, .2
    theta_true = VasicekParam(mean, kappa, eta)
    vasicek = Vasicek(theta_true)

    x0, nperiods, nsub, ndiscr, nsim = 1, 500, 2, 10, 3
    nobs = nperiods * nsub
    paths = vasicek.simulate(x0, nsub=nsub, ndiscr=ndiscr,
                             nobs=nobs, nsim=nsim)
    data = paths[:, 0, 0]

    plot_trajectories(data, nsub, 'returns')


def try_marginal():
    mean, kappa, eta = .5, .1, .2
    theta_true = VasicekParam(mean, kappa, eta)
    vasicek = Vasicek(theta_true)

    x0, nperiods, nsub, ndiscr, nsim = mean, 500, 2, 10, 20
    nobs = nperiods * nsub
    paths = vasicek.simulate(x0, nsub=nsub, ndiscr=ndiscr,
                             nobs=nobs, nsim=nsim)
    data = paths[:, :, 0]

    plot_final_distr(data, 'returns')


def try_sim_realized():
    mean, kappa, eta = .5, .1, .2
    theta_true = VasicekParam(mean, kappa, eta)
    vasicek = Vasicek(theta_true)

    start, nperiods, nsub, ndiscr, nsim = 1, 500, 80, 1, 1
    aggh = 10
    returns, rvar = vasicek.sim_realized(start, nsub=nsub,
                                         ndiscr=ndiscr, aggh=aggh,
                                         nperiods=nperiods, nsim=nsim, diff=0)

    plot_realized(returns, rvar)


if __name__ == '__main__':

    sns.set_context('notebook')
    try_marginal()
    try_simulation()
    try_sim_realized()
