#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Try Central Tendency model

"""
from __future__ import print_function, division

import time
import itertools

import numpy as np
import seaborn as sns

from diffusions import CentTend, CentTendParam
from diffusions import plot_trajectories, plot_final_distr, plot_realized


def try_simulation():

    riskfree = .01
    lmbd = .01
    mean_v = .5
    kappa_s = 1.5
    kappa_v = .05
    eta_s = .02**.5
    eta_v = .001**.5
    rho = -.9

    theta_true = CentTendParam(riskfree=riskfree, lmbd=lmbd,
                               mean_v=mean_v, kappa_s=kappa_s, kappa_v=kappa_v,
                               eta_s=eta_s, eta_v=eta_v, rho=rho)
    centtend = CentTend(theta_true)
    print(theta_true.is_valid())

    start = [1, mean_v, mean_v]
    nperiods, interval, ndiscr, nsim = 500, .1, 10, 3
    npoints = int(nperiods / interval)
    paths = centtend.simulate(start, interval, ndiscr, npoints, nsim, diff=0)

    returns = paths[:, 0, 0]
    volatility = paths[:, 0, 1]
    tendency = paths[:, 0, 2]

    plot_trajectories(returns, interval, 'returns')
    names = ['vol', 'ct']
    plot_trajectories([volatility, tendency], interval, names)


def try_marginal():

    riskfree = .01
    lmbd = .01
    mean_v = .5
    kappa_s = 1.5
    kappa_v = .05
    eta_s = .02**.5
    eta_v = .001**.5
    rho = -.9

    theta_true = CentTendParam(riskfree=riskfree, lmbd=lmbd,
                               mean_v=mean_v, kappa_s=kappa_s, kappa_v=kappa_v,
                               eta_s=eta_s, eta_v=eta_v, rho=rho)
    centtend = CentTend(theta_true)
    print(theta_true.is_valid())

    start = [1, mean_v, mean_v]
    nperiods, interval, ndiscr, nsim = 500, .1, 10, 1000
    npoints = int(nperiods / interval)
    paths = centtend.simulate(start, interval, ndiscr, npoints, nsim, diff=0)

    returns = paths[:, :, 0]
    volatility = paths[:, :, 1]
    tendency = paths[:, :, 2]

    plot_final_distr(returns, 'returns')
    names = ['vol', 'ct']
    plot_final_distr([volatility, tendency], names)


def try_sim_realized():

    riskfree = .01
    lmbd = .01
    mean_v = .2
    kappa_s = .1
    kappa_v = .05
    eta_s = .01**.5
    eta_v = .001**.5
    rho = -.9

    theta_true = CentTendParam(riskfree=riskfree, lmbd=lmbd,
                               mean_v=mean_v, kappa_s=kappa_s, kappa_v=kappa_v,
                               eta_s=eta_s, eta_v=eta_v, rho=rho)
    centtend = CentTend(theta_true)
    print(theta_true.is_valid())

    start = [1, mean_v, mean_v]
    nperiods, interval, ndiscr, nsim = 1000, 1/80, 1, 1
    aggh = 1

    returns, rvar = centtend.sim_realized(start, interval=interval,
                                          ndiscr=ndiscr, aggh=aggh,
                                          nperiods=nperiods, nsim=nsim, diff=0)

    plot_realized(returns, rvar)


def try_integrated_gmm():

    riskfree = .0

    lmbd = .01
    mean_v = .2
    kappa_s = .1
    kappa_v = .05
    eta_s = .01**.5
    eta_v = .001**.5
    rho = -.9

    theta_true = CentTendParam(riskfree=riskfree, lmbd=lmbd,
                               mean_v=mean_v, kappa_s=kappa_s, kappa_v=kappa_v,
                               eta_s=eta_s, eta_v=eta_v, rho=rho)
    centtend = CentTend(theta_true)
    print(theta_true.is_valid())

    start = [1, mean_v, mean_v]
    nperiods, interval, ndiscr, nsim = 2000, 1/80, 1, 1
    aggh = 1
    data = centtend.sim_realized(start, interval=interval, ndiscr=ndiscr,
                                 aggh=aggh, nperiods=nperiods,
                                 nsim=nsim, diff=0)
    ret, rvar = data
    plot_realized(ret, rvar)

    instr_data = np.vstack([rvar, rvar**2])

    theta_start = theta_true
    theta_start.update(theta_true.get_theta()/2)

    tasks = itertools.product(np.arange(1, 4), ['L-BFGS-B', 'TNC', 'SLSQP'])
    for lag, method in tasks:
        time_start = time.time()
        res = centtend.integrated_gmm(theta_start, data=data, instrlag=lag,
                                      instr_data=instr_data, aggh=aggh,
                                      instr_choice='var', method=method,
                                      use_jacob=True, exact_jacob=False,
                                      bounds=theta_start.get_bounds(), iter=3)
        res.print_results()
        print(lag, method)
        print('Elapsed time = %.2f min' % ((time.time() - time_start)/60))


if __name__ == '__main__':

    np.set_printoptions(precision=4, suppress=True)
    sns.set_context('notebook')
#    try_simulation()
#    try_marginal()
    try_sim_realized()
#    try_integrated_gmm()
