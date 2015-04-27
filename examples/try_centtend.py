#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Try Central Tendency model

"""
from __future__ import print_function, division

from statsmodels.tsa.stattools import acf

import time
import itertools

import numpy as np
import matplotlib.pylab as plt
import seaborn as sns

from diffusions import CentTend, CentTendParam
from diffusions import plot_trajectories, plot_final_distr, plot_realized
from .load_real_data import load_data


def try_simulation():

    riskfree = .01
    lmbd = .01
    mean_v = .5
    kappa_s = 1.5
    kappa_y = .05
    eta_s = .02**.5
    eta_y = .001**.5
    rho = -.9

    param_true = CentTendParam(riskfree=riskfree, lmbd=lmbd,
                               mean_v=mean_v, kappa_s=kappa_s, kappa_y=kappa_y,
                               eta_s=eta_s, eta_y=eta_y, rho=rho)
    centtend = CentTend(param_true)
    print(param_true.is_valid())

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
    kappa_y = .05
    eta_s = .02**.5
    eta_y = .001**.5
    rho = -.9

    param_true = CentTendParam(riskfree=riskfree, lmbd=lmbd,
                               mean_v=mean_v, kappa_s=kappa_s, kappa_y=kappa_y,
                               eta_s=eta_s, eta_y=eta_y, rho=rho)
    centtend = CentTend(param_true)
    print(param_true.is_valid())

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
    kappa_y = .05
    eta_s = .01**.5
    eta_y = .001**.5
    rho = -.9

    param_true = CentTendParam(riskfree=riskfree, lmbd=lmbd,
                               mean_v=mean_v, kappa_s=kappa_s, kappa_y=kappa_y,
                               eta_s=eta_s, eta_y=eta_y, rho=rho)
    centtend = CentTend(param_true)
    print(param_true.is_valid())

    start = [1, mean_v, mean_v]
    nperiods, interval, ndiscr, nsim = 2000, 1/80, 1, 1
    aggh = 1

    returns, rvar = centtend.sim_realized(start, interval=interval,
                                          ndiscr=ndiscr, aggh=aggh,
                                          nperiods=nperiods, nsim=nsim, diff=0)

    plot_realized(returns, rvar)

    nlags = 90
    lw = 2
    grid = range(nlags+1)
    plt.plot(grid, acf(rvar, nlags=nlags), lw=lw, label='RV')
    plt.show()


def try_integrated_gmm_single():

    riskfree = .0

    mean_v = .2
    kappa_s = .1
    kappa_y = .05
    eta_s = .01**.5 # .1
    eta_y = .001**.5 # .0316

    lmbd = .01
    rho = -.9

    param_true = CentTendParam(riskfree=riskfree, lmbd=lmbd,
                               mean_v=mean_v, kappa_s=kappa_s, kappa_y=kappa_y,
                               eta_s=eta_s, eta_y=eta_y, rho=rho)
    centtend = CentTend(param_true)
    print(param_true.is_valid())

    start = [1, mean_v, mean_v]
    nperiods, interval, ndiscr, nsim = 2000, 1/80, 1, 1
    aggh = 1

    data = centtend.sim_realized(start, interval=interval, ndiscr=ndiscr,
                                 aggh=aggh, nperiods=nperiods,
                                 nsim=nsim, diff=0)
    ret, rvar = data

    data = load_data()
    ret, rvar = data[['logR', 'RV']].values.T
    rvar = (rvar / 100) ** 2 / 252

    plot_realized(ret, rvar)

    nlags = 90
    lw = 2
    grid = range(nlags+1)
    plt.plot(grid, acf(rvar, nlags=nlags), lw=lw, label='RV')
    plt.show()

    instr_data = np.vstack([rvar, rvar**2])

    param_start = param_true
#    param_start.update(param_true.get_theta()/2)
    subset = 'vol'
    theta_start = param_start.get_theta(subset=subset)
    bounds = param_start.get_bounds(subset=subset)
    cons = ({'type': 'ineq', 'fun': lambda x:  x[1] - x[2]},
             {'type': 'ineq', 'fun': lambda x: x[3] - x[4]})

    time_start = time.time()
    res = centtend.integrated_gmm(theta_start, data=data, instrlag=2,
                                  instr_data=instr_data, aggh=aggh,
                                  instr_choice='var', method='TNC',
                                  subset=subset, iter=2, bounds=bounds,
                                  constraints=cons)
    res.print_results()
    print('Elapsed time = %.2f min' % ((time.time() - time_start)/60))


def try_integrated_gmm_real():

    riskfree = .0

    mean_v = .02
    kappa_s = .22
    kappa_y = .05
    eta_s = .36
    eta_y = .05

    lmbd = .01
    rho = -.9

    param_start = CentTendParam(riskfree=riskfree, lmbd=lmbd,
                               mean_v=mean_v, kappa_s=kappa_s, kappa_y=kappa_y,
                               eta_s=eta_s, eta_y=eta_y, rho=rho)
    centtend = CentTend(param_start)
    print(param_start.is_valid())

    aggh = 1

    data = load_data()
    ret, rvar = data

    plot_realized(ret, rvar)

    nlags = 90
    lw = 2
    grid = range(nlags+1)
    plt.plot(grid, acf(rvar, nlags=nlags), lw=lw, label='RV')
    plt.show()

    instr_data = np.vstack([rvar, rvar**2])

    subset = 'vol'
    theta_start = param_start.get_theta(subset=subset)
    bounds = param_start.get_bounds(subset=subset)
    cons = ({'type': 'ineq', 'fun': lambda x: x[1] - x[2]},
             {'type': 'ineq', 'fun': lambda x: x[3] - x[4]})

    time_start = time.time()
    res = centtend.integrated_gmm(theta_start, data=data, instrlag=2,
                                  instr_data=instr_data, aggh=aggh,
                                  instr_choice='var', method='TNC',
                                  subset=subset, iter=2, bounds=bounds,
                                  constraints=cons)
    res.print_results()
    print('Elapsed time = %.2f min' % ((time.time() - time_start)/60))


def try_integrated_gmm():

    riskfree = .0

    mean_v = .2
    kappa_s = .1
    kappa_y = .05
    eta_s = .01**.5 # .1
    eta_y = .001**.5 # .0316

    lmbd = .01
    rho = -.9

    param_true = CentTendParam(riskfree=riskfree, lmbd=lmbd,
                               mean_v=mean_v, kappa_s=kappa_s, kappa_y=kappa_y,
                               eta_s=eta_s, eta_y=eta_y, rho=rho)
    centtend = CentTend(param_true)
    print(param_true.is_valid())

    start = [1, mean_v, mean_v]
    nperiods, interval, ndiscr, nsim = 2000, 1/80, 1, 1
    aggh = 1
    data = centtend.sim_realized(start, interval=interval, ndiscr=ndiscr,
                                 aggh=aggh, nperiods=nperiods,
                                 nsim=nsim, diff=0)
    ret, rvar = data
    plot_realized(ret, rvar)

    instr_data = np.vstack([rvar, rvar**2])

    param_start = param_true
    param_start.update(param_true.get_theta()/2)
    subset = 'vol'
    theta_start = param_start.get_theta(subset=subset)
    bounds = param_start.get_bounds(subset=subset)

    tasks = itertools.product(np.arange(1, 4), ['L-BFGS-B', 'TNC', 'SLSQP'])
    for lag, method in tasks:
        time_start = time.time()
        res = centtend.integrated_gmm(theta_start, data=data, instrlag=lag,
                                    instr_data=instr_data, aggh=aggh,
                                    instr_choice='var', method=method,
                                    subset='vol', bounds=bounds, iter=3)
        res.print_results()
        print(lag, method)
        print('Elapsed time = %.2f min' % ((time.time() - time_start)/60))


if __name__ == '__main__':

    np.set_printoptions(precision=4, suppress=True)
    sns.set_context('notebook')
#    try_simulation()
#    try_marginal()
#    try_sim_realized()
#    try_integrated_gmm_single()
    try_integrated_gmm_real()
#    try_integrated_gmm()
#    check_moments()
