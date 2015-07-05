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
from load_real_data import load_data


def try_simulation():
    """Try simulating and plotting Central Tendency model.

    """
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
    """Simulate and plot marginal distribution of the data
    in Central Tendency model.

    """
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
    """Simulate realized data from Central Tendency model and plot it.

    """
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


def try_sim_realized_pq():
    """Simulate realized data from Central Tendency model
    under P and Q measures.

    """
    riskfree = .01
    lmbd = 1.01
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
    nperiods, interval, ndiscr, nsim = 500, 1/80, 1, 1
    aggh = 1

    data = centtend.sim_realized(start, interval=interval, ndiscr=ndiscr,
                                 aggh=aggh, nperiods=nperiods, nsim=nsim,
                                 diff=0)

    returns, rvar = data

    lmbd = 0
    lmbd_s, lmbd_y = .5, .5
    assert lmbd_s < kappa_s / eta_s, 'kappa_v is not positive!'
    assert lmbd_y < kappa_y / eta_y, 'kappa_y is not positive!'
    kappa_sq = kappa_s - lmbd_s * eta_s
    kappa_yq = kappa_y - lmbd_y * eta_y
    scale = kappa_s / kappa_sq
    mean_vq = mean_v * kappa_y / kappa_yq * scale
    eta_yq = eta_y * scale**.5

    param_true_new = CentTendParam(riskfree=riskfree, lmbd=lmbd,
                                   mean_v=mean_vq, kappa_s=kappa_sq,
                                   kappa_y=kappa_yq, eta_s=eta_s, eta_y=eta_yq,
                                   rho=rho)
    centtend.update_theta(param_true_new)
    start_q = [1, mean_vq, mean_vq]
    aggh = 10
    data_new = centtend.sim_realized(start_q, interval=interval, ndiscr=ndiscr,
                                   aggh=aggh, nperiods=nperiods, nsim=nsim,
                                   diff=0, new_innov=False)
    returns_new, rvar_new = data_new
    plot_realized([returns[aggh-1:], returns_new],
                  [rvar[aggh-1:], rvar_new],
                  suffix=['P', 'Q'])


def try_integrated_gmm_single():
    """Simulate realized data from Central Tendency model. Estimate parameters.

    """
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
    """Estimate Central Tendency model parameters with real data.

    """
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
    """Simulate realized data from Central Tendency model. Estimate parameters.
    Check various optimization methods.

    """
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
    try_sim_realized_pq()
#    try_integrated_gmm_single()
#    try_integrated_gmm_real()
#    try_integrated_gmm()
#    check_moments()
