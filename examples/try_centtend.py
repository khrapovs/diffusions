#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Try Central Tendency model

"""
from __future__ import print_function, division

import time
import itertools

import numpy as np
import matplotlib.pylab as plt
import seaborn as sns

from statsmodels.tsa.stattools import acf

from diffusions import CentTend, CentTendParam
from diffusions.helper_functions import (plot_trajectories, plot_final_distr,
                                         plot_realized)
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
    print(param_true)
    print(param_true.is_valid())

    start = [1, mean_v, mean_v]
    nperiods, interval, ndiscr, nsim = 500, .1, 10, 3
    nobs = int(nperiods / interval)
    paths = centtend.simulate(start, interval=interval, ndiscr=ndiscr,
                              nobs=nobs, nsim=nsim, diff=0)

    returns = paths[:, 0, 0]
    volatility = paths[:, 0, 1]
    tendency = paths[:, 0, 2]

    plot_trajectories(returns, interval, 'returns')
    names = ['vol', 'ct']
    plot_trajectories([volatility, tendency], interval, names)


def try_simulation_pq():
    """Try simulating and plotting Central Tendency model
    under P and Q measures.

    """
    riskfree = .01
    lmbd = 1.01
    lmbd_s = .5
    lmbd_y = .5
    mean_v = .5
    kappa_s = 1.5
    kappa_y = .05
    eta_s = .02**.5
    eta_y = .001**.5
    rho = -.9

    param_true = CentTendParam(riskfree=riskfree, lmbd=lmbd, lmbd_s=lmbd_s,
                               lmbd_y=lmbd_y, mean_v=mean_v, kappa_s=kappa_s,
                               kappa_y=kappa_y, eta_s=eta_s, eta_y=eta_y,
                               rho=rho)
    centtend = CentTend(param_true)
    print(param_true)
    print(param_true.is_valid())

    start = [1, mean_v, mean_v]
    nperiods, interval, ndiscr, nsim = 500, .1, 100, 3
    nobs = int(nperiods / interval)
    paths = centtend.simulate(start, interval=interval, ndiscr=ndiscr,
                              nobs=nobs, nsim=nsim, diff=0)

    returns = paths[:, 0, 0]
    volatility = paths[:, 0, 1]
    tendency = paths[:, 0, 2]

    param_true_new = CentTendParam(riskfree=riskfree, lmbd=lmbd, lmbd_s=lmbd_s,
                                   lmbd_y=lmbd_y, mean_v=mean_v,
                                   kappa_s=kappa_s, kappa_y=kappa_y,
                                   eta_s=eta_s, eta_y=eta_y,
                                   rho=rho, measure='Q')
    print(param_true_new)
    centtend.update_theta(param_true_new)
    start_q = [1, param_true_new.mean_v, param_true_new.mean_v]

    paths_q = centtend.simulate(start_q, interval=interval, ndiscr=ndiscr,
                                nobs=nobs, nsim=nsim,
                                diff=0, new_innov=False)

    returns_q = paths_q[:, 0, 0]
    volatility_q = paths_q[:, 0, 1]
    tendency_q = paths_q[:, 0, 2] / param_true_new.scale

    plot_trajectories([returns, returns_q], interval, ['returns', 'returns Q'])
    names = ['vol', 'ct', 'vol Q', 'ct Q']
    plot_trajectories([volatility, tendency, volatility_q, tendency_q],
                      interval, names)


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
    print(param_true)
    print(param_true.is_valid())

    start = [1, mean_v, mean_v]
    nperiods, interval, ndiscr, nsim = 500, .1, 10, 100
    nobs = int(nperiods / interval)
    paths = centtend.simulate(start, interval=interval, ndiscr=ndiscr,
                              nobs=nobs, nsim=nsim, diff=0)

    returns = paths[:, :, 0]
    volatility = paths[:, :, 1]
    tendency = paths[:, :, 2]

    plot_final_distr(returns, 'returns')
    names = ['vol', 'ct']
    plot_final_distr([volatility, tendency], names)


def try_sim_realized():
    """Simulate realized data from Central Tendency model and plot it.

    """
    riskfree = .0
    mean_v = .5
    kappa_s = .05
    kappa_y = .02
    eta_s = .1
    eta_y = .01
    rho = -.9
    lmbd = .5

    param_true = CentTendParam(riskfree=riskfree, lmbd=lmbd,
                               mean_v=mean_v, kappa_s=kappa_s, kappa_y=kappa_y,
                               eta_s=eta_s, eta_y=eta_y, rho=rho)
    centtend = CentTend(param_true)
    print(param_true)
    print(param_true.is_valid())

    nperiods, interval, ndiscr, nsim = 2000, 1/80, 10, 1
    aggh = 1

    returns, rvar = centtend.sim_realized(interval=interval,
                                          ndiscr=ndiscr, aggh=aggh,
                                          nperiods=nperiods, nsim=nsim, diff=0)

    plot_realized(returns, rvar)

    nlags, lw = 90, 2
    grid = range(nlags+1)
    plt.plot(grid, acf(rvar, nlags=nlags), lw=lw, label='RV')
    plt.show()


def try_sim_realized_pq():
    """Simulate realized data from Central Tendency model
    under P and Q measures.

    """
    riskfree = .0
    lmbd = 1.01
    lmbd_s, lmbd_y = .5, .5
    mean_v = .2
    kappa_s = .1
    kappa_y = .02
    eta_s = .1
    eta_y = .01
    rho = -.9

    param_true = CentTendParam(riskfree=riskfree, lmbd=lmbd, lmbd_s=lmbd_s,
                               lmbd_y=lmbd_y, mean_v=mean_v, kappa_s=kappa_s,
                               kappa_y=kappa_y, eta_s=eta_s, eta_y=eta_y,
                               rho=rho)
    centtend = CentTend(param_true)
    print(param_true)

    nperiods, interval, ndiscr, nsim = 500, 1/80, 10, 1
    aggh = [1, 1]

    data = centtend.sim_realized_pq(interval=interval, ndiscr=ndiscr,
                                    aggh=aggh, nperiods=nperiods, nsim=nsim,
                                    diff=0)
    print(param_true)
    (ret_p, rvar_p), (ret_q, rvar_q) = data
    nobs = np.min([ret_p.size, ret_q.size])

    plot_realized([ret_p[-nobs:], ret_q[-nobs:]],
                  [rvar_p[-nobs:], rvar_q[-nobs:] / param_true.scale],
                  suffix=['P', 'Q'])


def try_integrated_gmm_single():
    """Simulate realized data from Central Tendency model. Estimate parameters.

    """
    riskfree = .0
    mean_v = .2
    kappa_s = 1.5
    kappa_y = .008
    eta_s = .5
    eta_y = .05
    rho = -.9
    lmbd = .5

    param_true = CentTendParam(riskfree=riskfree, lmbd=lmbd, mean_v=mean_v,
                               kappa_s=kappa_s, kappa_y=kappa_y,
                               eta_s=eta_s, eta_y=eta_y, rho=rho)
    centtend = CentTend(param_true)
    print(param_true)

    nperiods, interval, ndiscr, nsim = 2000, 1/80, 10, 1
    aggh = 1

    data = centtend.sim_realized(interval=interval, ndiscr=ndiscr,
                                 aggh=aggh, nperiods=nperiods,
                                 nsim=nsim, diff=0)
    ret, rvar = data
    plot_realized(ret, rvar)
    nlags, lw = 90, 2
    grid = range(nlags+1)
    plt.plot(grid, acf(rvar, nlags=nlags), lw=lw, label='RV')
    plt.show()

    instr_data = np.vstack([rvar])

    subset = 'vol'
    measure = 'P'

#    theta = param_true.get_theta(subset=subset, measure=measure)
#    mom, dmom = centtend.integrated_mom(theta/10, data=data,
#                                        instr_data=instr_data,
#                                        instr_choice='var', aggh=1,
#                                        subset=subset, instrlag=1,
#                                        measure=measure)
#
#    fig, axes = plt.subplots(nrows=mom.shape[1], ncols=1, sharex=True,
#                             figsize=(10, 2*mom.shape[1]))
#    for momf, ax in zip(mom.T, axes):
#        ax.plot(momf)
#        ax.axhline(momf.mean(), c='red')
#    plt.show()
#    print(mom.mean(0) / mom.std(0))

    time_start = time.time()
    res = centtend.integrated_gmm(param_true, data=data, instrlag=3,
                                  instr_data=instr_data, aggh=aggh,
                                  instr_choice='var', method='SLSQP',
                                  subset=subset, measure=measure, iter=3)
    print(res)
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
    print(param_start)
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

    time_start = time.time()
    res = centtend.integrated_gmm(theta_start, data=data, instrlag=2,
                                  instr_data=instr_data, aggh=aggh,
                                  instr_choice='var', method='TNC',
                                  subset=subset, iter=2)
    print(res)
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
    print(param_true)
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

    tasks = itertools.product(np.arange(1, 4), ['L-BFGS-B', 'TNC', 'SLSQP'])
    for lag, method in tasks:
        time_start = time.time()
        res = centtend.integrated_gmm(theta_start, data=data, instrlag=lag,
                                    instr_data=instr_data, aggh=aggh,
                                    instr_choice='var', method=method,
                                    subset='vol', iter=3)
        print(res)
        print(lag, method)
        print('Elapsed time = %.2f min' % ((time.time() - time_start)/60))


if __name__ == '__main__':

    np.set_printoptions(precision=4, suppress=True)
    sns.set_context('notebook')

#    try_simulation()
#    try_simulation_pq()
#    try_marginal()
#    try_sim_realized()
#    try_sim_realized_pq()
    try_integrated_gmm_single()
#    try_integrated_gmm_real()
#    try_integrated_gmm()
#    check_moments()
