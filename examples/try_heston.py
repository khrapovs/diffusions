#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Try Heston model

"""
from __future__ import print_function, division

import time
import itertools

import numpy as np
import seaborn as sns

from diffusions import Heston, HestonParam
from diffusions.helper_functions import (plot_trajectories, plot_final_distr,
                                         plot_realized)
from load_real_data import load_data


def try_simulation():
    """Try simulating and plotting Heston model.

    """
    riskfree = .0
    lmbd = .0
    mean_v = .5
    kappa = .1
    eta = .02**.5
    rho = -.9
    # 2 * kappa * mean_v - eta**2 > 0
    param_true = HestonParam(riskfree=riskfree, lmbd=lmbd,
                             mean_v=mean_v, kappa=kappa,
                             eta=eta, rho=rho)
    heston = Heston(param_true)
    print(param_true.is_valid())

    start = [1, mean_v]
    nperiods, interval, ndiscr, nsim = 500, .1, 10, 3
    nobs = int(nperiods / interval)
    paths = heston.simulate(start, interval=interval, ndiscr=ndiscr,
                            nobs=nobs, nsim=nsim, diff=0)

    returns = paths[:, 0, 0]
    volatility = paths[:, 0, 1]
    plot_trajectories(returns, interval, 'returns')
    plot_trajectories(volatility, interval, 'volatility')


def try_simulation_pq():
    """Try simulating and plotting Heston model.

    """
    riskfree = .0
    lmbd = .0
    lmbd_v = .5
    mean_v = .5
    kappa = .1
    eta = .15
    rho = -.9
    # 2 * kappa * mean_v - eta**2 > 0
    param_true = HestonParam(riskfree=riskfree, lmbd=lmbd, lmbd_v=lmbd_v,
                             mean_v=mean_v, kappa=kappa,
                             eta=eta, rho=rho, measure='P')
    heston = Heston(param_true)
    print(param_true.is_valid())

    nperiods, interval, ndiscr, nsim = 100, .1, 10, 3
    start = [1, mean_v]
    nobs = int(nperiods / interval)
    paths = heston.simulate(start, interval=interval, ndiscr=ndiscr,
                            nobs=nobs, nsim=nsim, diff=0)

    param_true = HestonParam(riskfree=riskfree, lmbd=lmbd, lmbd_v=lmbd_v,
                             mean_v=mean_v, kappa=kappa,
                             eta=eta, rho=rho, measure='Q')
    heston.update_theta(param_true)
    start_q = [1, param_true.mean_v]
    paths_q = heston.simulate(start_q, interval=interval, ndiscr=ndiscr,
                              nobs=nobs, nsim=nsim, diff=0, new_innov=False)

    returns = paths[:, 0, 0]
    volatility = paths[:, 0, 1]
    returns_q = paths_q[:, 0, 0]
    volatility_q = paths_q[:, 0, 1]
    plot_trajectories([returns, returns_q], interval, ['returns', 'returns_q'])
    plot_trajectories([volatility, volatility_q], interval,
                      ['volatility', 'volatility_q'])


def try_marginal():
    """Simulate and plot marginal distribution of the data in Heston model.

    """
    riskfree = .0
    lmbd = .0
    mean_v = .5
    kappa = .1
    eta = .02**.5
    rho = -.9
    # 2 * kappa * mean_v - eta**2 > 0
    param_true = HestonParam(riskfree=riskfree, lmbd=lmbd,
                             mean_v=mean_v, kappa=kappa,
                             eta=eta, rho=rho)
    heston = Heston(param_true)

    start = [1, mean_v]
    nperiods, interval, ndiscr, nsim = 500, .1, 10, 20
    nobs = int(nperiods / interval)
    paths = heston.simulate(start, interval=interval, ndiscr=ndiscr,
                            nobs=nobs, nsim=nsim, diff=0)

    returns = paths[:, :, 0]
    volatility = paths[:, :, 1]

    plot_final_distr(returns, names='returns')
    plot_final_distr(volatility, names='volatility')


def try_sim_realized():
    """Simulate realized data from Heston model and plot it.

    """
    riskfree = .0
    lmbd = .0
    mean_v = .5
    kappa = .1
    eta = .02**.5
    rho = -.9
    # 2 * kappa * mean_v - eta**2 > 0
    param_true = HestonParam(riskfree=riskfree, lmbd=lmbd,
                             mean_v=mean_v, kappa=kappa,
                             eta=eta, rho=rho)
    heston = Heston(param_true)

    start = [1, mean_v]
    nperiods, interval, ndiscr, nsim = 500, 1/80, 1, 1
    aggh = 10

    returns, rvar = heston.sim_realized(start, interval=interval,
                                        ndiscr=ndiscr, aggh=aggh,
                                        nperiods=nperiods, nsim=nsim, diff=0)

    plot_realized(returns, rvar)


def try_sim_realized_pq():
    """Simulate realized data from Heston model under P and Q measures.

    """
    riskfree = .0
    lmbd = 1.5
    lmbd_v = .1
    mean_v = .5
    kappa = .04
    eta = .15
    rho = -.9
    # 2 * kappa * mean_v - eta**2 > 0
    param_true = HestonParam(riskfree=riskfree, lmbd=lmbd, mean_v=mean_v,
                             kappa=kappa, eta=eta, rho=rho, lmbd_v=lmbd_v)
    heston = Heston(param_true)

    start = [1, mean_v]
    nperiods, interval, ndiscr, nsim = 500, 1/100, 1, 1
    aggh = [1, 2]

    print(heston.param)

    data = heston.sim_realized_pq(start, start, interval=interval,
                                  ndiscr=ndiscr, aggh=aggh, nperiods=nperiods,
                                  nsim=nsim, diff=0)
    (ret_p, rvar_p), (ret_q, rvar_q) = data
    print(heston.param)
    nobs = np.min([ret_p.size, ret_q.size])

    plot_realized([ret_p[-nobs:], ret_q[-nobs:]],
                  [rvar_p[-nobs:], rvar_q[-nobs:]], suffix=['P', 'Q'])


def try_integrated_gmm_single():
    """Simulate realized data from Heston model. Estimate parameters.

    """
    riskfree = .0

    mean_v = .5
    kappa = .1
    eta = .02**.5 # 0.1414
    lmbd = .3
    rho = -.5
    # 2 * kappa * mean_v - eta**2 > 0
    param_true = HestonParam(riskfree=riskfree, lmbd=lmbd,
                             mean_v=mean_v, kappa=kappa,
                             eta=eta, rho=rho)
    heston = Heston(param_true)

    start = [1, mean_v]
    nperiods, interval, ndiscr, nsim = 1000, 1/10, 1, 1
    aggh = 1
    data = heston.sim_realized(start, interval=interval, ndiscr=ndiscr,
                               aggh=aggh, nperiods=nperiods, nsim=nsim, diff=0)
    ret, rvar = data
    plot_realized(ret, rvar)

    instr_data = np.vstack([rvar, rvar**2])

    param_start = param_true
    param_start.update(param_true.get_theta()/2)
    subset = 'vol'
    theta_start = param_start.get_theta(subset=subset)
    bounds = param_start.get_bounds(subset=subset)

    time_start = time.time()
    res = heston.integrated_gmm(theta_start, data=data, instrlag=2,
                                instr_data=instr_data, aggh=aggh,
                                instr_choice='var', method='TNC',
                                subset=subset, iter=3, bounds=bounds)
    res.print_results()
    print('Elapsed time = %.2f min' % ((time.time() - time_start)/60))


def try_integrated_gmm_single_rn():
    """Simulate realized data from risk-neutral Heston model.
    Estimate parameters.

    """
    riskfree = .0
    lmbd = 1.5
    lmbd_v = .1
    mean_v = .5
    kappa = .04
    eta = .15
    rho = -.9
    # 2 * kappa * mean_v - eta**2 > 0
    param_true = HestonParam(riskfree=riskfree, lmbd=lmbd, mean_v=mean_v,
                             kappa=kappa, eta=eta, rho=rho, lmbd_v=lmbd_v)
    print('P parameters:\n', param_true)
    heston = Heston(param_true)

    start = [1, param_true.mean_v]
    aggh = [1, 1]
    nperiods, interval, ndiscr, nsim = 2000, 1/100, 1, 1

    data_p, data_q = heston.sim_realized_pq(start, start, interval=interval,
                                  ndiscr=ndiscr, aggh=aggh, nperiods=nperiods,
                                  nsim=nsim, diff=0)
    print('Q parameters:\n', param_true)
    ret_p, rvar_p = data_p
    ret_q, rvar_q = data_q
    nobs = np.min([ret_p.size, ret_q.size])
    plot_realized([ret_p[-nobs:], ret_q[-nobs:]],
                  [rvar_p[-nobs:], rvar_q[-nobs:]], suffix=['P', 'Q'])

    instr_data = np.vstack([rvar_p, rvar_p**2])

    subset = 'vol'
    theta_start = param_true.get_theta(subset=subset) / 2
    bounds = param_true.get_bounds(subset=subset)

    res = heston.integrated_gmm(theta_start, data=data_p, instrlag=2,
                                instr_data=instr_data, aggh=aggh[0],
                                instr_choice='var', method='SLSQP',
                                subset=subset, iter=3, bounds=bounds)

    res.print_results()

    time_start = time.time()
    res = heston.integrated_gmm(theta_start, data=data_q, instrlag=2,
                                instr_data=instr_data, aggh=aggh[1],
                                instr_choice='var', method='SLSQP',
                                subset=subset, iter=3, bounds=bounds)
    res.print_results()
    print('Elapsed time = %.2f min' % ((time.time() - time_start)/60))


def try_integrated_gmm_real():
    """Estimate Heston model parameters with real data.

    """
    riskfree = .0

    mean_v = .2
    kappa = .22
    eta = .12

    lmbd = .3
    rho = -.5
    # 2 * kappa * mean_v - eta**2 > 0
    param_start = HestonParam(riskfree=riskfree, lmbd=lmbd,
                             mean_v=mean_v, kappa=kappa,
                             eta=eta, rho=rho)
    heston = Heston(param_start)

    aggh = 1

    data = load_data()
    ret, rvar = data
    plot_realized(ret, rvar)

    instr_data = np.vstack([rvar, rvar**2])

    subset = 'vol'
    theta_start = param_start.get_theta(subset=subset)
    bounds = param_start.get_bounds(subset=subset)

    time_start = time.time()
    res = heston.integrated_gmm(theta_start, data=data, instrlag=2,
                                instr_data=instr_data, aggh=aggh,
                                instr_choice='var', method='TNC',
                                subset=subset, iter=3, bounds=bounds)
    res.print_results()
    print('Elapsed time = %.2f min' % ((time.time() - time_start)/60))


def try_integrated_gmm():
    """Simulate realized data from Heston model. Estimate parameters.
    Check various optimization methods.

    """
    riskfree = .0

    mean_v = .5
    kappa = .1
    eta = .02**.5 # 0.1414
    lmbd = .3
    rho = -.5
    # 2 * kappa * mean_v - eta**2 > 0
    param_true = HestonParam(riskfree=riskfree, lmbd=lmbd,
                             mean_v=mean_v, kappa=kappa,
                             eta=eta, rho=rho)
    heston = Heston(param_true)

    start, nperiods, interval, ndiscr, nsim = [1, mean_v], 2000, 1/80, 1, 1
    aggh = 10
    data = heston.sim_realized(start, interval=interval, ndiscr=ndiscr,
                               aggh=aggh, nperiods=nperiods, nsim=nsim, diff=0)
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
        res = heston.integrated_gmm(theta_start, data=data, instrlag=lag,
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
#    try_simulation_pq()
#    try_marginal()
#    try_sim_realized()
#    try_sim_realized_pq()
#    try_integrated_gmm_single()
    try_integrated_gmm_single_rn()
#    try_integrated_gmm_real()
#    try_integrated_gmm()
