#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Try Heston model

"""
from __future__ import print_function, division

import time
import itertools

import numpy as np
import matplotlib.pylab as plt
import seaborn as sns

from statsmodels.tsa.stattools import acf

from diffusions import Heston, HestonParam
from diffusions.helper_functions import (plot_trajectories, plot_final_distr,
                                         plot_realized, take_time)
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
    param_true = HestonParam(riskfree=riskfree, lmbd=lmbd, mean_v=mean_v,
                             kappa=kappa, eta=eta, rho=rho)
    heston = Heston(param_true)
    print(param_true.is_valid())

    start = [1, mean_v]
    nperiods, nsub, ndiscr, nsim = 500, 10, 10, 3
    nobs = nperiods * nsub
    paths = heston.simulate(start, nsub=nsub, ndiscr=ndiscr,
                            nobs=nobs, nsim=nsim, diff=0)

    returns = paths[:, 0, 0]
    volatility = paths[:, 0, 1]
    plot_trajectories(returns, nsub, 'returns')
    plot_trajectories(volatility, nsub, 'volatility')


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

    nperiods, nsub, ndiscr, nsim = 100, 10, 10, 3
    start = [1, mean_v]
    nobs = nperiods * nsub
    paths = heston.simulate(start, nsub=nsub, ndiscr=ndiscr,
                            nobs=nobs, nsim=nsim, diff=0)

    param_true = HestonParam(riskfree=riskfree, lmbd=lmbd, lmbd_v=lmbd_v,
                             mean_v=mean_v, kappa=kappa,
                             eta=eta, rho=rho, measure='Q')
    heston.update_theta(param_true)
    start_q = [1, param_true.mean_v]
    paths_q = heston.simulate(start_q, nsub=nsub, ndiscr=ndiscr,
                              nobs=nobs, nsim=nsim, diff=0, new_innov=False)

    returns = paths[:, 0, 0]
    volatility = paths[:, 0, 1]
    returns_q = paths_q[:, 0, 0]
    volatility_q = paths_q[:, 0, 1]
    plot_trajectories([returns, returns_q], nsub, ['returns', 'returns_q'])
    plot_trajectories([volatility, volatility_q], nsub,
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
    param_true = HestonParam(riskfree=riskfree, lmbd=lmbd, mean_v=mean_v,
                             kappa=kappa, eta=eta, rho=rho)
    heston = Heston(param_true)

    start = [1, mean_v]
    nperiods, nsub, ndiscr, nsim = 500, 10, 10, 200
    nobs = nperiods * nsub
    paths = heston.simulate(start, nsub=nsub, ndiscr=ndiscr,
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
    param_true = HestonParam(riskfree=riskfree, lmbd=lmbd, mean_v=mean_v,
                             kappa=kappa, eta=eta, rho=rho)
    heston = Heston(param_true)

    # start = [1, mean_v]
    nperiods, nsub, ndiscr, nsim = 500, 80, 1, 1
    aggh = 10

    returns, rvar = heston.sim_realized(nsub=nsub, ndiscr=ndiscr, aggh=aggh,
                                        nperiods=nperiods, nsim=nsim, diff=0)

    plot_realized(returns, rvar)


def try_sim_realized_pq():
    """Simulate realized data from Heston model under P and Q measures.

    """
    riskfree = .0
    mean_v = .5
    kappa = .04
    eta = .15
    rho = -.9
    lmbd = 1.5
    lmbd_v = .2
    # 2 * kappa * mean_v - eta**2 > 0
    param_true = HestonParam(riskfree=riskfree, lmbd=lmbd, mean_v=mean_v,
                             kappa=kappa, eta=eta, rho=rho, lmbd_v=lmbd_v)
    heston = Heston(param_true)

    nperiods, nsub, ndiscr, nsim = 500, 100, 1, 1
    aggh = [1, 2]

    print(heston.param)

    data = heston.sim_realized_pq(nsub=nsub, ndiscr=ndiscr, aggh=aggh,
                                  nperiods=nperiods, nsim=nsim, diff=0)
    (ret_p, rvar_p), (ret_q, rvar_q) = data
    print(heston.param)
    nobs = np.min([ret_p.size, ret_q.size])

    plot_realized([ret_p[-nobs:], ret_q[-nobs:]],
                  [rvar_p[-nobs:], rvar_q[-nobs:]], suffix=['P', 'Q'])


def try_integrated_gmm_single():
    """Simulate realized data from Heston model. Estimate parameters.

    """
    riskfree = .0

    mean_v = .2
    kappa = .06
    eta = .15
    lmbd = .5
    rho = -.5
    # 2 * kappa * mean_v - eta**2 > 0
    param_true = HestonParam(riskfree=riskfree, lmbd=lmbd, mean_v=mean_v,
                             kappa=kappa, eta=eta, rho=rho)
    heston = Heston(param_true)
    print(param_true)

    nperiods, nsub, ndiscr, nsim = 2000, 80, 10, 1
    aggh = 1
    data = heston.sim_realized(nsub=nsub, ndiscr=ndiscr,
                               aggh=aggh, nperiods=nperiods, nsim=nsim, diff=0)
    ret, rvar = data
    plot_realized(ret, rvar)
    nlags, lw = 90, 2
    grid = range(nlags+1)
    plt.plot(grid, acf(rvar, nlags=nlags), lw=lw, label='RV')
    plt.show()

    instr_data = np.vstack([rvar, rvar**2])

    subset = 'vol'
    measure = 'P'
    time_start = time.time()
    res = heston.integrated_gmm(param_true, data=data, instrlag=3,
                                instr_data=instr_data, aggh=aggh,
                                instr_choice='var', method='SLSQP',
                                subset=subset, measure=measure, iter=3)
    print(res)
    print('Elapsed time = %.2f min' % ((time.time() - time_start)/60))


def try_integrated_gmm_single_rn():
    """Simulate realized data from risk-neutral Heston model.
    Estimate parameters.

    """
    riskfree = .0
    mean_v = .5
    kappa = .04
    eta = .15
    rho = -.9
    lmbd = 1.5
    lmbd_v = .1
    # 2 * kappa * mean_v - eta**2 > 0
    param_true = HestonParam(riskfree=riskfree, lmbd=lmbd, mean_v=mean_v,
                             kappa=kappa, eta=eta, rho=rho, lmbd_v=lmbd_v)
    print('P parameters:\n', param_true)
    heston = Heston(param_true)

    aggh = [1, 1]
    nperiods, nsub, ndiscr, nsim = 2000, 100, 1, 1

    data = heston.sim_realized_pq(nsub=nsub, ndiscr=ndiscr, aggh=aggh,
                                  nperiods=nperiods, nsim=nsim, diff=0)
    print('Q parameters:\n', param_true)
    data_p, data_q = data
    ret_p, rvar_p = data_p
    ret_q, rvar_q = data_q
    nobs = np.min([ret_p.size, ret_q.size])
    plot_realized([ret_p[-nobs:], ret_q[-nobs:]],
                  [rvar_p[-nobs:], rvar_q[-nobs:]], suffix=['P', 'Q'])

    instr_data = np.vstack([rvar_p, rvar_p**2])

    subset = 'vol'
    measure = 'P'

    res = heston.integrated_gmm(param_true, data=data_p, instrlag=2,
                                instr_data=instr_data, aggh=aggh[0],
                                instr_choice='var', method='TNC',
                                subset=subset, iter=3,
                                measure=measure)

    print(res)

    time_start = time.time()
    res = heston.integrated_gmm(param_true, data=data_q, instrlag=2,
                                instr_data=instr_data, aggh=aggh[1],
                                instr_choice='var', method='TNC',
                                subset=subset, iter=3,
                                measure=measure)
    print(res)
    print('Elapsed time = %.2f min' % ((time.time() - time_start)/60))


def try_integrated_gmm_joint():
    """Simulate realized data from risk-neutral Heston model.
    Estimate parameters.

    """
    riskfree = .0
    mean_v = .5
    kappa = .04
    eta = .15
    rho = -.9
    lmbd = 1.5
    lmbd_v = .1
    # 2 * kappa * mean_v - eta**2 > 0
    param_true = HestonParam(riskfree=riskfree, lmbd=lmbd, mean_v=mean_v,
                             kappa=kappa, eta=eta, rho=rho, lmbd_v=lmbd_v)
    print('P parameters:\n', param_true)
    heston = Heston(param_true)

    aggh = [1, 1]
    nperiods, nsub, ndiscr, nsim = 1000, 100, 1, 1

    data = heston.sim_realized_pq(nsub=nsub, ndiscr=ndiscr, aggh=aggh,
                                  nperiods=nperiods, nsim=nsim, diff=0)
    print('Q parameters:\n', param_true)
    data_p, data_q = data
    ret_p, rvar_p = data_p
    ret_q, rvar_q = data_q
    nobs = np.min([ret_p.size, ret_q.size])
    plot_realized([ret_p[-nobs:], ret_q[-nobs:]],
                  [rvar_p[-nobs:], rvar_q[-nobs:]], suffix=['P', 'Q'])

    instr_data = np.vstack([rvar_p, rvar_p**2])

    subset = 'vol'
    measure = 'PQ'

    time_start = time.time()
    res = heston.integrated_gmm(param_true, data=data, instrlag=2,
                                instr_data=instr_data, aggh=aggh,
                                instr_choice='var', method='TNC',
                                subset=subset, iter=3,
                                measure=measure)
    print(res)
    print('Elapsed time = %.2f min' % ((time.time() - time_start)/60))

    return res


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
    param_start = HestonParam(riskfree=riskfree, lmbd=lmbd, mean_v=mean_v,
                              kappa=kappa, eta=eta, rho=rho)
    heston = Heston(param_start)

    aggh = 1

    data = load_data()
    ret, rvar = data
    plot_realized(ret, rvar)

    instr_data = np.vstack([rvar, rvar**2])

    subset = 'vol'

    time_start = time.time()
    res = heston.integrated_gmm(param_start, data=data, instrlag=2,
                                instr_data=instr_data, aggh=aggh,
                                instr_choice='var', method='TNC',
                                subset=subset, iter=3)
    print(res)
    print('Elapsed time = %.2f min' % ((time.time() - time_start)/60))


def try_integrated_gmm_opt_methods():
    """Simulate realized data from Heston model. Estimate parameters.
    Check various optimization methods.

    """
    riskfree = .0

    mean_v = .5
    kappa = .1
    eta = .15
    lmbd = .3
    rho = -.5
    # 2 * kappa * mean_v - eta**2 > 0
    param_true = HestonParam(riskfree=riskfree, lmbd=lmbd,
                             mean_v=mean_v, kappa=kappa,
                             eta=eta, rho=rho)
    heston = Heston(param_true)

    start, nperiods, nsub, ndiscr, nsim = [1, mean_v], 2000, 80, 1, 1
    aggh = 10
    data = heston.sim_realized(start, nsub=nsub, ndiscr=ndiscr,
                               aggh=aggh, nperiods=nperiods, nsim=nsim, diff=0)
    ret, rvar = data
    plot_realized(ret, rvar)

    instr_data = np.vstack([rvar, rvar**2])

    param_start = param_true
    param_start.update(param_true.get_theta()/2)

    tasks = itertools.product(np.arange(1, 4), ['L-BFGS-B', 'TNC', 'SLSQP'])
    for lag, method in tasks:
        time_start = time.time()
        res = heston.integrated_gmm(param_start, data=data, instrlag=lag,
                                    instr_data=instr_data, aggh=aggh,
                                    instr_choice='var', method=method,
                                    subset='vol', iter=3)
        print(res)
        print(lag, method)
        print('Elapsed time = %.2f min' % ((time.time() - time_start)/60))


if __name__ == '__main__':

    np.set_printoptions(precision=4, suppress=True)
    sns.set_context('notebook')

#    with take_time('Simulation'):
#        try_simulation()
#    with take_time('Simulation PQ'):
#        try_simulation_pq()
#    with take_time('Marginal density'):
#        try_marginal()
#    with take_time('Simulate realized'):
#        try_sim_realized()
#    with take_time('Simulate realized PQ'):
#        try_sim_realized_pq()
#    with take_time('Integrated GMM'):
#        try_integrated_gmm_single()
#    with take_time('Integrated GMM under Q'):
#        try_integrated_gmm_single_rn()
#    with take_time('Integrated GMM under P and Q'):
#        res = try_integrated_gmm_joint()
#    with take_time('Integrated GMM with real data'):
#        try_integrated_gmm_real()
    with take_time('Integrated GMM with real data'):
        try_integrated_gmm_opt_methods()
