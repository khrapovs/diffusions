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
from diffusions import plot_trajectories, plot_final_distr, plot_realized
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
    # 2 * self.kappa * self.mean_v - self.eta**2 > 0
    param_true = HestonParam(riskfree=riskfree, lmbd=lmbd,
                             mean_v=mean_v, kappa=kappa,
                             eta=eta, rho=rho)
    heston = Heston(param_true)
    print(param_true.is_valid())

    start, nperiods, interval, ndiscr, nsim = [1, mean_v], 500, .1, 10, 3
    npoints = int(nperiods / interval)
    paths = heston.simulate(start, interval, ndiscr, npoints, nsim, diff=0)

    returns = paths[:, 0, 0]
    volatility = paths[:, 0, 1]
    plot_trajectories(returns, interval, 'returns')
    plot_trajectories(volatility, interval, 'volatility')


def try_marginal():
    """Simulate and plot marginal distribution of the data in Heston model.

    """
    riskfree = .0
    lmbd = .0
    mean_v = .5
    kappa = .1
    eta = .02**.5
    rho = -.9
    # 2 * self.kappa * self.mean_v - self.eta**2 > 0
    param_true = HestonParam(riskfree=riskfree, lmbd=lmbd,
                             mean_v=mean_v, kappa=kappa,
                             eta=eta, rho=rho)
    heston = Heston(param_true)

    start, nperiods, interval, ndiscr, nsim = [1, mean_v], 500, .1, 10, 20
    npoints = int(nperiods / interval)
    paths = heston.simulate(start, interval, ndiscr, npoints, nsim, diff=0)

    returns = paths[:, :, 0]
    volatility = paths[:, :, 1]

    plot_final_distr(returns)
    plot_final_distr(volatility)


def try_sim_realized():
    """Simulate realized data from Heston model and plot it.

    """
    riskfree = .0
    lmbd = .0
    mean_v = .5
    kappa = .1
    eta = .02**.5
    rho = -.9
    # 2 * self.kappa * self.mean_v - self.eta**2 > 0
    param_true = HestonParam(riskfree=riskfree, lmbd=lmbd,
                             mean_v=mean_v, kappa=kappa,
                             eta=eta, rho=rho)
    heston = Heston(param_true)

    start, nperiods, interval, ndiscr, nsim = [1, mean_v], 500, 1/80, 1, 1
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
    mean_v = .5
    kappa = .1
    eta = .02**.5
    rho = -.9
    # 2 * self.kappa * self.mean_v - self.eta**2 > 0
    param_true = HestonParam(riskfree=riskfree, lmbd=lmbd, mean_v=mean_v,
                             kappa=kappa, eta=eta, rho=rho)
    heston = Heston(param_true)

    start, nperiods, interval, ndiscr, nsim = [1, mean_v], 100, 1/80, 1, 1
    aggh = 1

    data = heston.sim_realized(start, interval=interval, ndiscr=ndiscr,
                               aggh=aggh, nperiods=nperiods, nsim=nsim, diff=0)
    returns, rvar = data

    lmbd = 0
    lmbd_v = .5
    kappa_q = kappa - lmbd_v * eta
    mean_vq = mean_v * kappa / kappa_q
    param_true_new = HestonParam(riskfree=riskfree, lmbd=lmbd, mean_v=mean_vq,
                                 kappa=kappa_q, eta=eta, rho=rho)
    heston.update_theta(param_true_new)
    start_q = [1, mean_vq]
    data_new = heston.sim_realized(start_q, interval=interval, ndiscr=ndiscr,
                                   aggh=aggh, nperiods=nperiods, nsim=nsim,
                                   diff=0, new_innov=False)
    returns_new, rvar_new = data_new
    plot_realized([returns, returns_new], [rvar, rvar_new], suffix=['P', 'Q'])


def try_integrated_gmm_single():
    """Simulate realized data from Heston model. Estimate parameters.

    """
    riskfree = .0

    mean_v = .5
    kappa = .1
    eta = .02**.5 # 0.1414
    lmbd = .3
    rho = -.5
    # 2 * self.kappa * self.mean_v - self.eta**2 > 0
    param_true = HestonParam(riskfree=riskfree, lmbd=lmbd,
                             mean_v=mean_v, kappa=kappa,
                             eta=eta, rho=rho)
    heston = Heston(param_true)

    start, nperiods, interval, ndiscr, nsim = [1, mean_v], 2000, 1/80, 1, 1
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


def try_integrated_gmm_real():
    """Estimate Heston model parameters with real data.

    """
    riskfree = .0

    mean_v = .2
    kappa = .22
    eta = .12

    lmbd = .3
    rho = -.5
    # 2 * self.kappa * self.mean_v - self.eta**2 > 0
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
    # 2 * self.kappa * self.mean_v - self.eta**2 > 0
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
#    try_marginal()
#    try_sim_realized()
    try_sim_realized_pq()
#    try_integrated_gmm_single()
#    try_integrated_gmm_real()
#    try_integrated_gmm()
