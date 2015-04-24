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


def try_simulation():
    riskfree = .0
    lmbd = .0
    mean_v = .5
    kappa = .1
    eta = .02**.5
    rho = -.9
    # 2 * self.kappa * self.mean_v - self.eta**2 > 0
    theta_true = HestonParam(riskfree=riskfree, lmbd=lmbd,
                             mean_v=mean_v, kappa=kappa,
                             eta=eta, rho=rho)
    heston = Heston(theta_true)
    print(theta_true.is_valid())

    start, nperiods, interval, ndiscr, nsim = [1, mean_v], 500, .1, 10, 3
    npoints = int(nperiods / interval)
    paths = heston.simulate(start, interval, ndiscr, npoints, nsim, diff=0)

    returns = paths[:, 0, 0]
    volatility = paths[:, 0, 1]
    plot_trajectories(returns, interval, 'returns')
    plot_trajectories(volatility, interval, 'volatility')


def try_marginal():
    riskfree = .0
    lmbd = .0
    mean_v = .5
    kappa = .1
    eta = .02**.5
    rho = -.9
    # 2 * self.kappa * self.mean_v - self.eta**2 > 0
    theta_true = HestonParam(riskfree=riskfree, lmbd=lmbd,
                             mean_v=mean_v, kappa=kappa,
                             eta=eta, rho=rho)
    heston = Heston(theta_true)

    start, nperiods, interval, ndiscr, nsim = [1, mean_v], 500, .1, 10, 20
    npoints = int(nperiods / interval)
    paths = heston.simulate(start, interval, ndiscr, npoints, nsim, diff=0)

    returns = paths[:, :, 0]
    volatility = paths[:, :, 1]

    plot_final_distr(returns)
    plot_final_distr(volatility)


def try_sim_realized():
    riskfree = .0
    lmbd = .0
    mean_v = .5
    kappa = .1
    eta = .02**.5
    rho = -.9
    # 2 * self.kappa * self.mean_v - self.eta**2 > 0
    theta_true = HestonParam(riskfree=riskfree, lmbd=lmbd,
                             mean_v=mean_v, kappa=kappa,
                             eta=eta, rho=rho)
    heston = Heston(theta_true)

    start, nperiods, interval, ndiscr, nsim = [1, mean_v], 500, 1/80, 1, 1
    aggh = 10

    returns, rvar = heston.sim_realized(start, interval=interval,
                                        ndiscr=ndiscr, aggh=aggh,
                                        nperiods=nperiods, nsim=nsim, diff=0)

    plot_realized(returns, rvar)


def try_integrated_gmm_single():
    riskfree = .0

    mean_v = .5
    kappa = .1
    eta = .02**.5 # 0.1414
    lmbd = .3
    rho = -.5
    # 2 * self.kappa * self.mean_v - self.eta**2 > 0
    theta_true = HestonParam(riskfree=riskfree, lmbd=lmbd,
                             mean_v=mean_v, kappa=kappa,
                             eta=eta, rho=rho)
    heston = Heston(theta_true)

    start, nperiods, interval, ndiscr, nsim = [1, mean_v], 2000, 1/80, 1, 1
    aggh = 1
    data = heston.sim_realized(start, interval=interval, ndiscr=ndiscr,
                               aggh=aggh, nperiods=nperiods, nsim=nsim, diff=0)
    ret, rvar = data
    plot_realized(ret, rvar)

    instr_data = np.vstack([rvar, rvar**2])

    theta_start = theta_true
    theta_start.update(theta_true.get_theta()/2)

    time_start = time.time()
    subset = 'vol'
    res = heston.integrated_gmm(theta_start.get_theta(subset=subset),
                                data=data, instrlag=2,
                                instr_data=instr_data, aggh=aggh,
                                instr_choice='var', method='TNC',
                                use_jacob=True, exact_jacob=False,
                                subset=subset, iter=3,
                                bounds=theta_start.get_bounds(subset=subset))
    res.print_results()
    print('Elapsed time = %.2f min' % ((time.time() - time_start)/60))


def try_integrated_gmm():
    riskfree = .0

    mean_v = .5
    kappa = .1
    eta = .02**.5 # 0.1414
    lmbd = .3
    rho = -.5
    # 2 * self.kappa * self.mean_v - self.eta**2 > 0
    theta_true = HestonParam(riskfree=riskfree, lmbd=lmbd,
                             mean_v=mean_v, kappa=kappa,
                             eta=eta, rho=rho)
    heston = Heston(theta_true)

    start, nperiods, interval, ndiscr, nsim = [1, mean_v], 2000, 1/80, 1, 1
    aggh = 10
    data = heston.sim_realized(start, interval=interval, ndiscr=ndiscr,
                               aggh=aggh, nperiods=nperiods, nsim=nsim, diff=0)
    ret, rvar = data
    plot_realized(ret, rvar)

    instr_data = np.vstack([rvar, rvar**2])

    theta_start = theta_true
    theta_start.update(theta_true.get_theta()/2)

    tasks = itertools.product(np.arange(1, 4), ['L-BFGS-B', 'TNC', 'SLSQP'])
    for lag, method in tasks:
        time_start = time.time()
        res = heston.integrated_gmm(theta_start, data=data, instrlag=lag,
                                    instr_data=instr_data, aggh=aggh,
                                    instr_choice='var', method=method,
                                    use_jacob=True, exact_jacob=False,
                                    subset='vol',
                                    bounds=theta_start.get_bounds(), iter=3)
        res.print_results()
        print(lag, method)
        print('Elapsed time = %.2f min' % ((time.time() - time_start)/60))


if __name__ == '__main__':

    np.set_printoptions(precision=4, suppress=True)
    sns.set_context('notebook')
#    try_simulation()
#    try_marginal()
#    try_sim_realized()
    try_integrated_gmm_single()
#    try_integrated_gmm()
