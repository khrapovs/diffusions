#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Try Heston model

"""
from __future__ import print_function, division

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
    plot_trajectories(returns, interval)
    plot_trajectories(volatility, interval)


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
    returns, rvar = heston.sim_realized(start, interval, ndiscr,
                                        nperiods, nsim, diff=0)

    plot_realized(returns, rvar)


def try_integrated_gmm():
    riskfree = .0

    lmbd = .3
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
    data = heston.sim_realized(start, interval, ndiscr,
                               nperiods, nsim, diff=0)

    theta_start = theta_true
    theta_start.update(theta_true.get_theta()/2)
    res = heston.integrated_gmm(theta_start, data=data, instrlag=1,
                                instr_choice='var', method='L-BFGS-B',
                                use_jacob=False,
                                bounds=theta_start.get_bounds())
    res.print_results()

    res = heston.integrated_gmm(theta_start, data=data, instrlag=1,
                                instr_choice='var', method='L-BFGS-B',
                                use_jacob=True,
                                bounds=theta_start.get_bounds())
    res.print_results()


if __name__ == '__main__':

    np.set_printoptions(precision=4, suppress=True)
    sns.set_context('notebook')
#    try_simulation()
#    try_marginal()
#    try_sim_realized()
    try_integrated_gmm()
