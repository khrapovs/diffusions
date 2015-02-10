#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Try Heston model

"""
from __future__ import print_function, division

#import numpy as np

from diffusions.heston import Heston, HestonParam
from diffusions.helper_functions import plot_trajectories, plot_final_distr


def try_simulation():
    mean_r = .0
    mean_v = .5
    kappa = .1
    eta = .02**.5
    rho = -.9
    # 2 * self.kappa * self.mean_v - self.eta**2 > 0
    theta_true = HestonParam(mean_r=mean_r, mean_v=mean_v, kappa=kappa,
                             eta=eta, rho=rho)
    heston = Heston(theta_true)
    print(theta_true.is_valid())

    start, nperiods, interval, ndiscr, nsim = [1, mean_v], 500, .1, 10, 3
    npoints = int(nperiods / interval)
    paths = heston.simulate(start, interval, ndiscr, npoints, nsim)

    price = paths[:, 0, 0]
    returns = price[1:] - price[:-1]
    volatility = paths[1:, 0, 1]
    plot_trajectories(returns, interval)
    plot_trajectories(volatility, interval)

    return paths


if __name__ == '__main__':

    paths = try_simulation()
    price = paths[:, 0, 0]
    returns = price[1:] - price[:-1]
    volatility = paths[1:, 0, 1]
