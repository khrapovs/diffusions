#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Generic Model

"""
from __future__ import print_function, division

#import numpy as np

from diffusions import CIR, CIRparam
from diffusions import plot_trajectories, plot_final_distr


def try_simulation():
    mean, kappa, eta = .5, .1, .2
    theta_true = CIRparam(mean, kappa, eta)
    # 2 * kappa * mean - eta**2 > 0
    print(theta_true.is_valid())
    cir = CIR(theta_true)

    x0, nperiods, interval, ndiscr, nsim = mean, 500, .5, 10, 3
    npoints = int(nperiods / interval)
    paths = cir.simulate(x0, interval, ndiscr, npoints, nsim)
    data = paths[:, 0, 0]

    plot_trajectories(data, interval)


def try_marginal():
    mean, kappa, eta = .5, .1, .2
    theta_true = CIRparam(mean, kappa, eta)
    # 2 * kappa * mean - eta**2 > 0
    print(theta_true.is_valid())
    cir = CIR(theta_true)

    x0, nperiods, interval, ndiscr, nsim = mean, 500, .5, 10, 20
    nobs = int(nperiods / interval)
    paths = cir.simulate(x0, interval, ndiscr, nobs, nsim)
    data = paths[:, :, 0]

    plot_final_distr(data)


if __name__ == '__main__':

    try_simulation()
    try_marginal()
