#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Generic Model

"""
from __future__ import print_function, division

#import numpy as np

from diffusions.cir import CIR, CIRparam
from diffusions.helper_functions import plot_trajectories, plot_final_distr


def try_simulation():
    mean, kappa, eta = .5, .1, .2
    theta_true = CIRparam(mean, kappa, eta)
    # 2 * kappa * mean - eta**2 > 0
    print(theta_true.is_valid())
    cir = CIR(theta_true)

    x0, nperiods, interval, ndiscr, nsim = 1, 500, .5, 10, 3
    npoints = int(nperiods / interval)
    paths = cir.simulate(x0, interval, ndiscr, npoints, nsim)
    data = paths[:, 0, 0]

    plot_trajectories(data, interval)


if __name__ == '__main__':

    try_simulation()
