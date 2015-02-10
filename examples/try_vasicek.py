#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Generic Model

"""
from __future__ import print_function, division

#import numpy as np

from diffusions.vasicek import Vasicek, VasicekParam
from diffusions.helper_functions import plot_trajectories, plot_final_distr


def try_simulation():
    mean, kappa, eta = .5, .1, .2
    theta_true = VasicekParam(mean, kappa, eta)
    vasicek = Vasicek(theta_true)

    x0, nperiods, interval, ndiscr, nsim = 1, 500, .5, 10, 3
    npoints = int(nperiods / interval)
    paths = vasicek.simulate(x0, interval, ndiscr, npoints, nsim)
    data = paths[:, 0, 0]

    plot_trajectories(data, interval)


if __name__ == '__main__':

    try_simulation()
