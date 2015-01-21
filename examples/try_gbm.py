#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Generic Model

"""
from __future__ import print_function, division

#import numpy as np

from diffusions.gbm import GBM, GBMparam
from diffusions.generic_model import plot_trajectories, plot_final_distr


if __name__ == '__main__':
    mean, sigma = 1.5, .2
    theta_true = GBMparam(mean, sigma)
    gbm = GBM(theta_true)

    x0, nperiods, interval, ndiscr, nsim = 1, 500, .5, 10, 3
    npoints = int(nperiods / interval)
    gbm.simulate(x0, interval, ndiscr, npoints, nsim)
    data = gbm.paths[:, 0]

    plot_trajectories(data, interval)

    #plot_final_distr(data/interval)

    mean, sigma = 2.5, .4
    theta_start = GBMparam(mean, sigma)
    res = gbm.gmmest(theta_start, data=data, instrlag=2)
    res.print_results()
