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

    x0, nperiods, interval, ndiscr, nsim = 1, 500, .1, 50, 100
    npoints = int(nperiods / interval)
    gbm.simulate(x0, interval, ndiscr, npoints, nsim)
    data = gbm.paths

    plot_trajectories(data[:, 3], interval)

    plot_final_distr(data/interval)

    res = gbm.gmmest(theta_true, data=data[:, 0])
    res.print_results()
