#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Generic Model

"""
from __future__ import print_function, division

#import numpy as np

from diffusions.gbm import GBM, GBMparam


if __name__ == '__main__':
    mean, sigma = .5, .2
    theta_true = GBMparam(mean, sigma)
    gbm = GBM(theta_true)

    x0, nperiods, interval, ndiscr, nsim = 0, 200, .1, 100, 1
    npoints = int(nperiods / interval)
    gbm.simulate(x0, interval, ndiscr, npoints, nsim)
    data = gbm.paths

    gbm.plot_trajectories()

    #gbm.plot_final_distr()

    res = gbm.gmmest(theta_true, data=data)
    res.print_results()
