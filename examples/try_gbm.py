#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Generic Model

"""
from __future__ import print_function, division

import numpy as np

from diffusions.gbm import GBM


if __name__ == '__main__':
    mu, sigma = .05, .1
    theta_true = np.array([mu, sigma])
    gbm = GBM(theta_true)

    x0, T, h, M, S = 0, 200, 1., 100, 3
    x0, nperiods, interval, ndiscr, nsim = 0, 200, .5, 100, 3
    npoints = int(nperiods / interval)
    gbm.simulate(x0, interval, ndiscr, npoints, nsim)

    gbm.plot_trajectories(3)

    gbm.plot_final_distr()
