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
    gbm = GBM()

    x0, T, h, M, S = mu, 200, 1., 100, 3
    N = int(float(T) / h)
    gbm.simulate(x0, theta_true, h, M, N, S)

    gbm.plot_trajectories(3)