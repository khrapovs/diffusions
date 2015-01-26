#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Test suite for diffusions package.

"""

from __future__ import print_function, division

import unittest as ut
import numpy as np

from diffusions import GBM, GBMparam
from diffusions import SDE
from diffusions import nice_errors, ajd_drift, ajd_diff


class SDEParameterTestCase(ut.TestCase):
    """Test SDE, GBM classes."""

    def test_gbmparam_class(self):
        """Test parameter class."""

        mean, sigma = 1.5, .2
        param = GBMparam(mean, sigma)

        self.assertEqual(param.mean, mean)
        self.assertEqual(param.sigma, sigma)
        np.testing.assert_array_equal(param.theta, np.array([mean, sigma]))
        np.testing.assert_array_equal(param.mat_k0,
                                      np.array([mean - sigma**2/2]))
        np.testing.assert_array_equal(param.mat_k1, np.array([[0]]))
        np.testing.assert_array_equal(param.mat_h0, np.array([[sigma**2]]))
        np.testing.assert_array_equal(param.mat_h1, np.array([[[0]]]))


class HelperFunctionsTestCase(ut.TestCase):
    """Test helper functions."""

    def test_nice_errors(self):
        """Test nice errors function."""

        nvars, nobs, nsim = 2, 3, 4
        size = (nvars, nobs, nsim)
        sim = 2
        new_size = (nvars, nobs, 2*nsim)
        errors = np.random.normal(size=size)
        treated_errors = nice_errors(errors, sim)
        self.assertEqual(treated_errors.shape, tuple(new_size))
        np.testing.assert_array_equal(treated_errors.mean(sim), 0)
        np.testing.assert_almost_equal(treated_errors.std(sim),
                                       np.ones((nvars, nobs)))

    def test_ajd_drift(self):
        """Test AJD drift function."""

        mean, sigma = 1.5, .2
        param = GBMparam(mean, sigma)
        nvars, nsim = 1, 2
        size = (nvars, nsim)
        state = np.ones(size)
        drift = state * (mean - sigma**2/2)

        self.assertEqual(ajd_drift(state, param).shape, size)
        np.testing.assert_array_equal(ajd_drift(state, param), drift)

    def test_ajd_diff(self):
        """Test AJD diffusion function."""

        mean, sigma = 1.5, .2
        param = GBMparam(mean, sigma)
        nvars, nsim = 1, 2
        size = (nvars, nsim)
        state = np.ones(size)
        diff = np.ones((nvars, nvars, nsim)) * sigma

        self.assertEqual(ajd_diff(state, param).shape, (nvars, nvars, nsim))
        np.testing.assert_array_equal(ajd_diff(state, param), diff)


class SimulationTestCase(ut.TestCase):
    """Test simulation capabilities."""

    def test_gbm_simupdate(self):
        """Test simulation update of the GBM model."""

        mean, sigma = 1.5, .2
        param = GBMparam(mean, sigma)
        gbm = GBM(param)
        gbm.ndiscr, gbm.interval = 2, .5
        nvars, nsim = 1, 2
        size = (nvars, nsim)
        state = np.ones(size)
        error = np.zeros(size)

        new_state = gbm.update(state, error)
        loc = state * (mean - sigma**2/2)
        scale = np.ones((nvars, nvars, nsim)) * sigma
        delta = gbm.interval / gbm.ndiscr
        new_state_compute = loc * delta + (scale * error).sum(1) * delta**.5

        self.assertEqual(new_state.shape, size)
        np.testing.assert_array_equal(new_state, new_state_compute)


if __name__ == '__main__':
    ut.main()
