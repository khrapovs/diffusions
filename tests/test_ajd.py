#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Test suite for AJD parameterization.

"""
from __future__ import print_function, division

import unittest as ut
import numpy as np
import numpy.testing as npt

from diffusions import (GBMparam, VasicekParam, CIRparam,
                        HestonParam, CentTendParam)
from diffusions.helper_functions import ajd_drift, ajd_diff


class DriftTestCase(ut.TestCase):
    """Test Drift function."""

    def test_ajd_drift_gbm(self):
        """Test AJD drift function for GBM model."""

        mean, sigma = 1.5, .2
        param = GBMparam(mean, sigma)
        nvars, nsim = 1, 2
        size = (nsim, nvars)
        state = np.ones(size)
        drift = state * (mean - sigma**2/2)

        self.assertEqual(ajd_drift(state, param).shape, size)
        npt.assert_array_equal(ajd_drift(state, param), drift)

    def test_ajd_drift_vasicek(self):
        """Test AJD drift function for Vasicek model."""

        mean, kappa, eta = 1.5, 1, .2
        param = VasicekParam(mean, kappa, eta)
        nvars, nsim = 1, 2
        size = (nsim, nvars)
        state = np.ones(size)
        drift = kappa * (mean - state)

        self.assertEqual(ajd_drift(state, param).shape, size)
        npt.assert_array_equal(ajd_drift(state, param), drift)

    def test_ajd_drift_cir(self):
        """Test AJD drift function for CIR model."""

        mean, kappa, eta = 1.5, 1, .2
        param = CIRparam(mean, kappa, eta)
        nvars, nsim = 1, 2
        size = (nsim, nvars)
        state = np.ones(size)
        drift = kappa * (mean - state)

        self.assertEqual(ajd_drift(state, param).shape, size)
        npt.assert_array_equal(ajd_drift(state, param), drift)

    def test_ajd_drift_heston(self):
        """Test AJD drift function for Heston model."""

        riskfree, lmbd, mean_v, kappa, eta, rho = 0., .01, .2, 1.5, .2, -.5
        param = HestonParam(riskfree=riskfree, lmbd=lmbd,
                            mean_v=mean_v, kappa=kappa,
                            eta=eta, rho=rho)
        nvars, nsim = 2, 3
        size = (nsim, nvars)
        state = np.ones(size)
        drift = np.ones(size)
        drift_r = riskfree + state[:, 1]**2 * (lmbd - .5)
        drift_v = kappa * (mean_v - state[:, 1])
        drift = np.vstack([drift_r, drift_v]).T

        self.assertEqual(ajd_drift(state, param).shape, drift.shape)
        npt.assert_almost_equal(ajd_drift(state, param), drift)

    def test_ajd_drift_ct(self):
        """Test AJD drift function for CT model."""

        riskfree, lmbd, mean_v = 0., .01, .2
        kappa_s, kappa_y, eta_s, eta_y, rho = 1.5, .5, .2, .02, -.5
        param = CentTendParam(riskfree=riskfree, lmbd=lmbd,
                              mean_v=mean_v, kappa_s=kappa_s, kappa_y=kappa_y,
                              eta_s=eta_s, eta_y=eta_y, rho=rho)
        nvars, nsim = 3, 5
        size = (nsim, nvars)
        state = np.ones(size)
        drift = np.ones(size)
        drift_r = riskfree + state[:, 1]**2 * (lmbd - .5)
        drift_s = kappa_s * (state[:, 2] - state[:, 1])
        drift_y = kappa_y * (mean_v - state[:, 2])
        drift = np.vstack([drift_r, drift_s, drift_y]).T

        self.assertEqual(ajd_drift(state, param).shape, drift.shape)
        npt.assert_almost_equal(ajd_drift(state, param), drift)


class DiffusionTestCase(ut.TestCase):
    """Test Diffusio function."""

    def test_ajd_diff_gbm(self):
        """Test AJD diffusion function for GBM model."""

        mean, sigma = 1.5, .2
        param = GBMparam(mean, sigma)
        nvars, nsim = 1, 2
        size = (nsim, nvars)
        state = np.ones(size)
        diff = np.ones((nsim, nvars, nvars)) * sigma

        self.assertEqual(ajd_diff(state, param).shape, (nsim, nvars, nvars))
        npt.assert_array_equal(ajd_diff(state, param), diff)

    def test_ajd_diff_vasicek(self):
        """Test AJD diffusion function for Vasicek model."""

        mean, kappa, eta = 1.5, 1, .2
        param = VasicekParam(mean, kappa, eta)
        nvars, nsim = 1, 2
        size = (nsim, nvars)
        state = np.ones(size)
        diff = np.ones((nsim, nvars, nvars)) * eta

        self.assertEqual(ajd_diff(state, param).shape, (nsim, nvars, nvars))
        npt.assert_array_equal(ajd_diff(state, param), diff)

    def test_ajd_diff_cir(self):
        """Test AJD diffusion function for CIR model."""

        mean, kappa, eta = 1.5, 1, .2
        param = CIRparam(mean, kappa, eta)
        nvars, nsim = 1, 2
        size = (nsim, nvars)
        state_val = 4
        state = np.ones(size)*state_val
        diff = eta * state_val**.5 * np.ones((nsim, nvars, nvars))

        self.assertEqual(ajd_diff(state, param).shape, (nsim, nvars, nvars))
        npt.assert_array_equal(ajd_diff(state, param), diff)


    def test_ajd_diff_heston(self):
        """Test AJD diffusion function for Heston model."""

        riskfree, lmbd, mean_v, kappa, eta, rho = 0., .01, .2, 1.5, .2, -.0
        param = HestonParam(riskfree=riskfree, lmbd=lmbd,
                            mean_v=mean_v, kappa=kappa,
                            eta=eta, rho=rho)
        nvars, nsim = 2, 3
        size = (nsim, nvars)
        state = np.ones(size)
        diff = np.ones((nsim, nvars, nvars))
        var = np.array([[1, eta*rho], [eta*rho, eta**2]])
        var = ((np.ones((nsim, nvars, nvars)) * var).T * state[:, 1]).T
        diff = np.linalg.cholesky(var)

        self.assertEqual(ajd_diff(state, param).shape, diff.shape)
        npt.assert_array_equal(ajd_diff(state, param), diff)

    def test_ajd_diff_ct(self):
        """Test AJD diffusion function for CT model."""

        riskfree, lmbd, mean_v = 0., .01, .2
        kappa_s, kappa_y, eta_s, eta_y, rho = 1.5, .5, .2, .02, -.5
        param = CentTendParam(riskfree=riskfree, lmbd=lmbd,
                              mean_v=mean_v, kappa_s=kappa_s, kappa_y=kappa_y,
                              eta_s=eta_s, eta_y=eta_y, rho=rho)
        nvars, nsim = 3, 5
        size = (nsim, nvars)
        state = np.ones(size)
        diff = np.ones((nsim, nvars, nvars))
        var1 = np.array([[1, eta_s*rho, 0],
                         [eta_s*rho, eta_s**2, 0],
                         [0, 0, 0]])
        var2 = np.zeros((3, 3))
        var2[-1, -1] = eta_y**2

        var = ((np.ones((nsim, nvars, nvars)) * var1).T * state[:, 1]).T \
            + ((np.ones((nsim, nvars, nvars)) * var2).T * state[:, 2]).T
        diff = np.linalg.cholesky(var)

        self.assertEqual(ajd_diff(state, param).shape, diff.shape)
        npt.assert_array_equal(ajd_diff(state, param), diff)


if __name__ == '__main__':
    ut.main()
