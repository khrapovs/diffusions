#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Test suite for parameter classes.

"""
from __future__ import print_function, division

import unittest as ut
import numpy as np

from diffusions import (GBMparam, VasicekParam, CIRparam,
                        HestonParam, CentTendParam)


class SDEParameterTestCase(ut.TestCase):
    """Test parameter classes."""

    def test_gbmparam_class(self):
        """Test GBM parameter class."""

        mean, sigma = 1.5, .2
        param = GBMparam(mean, sigma)

        self.assertEqual(param.mean, mean)
        self.assertEqual(param.sigma, sigma)
        np.testing.assert_array_equal(param.get_theta(),
                                      np.array([mean, sigma]))

        theta = np.array([mean, sigma])
        np.testing.assert_array_equal(param.get_theta(), theta)

        theta = np.ones(2)
        param = GBMparam()
        param.update(theta=theta)
        np.testing.assert_array_equal(param.get_theta(), theta)

        mat_k0 = param.mean - param.sigma**2/2
        mat_k1 = 0.
        mat_h0 = param.sigma**2
        mat_h1 = 0.

        np.testing.assert_array_equal(param.mat_k0, mat_k0)
        np.testing.assert_array_equal(param.mat_k1, mat_k1)
        np.testing.assert_array_equal(param.mat_h0, mat_h0)
        np.testing.assert_array_equal(param.mat_h1, mat_h1)

    def test_vasicekparam_class(self):
        """Test Vasicek parameter class."""

        mean, kappa, eta = 1.5, 1., .2
        param = VasicekParam(mean, kappa, eta)

        self.assertEqual(param.mean, mean)
        self.assertEqual(param.kappa, kappa)
        self.assertEqual(param.eta, eta)

        np.testing.assert_array_equal(param.get_theta(),
                                      np.array([mean, kappa, eta]))

        theta = np.ones(3)
        param = VasicekParam()
        param.update(theta=theta)
        np.testing.assert_array_equal(param.get_theta(), theta)

        mat_k0 = param.kappa * param.mean
        mat_k1 = -param.kappa
        mat_h0 = param.eta**2
        mat_h1 = 0

        np.testing.assert_array_equal(param.mat_k0, mat_k0)
        np.testing.assert_array_equal(param.mat_k1, mat_k1)
        np.testing.assert_array_equal(param.mat_h0, mat_h0)
        np.testing.assert_array_equal(param.mat_h1, mat_h1)

    def test_cirparam_class(self):
        """Test CIR parameter class."""

        mean, kappa, eta = 1.5, 1., .2
        param = CIRparam(mean, kappa, eta)

        self.assertEqual(param.mean, mean)
        self.assertEqual(param.kappa, kappa)
        self.assertEqual(param.eta, eta)

        np.testing.assert_array_equal(param.get_theta(),
                                      np.array([mean, kappa, eta]))

        theta = np.ones(3)
        param = CIRparam()
        param.update(theta=theta)
        np.testing.assert_array_equal(param.get_theta(), theta)

        mat_k0 = param.kappa * param.mean
        mat_k1 = -param.kappa
        mat_h0 = 0.
        mat_h1 = param.eta**2

        np.testing.assert_array_equal(param.mat_k0, mat_k0)
        np.testing.assert_array_equal(param.mat_k1, mat_k1)
        np.testing.assert_array_equal(param.mat_h0, mat_h0)
        np.testing.assert_array_equal(param.mat_h1, mat_h1)

    def test_hestonparam_class(self):
        """Test Heston parameter class."""

        riskfree = .01
        lmbd = .01
        mean_v = .5
        kappa = 1.5
        eta = .1
        rho = -.5
        param = HestonParam(riskfree=riskfree, lmbd=lmbd,
                            mean_v=mean_v, kappa=kappa,
                            eta=eta, rho=rho)

        self.assertEqual(param.riskfree, riskfree)
        self.assertEqual(param.lmbd, lmbd)
        self.assertEqual(param.mean_v, mean_v)
        self.assertEqual(param.kappa, kappa)
        self.assertEqual(param.eta, eta)
        self.assertEqual(param.rho, rho)
        self.assertTrue(param.is_valid())

        theta = np.array([lmbd, mean_v, kappa, eta, rho])
        np.testing.assert_array_equal(param.get_theta(), theta)

        theta = np.ones(5)
        param = HestonParam()
        param.update(theta=theta)
        np.testing.assert_array_equal(param.get_theta(), theta)

        mat_k0 = [param.riskfree, param.kappa * param.mean_v]
        mat_k1 = [[0, param.lmbd - .5], [0, -param.kappa]]
        mat_h0 = np.zeros((2, 2))
        mat_h1 = [np.zeros((2, 2)), [[1, param.eta*param.rho],
                  [param.eta*param.rho, param.eta**2]]]

        np.testing.assert_array_equal(param.mat_k0, mat_k0)
        np.testing.assert_array_equal(param.mat_k1, mat_k1)
        np.testing.assert_array_equal(param.mat_h0, mat_h0)
        np.testing.assert_array_equal(param.mat_h1, mat_h1)

    def test_centtendparam_class(self):
        """Test CT parameter class."""

        riskfree = .01
        lmbd = .01
        mean_v = .5
        kappa_s = 1.5
        kappa_v = .5
        eta_s = .1
        eta_v = .01
        rho = -.5
        param = CentTendParam(riskfree=riskfree, lmbd=lmbd,
                              mean_v=mean_v, kappa_s=kappa_s, kappa_v=kappa_v,
                              eta_s=eta_s, eta_v=eta_v, rho=rho)

        self.assertEqual(param.riskfree, riskfree)
        self.assertEqual(param.lmbd, lmbd)
        self.assertEqual(param.mean_v, mean_v)
        self.assertEqual(param.kappa_s, kappa_s)
        self.assertEqual(param.kappa_v, kappa_v)
        self.assertEqual(param.eta_s, eta_s)
        self.assertEqual(param.eta_v, eta_v)
        self.assertEqual(param.rho, rho)
        self.assertTrue(param.is_valid())

        theta = np.array([lmbd, mean_v, kappa_s, kappa_v, eta_s, eta_v, rho])
        np.testing.assert_array_equal(param.get_theta(), theta)

        theta = np.ones(7)
        param = CentTendParam()
        param.update(theta=theta)
        np.testing.assert_array_equal(param.get_theta(), theta)

        mat_k0 = [param.riskfree, 0., param.kappa_v * param.mean_v]
        mat_k1 = [[0, param.lmbd - .5, 0],
                  [0, -param.kappa_s, param.kappa_s],
                  [0, 0, -param.kappa_v]]
        mat_h0 = np.zeros((3, 3))
        mat_h1 = np.zeros((3, 3, 3))
        mat_h1[1, 0] = [1, param.eta_s*param.rho, 0]
        mat_h1[1, 1] = [param.eta_s*param.rho, param.eta_s**2, 0]
        mat_h1[2, 2, 2] = param.eta_v**2

        np.testing.assert_array_equal(param.mat_k0, mat_k0)
        np.testing.assert_array_equal(param.mat_k1, mat_k1)
        np.testing.assert_array_equal(param.mat_h0, mat_h0)
        np.testing.assert_array_equal(param.mat_h1, mat_h1)


if __name__ == '__main__':
    ut.main()
