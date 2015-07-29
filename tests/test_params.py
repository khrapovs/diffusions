#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Test suite for parameter classes.

"""
from __future__ import print_function, division

import warnings

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

        self.assertEqual(param.get_model_name(), 'GBM')
        self.assertEquals(param.get_names(), ['mean', 'sigma'])

        self.assertEqual(param.mean, mean)
        self.assertEqual(param.sigma, sigma)
        np.testing.assert_array_equal(param.get_theta(),
                                      np.array([mean, sigma]))

        theta = np.array([mean, sigma])
        np.testing.assert_array_equal(param.get_theta(), theta)

        theta = np.ones(2)
        param = GBMparam.from_theta(theta)
        np.testing.assert_array_equal(param.get_theta(), theta)

        mat_k0 = param.mean - param.sigma**2/2
        mat_k1 = 0.
        mat_h0 = param.sigma**2
        mat_h1 = 0.

        np.testing.assert_array_equal(param.mat_k0, mat_k0)
        np.testing.assert_array_equal(param.mat_k1, mat_k1)
        np.testing.assert_array_equal(param.mat_h0, mat_h0)
        np.testing.assert_array_equal(param.mat_h1, mat_h1)

        theta *= 2
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

        self.assertEqual(param.get_model_name(), 'Vasicek')
        self.assertEquals(param.get_names(), ['mean', 'kappa', 'eta'])

        self.assertEqual(param.mean, mean)
        self.assertEqual(param.kappa, kappa)
        self.assertEqual(param.eta, eta)

        np.testing.assert_array_equal(param.get_theta(),
                                      np.array([mean, kappa, eta]))

        theta = np.ones(3)
        param = VasicekParam.from_theta(theta)
        np.testing.assert_array_equal(param.get_theta(), theta)

        mat_k0 = param.kappa * param.mean
        mat_k1 = -param.kappa
        mat_h0 = param.eta**2
        mat_h1 = 0

        np.testing.assert_array_equal(param.mat_k0, mat_k0)
        np.testing.assert_array_equal(param.mat_k1, mat_k1)
        np.testing.assert_array_equal(param.mat_h0, mat_h0)
        np.testing.assert_array_equal(param.mat_h1, mat_h1)

        theta *= 2
        param.update(theta)
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

        self.assertEqual(param.get_model_name(), 'CIR')
        self.assertEquals(param.get_names(), ['mean', 'kappa', 'eta'])

        self.assertEqual(param.mean, mean)
        self.assertEqual(param.kappa, kappa)
        self.assertEqual(param.eta, eta)

        np.testing.assert_array_equal(param.get_theta(),
                                      np.array([mean, kappa, eta]))

        theta = np.ones(3)
        param = CIRparam.from_theta(theta)
        np.testing.assert_array_equal(param.get_theta(), theta)

        mat_k0 = param.kappa * param.mean
        mat_k1 = -param.kappa
        mat_h0 = 0.
        mat_h1 = param.eta**2

        np.testing.assert_array_equal(param.mat_k0, mat_k0)
        np.testing.assert_array_equal(param.mat_k1, mat_k1)
        np.testing.assert_array_equal(param.mat_h0, mat_h0)
        np.testing.assert_array_equal(param.mat_h1, mat_h1)

        theta *= 2
        param.update(theta)
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
        mean_v = .5
        kappa = 1.5
        eta = .1
        lmbd = .01
        lmbd_v = .5
        rho = -.5
        param = HestonParam(riskfree=riskfree, lmbd=lmbd,
                            mean_v=mean_v, kappa=kappa,
                            eta=eta, rho=rho)
        names = ['mean_v', 'kappa', 'eta', 'lmbd', 'rho']

        self.assertEqual(param.get_model_name(), 'Heston')
        self.assertEquals(param.get_names(), names)
        self.assertEquals(param.get_names(subset='all'), names)
        self.assertEquals(param.get_names(subset='vol'), names[:3])

        self.assertEqual(param.riskfree, riskfree)
        self.assertEqual(param.lmbd, lmbd)
        self.assertEqual(param.lmbd_v, .0)
        self.assertEqual(param.mean_v, mean_v)
        self.assertEqual(param.kappa, kappa)
        self.assertEqual(param.eta, eta)
        self.assertEqual(param.rho, rho)
        self.assertTrue(param.is_valid())

        theta = [riskfree, mean_v, kappa, eta, lmbd, lmbd_v, rho]
        param = HestonParam.from_theta(theta, measure='P')

        self.assertEqual(param.riskfree, riskfree)
        self.assertEqual(param.lmbd, lmbd)
        self.assertEqual(param.lmbd_v, lmbd_v)
        self.assertEqual(param.mean_v, mean_v)
        self.assertEqual(param.kappa, kappa)
        self.assertEqual(param.eta, eta)
        self.assertEqual(param.rho, rho)
        self.assertTrue(param.is_valid())

        param.convert_to_q()

        self.assertEqual(param.riskfree, riskfree)
        self.assertEqual(param.lmbd, 0)
        self.assertEqual(param.lmbd_v, lmbd_v)
        self.assertEqual(param.mean_v, mean_v * kappa / param.kappa)
        self.assertEqual(param.kappa, kappa - lmbd_v * eta)
        self.assertEqual(param.eta, eta)
        self.assertEqual(param.rho, rho)
        self.assertTrue(param.is_valid())

        param = HestonParam(riskfree=riskfree, lmbd=lmbd, lmbd_v=lmbd_v,
                            mean_v=mean_v, kappa=kappa,
                            eta=eta, rho=rho, measure='Q')

        self.assertEqual(param.riskfree, riskfree)
        self.assertEqual(param.lmbd, 0)
        self.assertEqual(param.lmbd_v, lmbd_v)
        self.assertEqual(param.mean_v, mean_v * kappa / param.kappa)
        self.assertEqual(param.kappa, kappa - lmbd_v * eta)
        self.assertEqual(param.eta, eta)
        self.assertEqual(param.rho, rho)
        self.assertTrue(param.is_valid())

        param.convert_to_q()

        self.assertEqual(param.riskfree, riskfree)
        self.assertEqual(param.lmbd, 0)
        self.assertEqual(param.lmbd_v, lmbd_v)
        self.assertEqual(param.mean_v, mean_v * kappa / param.kappa)
        self.assertEqual(param.kappa, kappa - lmbd_v * eta)
        self.assertEqual(param.eta, eta)
        self.assertEqual(param.rho, rho)
        self.assertTrue(param.is_valid())

        theta = [riskfree, mean_v, kappa, eta, lmbd, lmbd_v, rho]
        param = HestonParam.from_theta(theta, measure='Q')

        self.assertEqual(param.riskfree, riskfree)
        self.assertEqual(param.lmbd, 0)
        self.assertEqual(param.lmbd_v, lmbd_v)
        self.assertEqual(param.mean_v, mean_v * kappa / param.kappa)
        self.assertEqual(param.kappa, kappa - lmbd_v * eta)
        self.assertEqual(param.eta, eta)
        self.assertEqual(param.rho, rho)
        self.assertTrue(param.is_valid())

        mean_v, kappa, eta, lmbd, rho = .6, 1.7, .2, .3, -.6
        theta = np.array([mean_v, kappa, eta, lmbd, rho])
        param.update(theta=theta, measure='Q')
        mean_vq = mean_v * kappa / param.kappa
        kappa_q = kappa - lmbd_v * eta

        self.assertEqual(param.riskfree, riskfree)
        self.assertEqual(param.lmbd, 0)
        self.assertEqual(param.lmbd_v, lmbd_v)
        self.assertAlmostEqual(param.mean_v, mean_vq)
        self.assertEqual(param.kappa, kappa_q)
        self.assertEqual(param.eta, eta)
        self.assertEqual(param.rho, rho)
        self.assertTrue(param.is_valid())

        np.testing.assert_array_almost_equal(param.mat_k0,
                                             [riskfree, kappa_q * mean_vq])
        np.testing.assert_array_equal(param.mat_k1,
                                      [[0, -.5], [0, -kappa_q]])

        param = HestonParam(riskfree=riskfree, lmbd=lmbd, lmbd_v=lmbd_v,
                            mean_v=mean_v, kappa=kappa,
                            eta=eta, rho=rho, measure='P')

        theta = np.array([mean_v, kappa, eta, lmbd, rho])
        theta_vol = theta[:3]
        np.testing.assert_array_equal(param.get_theta(), theta)
        np.testing.assert_array_equal(param.get_theta(subset='vol'), theta_vol)

        theta = np.ones(5)
        param = HestonParam()
        param.update(theta=theta)
        np.testing.assert_array_equal(param.get_theta(), theta)
        np.testing.assert_array_equal(param.get_theta(subset='vol'), theta[:3])

        mat_k0 = [param.riskfree, param.kappa * param.mean_v]
        mat_k1 = [[0, param.lmbd - .5], [0, -param.kappa]]
        mat_h0 = np.zeros((2, 2))
        mat_h1 = [np.zeros((2, 2)), [[1, param.eta*param.rho],
                  [param.eta*param.rho, param.eta**2]]]

        np.testing.assert_array_equal(param.mat_k0, mat_k0)
        np.testing.assert_array_equal(param.mat_k1, mat_k1)
        np.testing.assert_array_equal(param.mat_h0, mat_h0)
        np.testing.assert_array_equal(param.mat_h1, mat_h1)

        theta = np.arange(5)
        param.update(theta=theta)
        theta_vol = np.ones(3) * 2
        param.update(theta=theta_vol, subset='vol')
        theta[:3] = theta_vol
        np.testing.assert_array_equal(param.get_theta(), theta)

        self.assertEqual(len(param.get_bounds()), 5)
        self.assertEqual(len(param.get_bounds(subset='vol')), 3)


    def test_centtendparam_class(self):
        """Test CT parameter class."""

        riskfree = .01
        lmbd = .01
        lmbd_s = .5
        lmbd_y = .5
        mean_v = .5
        kappa_s = 1.5
        kappa_y = .5
        eta_s = .1
        eta_y = .01
        rho = -.5
        param = CentTendParam(riskfree=riskfree, lmbd=lmbd,
                              mean_v=mean_v, kappa_s=kappa_s, kappa_y=kappa_y,
                              eta_s=eta_s, eta_y=eta_y, rho=rho)

        self.assertEqual(param.riskfree, riskfree)
        self.assertEqual(param.lmbd, lmbd)
        self.assertEqual(param.lmbd_s, .0)
        self.assertEqual(param.lmbd_y, .0)
        self.assertEqual(param.mean_v, mean_v)
        self.assertEqual(param.kappa_s, kappa_s)
        self.assertEqual(param.kappa_y, kappa_y)
        self.assertEqual(param.eta_s, eta_s)
        self.assertEqual(param.eta_y, eta_y)
        self.assertEqual(param.rho, rho)
        self.assertTrue(param.is_valid())

        param = CentTendParam(riskfree=riskfree, lmbd=lmbd, lmbd_s=lmbd_s,
                              lmbd_y=lmbd_y, mean_v=mean_v, kappa_s=kappa_s,
                              kappa_y=kappa_y, eta_s=eta_s, eta_y=eta_y,
                              rho=rho, measure='Q')

        self.assertEqual(param.riskfree, riskfree)
        self.assertEqual(param.lmbd, 0)
        self.assertEqual(param.lmbd_s, lmbd_s)
        self.assertEqual(param.lmbd_y, lmbd_y)
        self.assertEqual(param.mean_v, mean_v * kappa_y / param.kappa_y
            * param.scale)
        self.assertEqual(param.kappa_s, kappa_s - lmbd_s * eta_s)
        self.assertEqual(param.kappa_y, kappa_y - lmbd_y * eta_y)
        self.assertEqual(param.scale, kappa_s / param.kappa_s)
        self.assertEqual(param.eta_s, eta_s)
        self.assertEqual(param.eta_y, eta_y * param.scale**.5)
        self.assertEqual(param.rho, rho)
        self.assertTrue(param.is_valid())

        mean_v, kappa_s, kappa_y = .6, 1.7, .6
        eta_s, eta_y, lmbd, rho = .2, .02, .1, -.6
        theta = np.array([mean_v, kappa_s, kappa_y, eta_s, eta_y, lmbd, rho])
        param.update(theta=theta, measure='Q')
        kappa_sq = kappa_s - lmbd_s * eta_s
        kappa_yq = kappa_y - lmbd_y * eta_y
        scale = kappa_s / kappa_sq
        mean_vq = mean_v * kappa_y / kappa_yq * scale
        eta_yq = eta_y * param.scale**.5

        self.assertEqual(param.riskfree, riskfree)
        self.assertEqual(param.lmbd, 0)
        self.assertEqual(param.lmbd_s, lmbd_s)
        self.assertEqual(param.lmbd_y, lmbd_y)
        self.assertAlmostEqual(param.mean_v, mean_vq)
        self.assertEqual(param.kappa_s, kappa_sq)
        self.assertEqual(param.kappa_y, kappa_yq)
        self.assertEqual(param.scale, scale)
        self.assertEqual(param.eta_s, eta_s)
        self.assertEqual(param.eta_y, eta_yq)
        self.assertEqual(param.rho, rho)

        mat_k0 = [riskfree, 0., kappa_yq * mean_vq]
        mat_k1 = [[0, -.5, 0],
                  [0, -kappa_sq, kappa_sq],
                  [0, 0, -kappa_yq]]
        mat_h0 = np.zeros((3, 3))
        mat_h1 = np.zeros((3, 3, 3))
        mat_h1[1, 0] = [1, eta_s*param.rho, 0]
        mat_h1[1, 1] = [eta_s*param.rho, eta_s**2, 0]
        mat_h1[2, 2, 2] = eta_yq**2

        np.testing.assert_array_almost_equal(param.mat_k0, mat_k0)
        np.testing.assert_array_equal(param.mat_k1, mat_k1)
        np.testing.assert_array_equal(param.mat_h0, mat_h0)
        np.testing.assert_array_equal(param.mat_h1, mat_h1)

        param = CentTendParam(riskfree=riskfree, lmbd=lmbd,
                              mean_v=mean_v, kappa_s=kappa_s, kappa_y=kappa_y,
                              eta_s=eta_s, eta_y=eta_y, rho=rho)

        theta = np.array([mean_v, kappa_s, kappa_y, eta_s, eta_y, lmbd, rho])
        np.testing.assert_array_equal(param.get_theta(), theta)

        theta = np.ones(7)
        param = CentTendParam()
        param.update(theta=theta)
        np.testing.assert_array_equal(param.get_theta(), theta)

        mat_k0 = [param.riskfree, 0., param.kappa_y * param.mean_v]
        mat_k1 = [[0, param.lmbd - .5, 0],
                  [0, -param.kappa_s, param.kappa_s],
                  [0, 0, -param.kappa_y]]
        mat_h0 = np.zeros((3, 3))
        mat_h1 = np.zeros((3, 3, 3))
        mat_h1[1, 0] = [1, param.eta_s*param.rho, 0]
        mat_h1[1, 1] = [param.eta_s*param.rho, param.eta_s**2, 0]
        mat_h1[2, 2, 2] = param.eta_y**2

        np.testing.assert_array_equal(param.mat_k0, mat_k0)
        np.testing.assert_array_equal(param.mat_k1, mat_k1)
        np.testing.assert_array_equal(param.mat_h0, mat_h0)
        np.testing.assert_array_equal(param.mat_h1, mat_h1)

        theta = np.arange(7)
        param.update(theta=theta)
        theta_vol = np.ones(5) * 2
        param.update(theta=theta_vol, subset='vol')
        theta[:5] = theta_vol
        np.testing.assert_array_equal(param.get_theta(), theta)

        self.assertEqual(len(param.get_bounds()), 7)
        self.assertEqual(len(param.get_bounds(subset='vol')), 5)


if __name__ == '__main__':
    ut.main()
