#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Test suite for Heston parameter class.

"""
from __future__ import print_function, division

import warnings

import unittest as ut
import numpy as np

from diffusions import (GBMparam, VasicekParam, CIRparam,
                        HestonParam, CentTendParam)


class SDEParameterTestCase(ut.TestCase):
    """Test parameter classes."""

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
        names = ['mean_v', 'kappa', 'eta', 'rho', 'lmbd', 'lmbd_v']

        self.assertEqual(param.measure, 'P')
        self.assertEqual(param.get_model_name(), 'Heston')
        self.assertEquals(param.get_names(), names)
        self.assertEquals(param.get_names(subset='all'), names)
        self.assertEquals(param.get_names(subset='vol'), names[:3] + names[5:])

        self.assertEqual(param.riskfree, riskfree)
        self.assertEqual(param.lmbd, lmbd)
        self.assertEqual(param.lmbd_v, .0)
        self.assertEqual(param.mean_v, mean_v)
        self.assertEqual(param.kappa, kappa)
        self.assertEqual(param.eta, eta)
        self.assertEqual(param.rho, rho)
        self.assertTrue(param.is_valid())

        theta = [riskfree, mean_v, kappa, eta, rho, lmbd, lmbd_v]
        param = HestonParam.from_theta(theta, measure='P')

        self.assertEqual(param.measure, 'P')
        self.assertEqual(param.riskfree, riskfree)
        self.assertEqual(param.lmbd, lmbd)
        self.assertEqual(param.lmbd_v, lmbd_v)
        self.assertEqual(param.mean_v, mean_v)
        self.assertEqual(param.kappa, kappa)
        self.assertEqual(param.eta, eta)
        self.assertEqual(param.rho, rho)
        self.assertTrue(param.is_valid())

        param.convert_to_q()

        self.assertEqual(param.measure, 'Q')
        self.assertEqual(param.riskfree, riskfree)
        self.assertEqual(param.lmbd, 0)
        self.assertEqual(param.lmbd_v, lmbd_v)
        self.assertEqual(param.mean_v, mean_v * kappa / param.kappa)
        self.assertEqual(param.kappa, kappa - lmbd_v * eta)
        self.assertEqual(param.eta, eta)
        self.assertEqual(param.rho, rho)
        self.assertTrue(param.is_valid())

        mat_k0 = [param.riskfree, param.kappa * param.mean_v]
        mat_k1 = [[0, param.lmbd - .5], [0, -param.kappa]]
        mat_h0 = np.zeros((2, 2))
        mat_h1 = [np.zeros((2, 2)), [[1, param.eta*param.rho],
                  [param.eta*param.rho, param.eta**2]]]

        np.testing.assert_array_equal(param.mat_k0, mat_k0)
        np.testing.assert_array_equal(param.mat_k1, mat_k1)
        np.testing.assert_array_equal(param.mat_h0, mat_h0)
        np.testing.assert_array_equal(param.mat_h1, mat_h1)

        param = HestonParam(riskfree=riskfree, lmbd=lmbd, lmbd_v=lmbd_v,
                            mean_v=mean_v, kappa=kappa,
                            eta=eta, rho=rho, measure='Q')

        self.assertEqual(param.measure, 'Q')
        self.assertEqual(param.riskfree, riskfree)
        self.assertEqual(param.lmbd, 0)
        self.assertEqual(param.lmbd_v, lmbd_v)
        self.assertEqual(param.mean_v, mean_v * kappa / param.kappa)
        self.assertEqual(param.kappa, kappa - lmbd_v * eta)
        self.assertEqual(param.eta, eta)
        self.assertEqual(param.rho, rho)
        self.assertTrue(param.is_valid())

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            param.convert_to_q()

        theta = [riskfree, mean_v, kappa, eta, rho, lmbd, lmbd_v]
        param = HestonParam.from_theta(theta, measure='Q')

        self.assertEqual(param.measure, 'Q')
        self.assertEqual(param.riskfree, riskfree)
        self.assertEqual(param.lmbd, 0)
        self.assertEqual(param.lmbd_v, lmbd_v)
        self.assertEqual(param.mean_v, mean_v * kappa / param.kappa)
        self.assertEqual(param.kappa, kappa - lmbd_v * eta)
        self.assertEqual(param.eta, eta)
        self.assertEqual(param.rho, rho)
        self.assertTrue(param.is_valid())

        mean_v, kappa, eta, rho, lmbd = .6, 1.7, .2, -.6, .3
        theta = np.array([mean_v, kappa, eta, rho, lmbd])
        param.update(theta=theta, measure='Q')
        mean_vq = mean_v * kappa / param.kappa
        kappa_q = kappa - lmbd_v * eta

        self.assertEqual(param.measure, 'Q')
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

        theta = np.array([mean_v, kappa, eta, rho, lmbd, lmbd_v])
        theta_vol = np.concatenate((theta[:3], theta[5:]))
        np.testing.assert_array_equal(param.get_theta(), theta)
        np.testing.assert_array_equal(param.get_theta(subset='vol'), theta_vol)

        theta = np.ones(6)
        theta_vol = np.concatenate((theta[:3], theta[5:]))
        param = HestonParam()
        param.update(theta=theta)
        np.testing.assert_array_equal(param.get_theta(), theta)
        np.testing.assert_array_equal(param.get_theta(subset='vol'), theta_vol)

        mat_k0 = [param.riskfree, param.kappa * param.mean_v]
        mat_k1 = [[0, param.lmbd - .5], [0, -param.kappa]]
        mat_h0 = np.zeros((2, 2))
        mat_h1 = [np.zeros((2, 2)), [[1, param.eta*param.rho],
                  [param.eta*param.rho, param.eta**2]]]

        np.testing.assert_array_equal(param.mat_k0, mat_k0)
        np.testing.assert_array_equal(param.mat_k1, mat_k1)
        np.testing.assert_array_equal(param.mat_h0, mat_h0)
        np.testing.assert_array_equal(param.mat_h1, mat_h1)

        theta = np.arange(6)
        param.update(theta=theta)
        theta_vol = np.ones(3) * 2
        param.update(theta=theta_vol, subset='vol', measure='P')
        theta[:3] = theta_vol
        np.testing.assert_array_equal(param.get_theta(), theta)

        self.assertEqual(len(param.get_bounds()), 6)
        self.assertEqual(len(param.get_bounds(subset='vol')), 4)

        self.assertTrue(param.is_valid())
        param = HestonParam(riskfree=riskfree, lmbd=lmbd,
                            mean_v=-mean_v, kappa=kappa,
                            eta=eta, rho=rho)
        self.assertFalse(param.is_valid())
        param = HestonParam(riskfree=riskfree, lmbd=lmbd,
                            mean_v=mean_v, kappa=-kappa,
                            eta=eta, rho=rho)
        self.assertFalse(param.is_valid())
        param = HestonParam(riskfree=riskfree, lmbd=lmbd,
                            mean_v=mean_v, kappa=-kappa,
                            eta=-eta, rho=rho)
        self.assertFalse(param.is_valid())


if __name__ == '__main__':
    ut.main()
