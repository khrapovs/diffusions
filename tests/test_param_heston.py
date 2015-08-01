#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Test suite for Heston parameter class.

"""
from __future__ import print_function, division

import warnings

import unittest as ut
import numpy as np
import numpy.testing as npt

from diffusions import HestonParam


class SDEParameterTestCase(ut.TestCase):
    """Test parameter classes."""

    def test_init(self):
        """Test initialization."""

        riskfree = .01
        mean_v = .5
        kappa = 1.5
        eta = .1
        lmbd = .01
        rho = -.5

        param = HestonParam(riskfree=riskfree, lmbd=lmbd, mean_v=mean_v,
                            kappa=kappa, eta=eta, rho=rho)
        names = ['mean_v', 'kappa', 'eta', 'rho', 'lmbd', 'lmbd_v']

        self.assertEqual(param.measure, 'P')
        self.assertEqual(param.get_model_name(), 'Heston')
        self.assertEquals(param.get_names(), names)
        self.assertEquals(param.get_names(subset='all'), names)
        self.assertEquals(param.get_names(subset='vol'), names[:3] + names[5:])
        self.assertEquals(param.get_names(subset='vol', measure='P'),
                          names[:3])
        self.assertEquals(param.get_names(subset='vol', measure='Q'),
                          names[:3])
        self.assertEquals(param.get_names(subset='all', measure='P'),
                          names[:-1])
        self.assertEquals(param.get_names(subset='all', measure='Q'),
                          names[:-1])

        self.assertEqual(param.riskfree, riskfree)
        self.assertEqual(param.lmbd, lmbd)
        self.assertEqual(param.lmbd_v, .0)
        self.assertEqual(param.mean_v, mean_v)
        self.assertEqual(param.kappa, kappa)
        self.assertEqual(param.eta, eta)
        self.assertEqual(param.rho, rho)
        self.assertTrue(param.is_valid())

    def test_init_q(self):
        """Test initialization under Q."""

        riskfree = .01
        mean_v = .5
        kappa = 1.5
        eta = .1
        lmbd = .01
        lmbd_v = .5
        rho = -.5

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

    def test_from_theta(self):
        """Test from theta."""

        riskfree = .01
        mean_v = .5
        kappa = 1.5
        eta = .1
        lmbd = .01
        lmbd_v = .5
        rho = -.5

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

    def test_from_theta_q(self):
        """Test from theta under Q."""

        riskfree = .01
        mean_v = .5
        kappa = 1.5
        eta = .1
        lmbd = .01
        lmbd_v = .5
        rho = -.5

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

    def test_convert_to_q(self):
        """Test conversion to Q."""

        riskfree = .01
        mean_v = .5
        kappa = 1.5
        eta = .1
        lmbd = .01
        lmbd_v = .5
        rho = -.5

        theta = [riskfree, mean_v, kappa, eta, rho, lmbd, lmbd_v]
        param = HestonParam.from_theta(theta)
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

    def test_ajd_matrices(self):
        """Test AJD matrices."""

        riskfree = .01
        mean_v = .5
        kappa = 1.5
        eta = .1
        lmbd = .01
        lmbd_v = .5
        rho = -.5

        param = HestonParam(riskfree=riskfree, lmbd=lmbd, lmbd_v=lmbd_v,
                            mean_v=mean_v, kappa=kappa, eta=eta, rho=rho)

        mat_k0 = [riskfree, kappa * mean_v]
        mat_k1 = [[0, lmbd - .5], [0, -kappa]]
        mat_h0 = np.zeros((2, 2))
        mat_h1 = [np.zeros((2, 2)), [[1, eta*rho], [eta*rho, eta**2]]]

        npt.assert_array_equal(param.mat_k0, mat_k0)
        npt.assert_array_equal(param.mat_k1, mat_k1)
        npt.assert_array_equal(param.mat_h0, mat_h0)
        npt.assert_array_equal(param.mat_h1, mat_h1)

        param.convert_to_q()

        kappa_q = kappa - lmbd_v * eta
        mean_v_q = mean_v * kappa / kappa_q

        mat_k0 = [riskfree, kappa_q * mean_v_q]
        mat_k1 = [[0, - .5], [0, -kappa_q]]
        mat_h0 = np.zeros((2, 2))
        mat_h1 = [np.zeros((2, 2)), [[1, eta*rho], [eta*rho, eta**2]]]

        npt.assert_array_equal(param.mat_k0, mat_k0)
        npt.assert_array_equal(param.mat_k1, mat_k1)
        npt.assert_array_equal(param.mat_h0, mat_h0)
        npt.assert_array_equal(param.mat_h1, mat_h1)

        theta = np.ones(6)
        theta_vol = np.concatenate((theta[:3], theta[5:]))
        param = HestonParam()
        param.update(theta=theta)
        npt.assert_array_equal(param.get_theta(), theta)
        npt.assert_array_equal(param.get_theta(subset='vol'), theta_vol)

        mat_k0 = [param.riskfree, param.kappa * param.mean_v]
        mat_k1 = [[0, param.lmbd - .5], [0, -param.kappa]]
        mat_h0 = np.zeros((2, 2))
        mat_h1 = [np.zeros((2, 2)), [[1, param.eta*param.rho],
                  [param.eta*param.rho, param.eta**2]]]

        npt.assert_array_equal(param.mat_k0, mat_k0)
        npt.assert_array_equal(param.mat_k1, mat_k1)
        npt.assert_array_equal(param.mat_h0, mat_h0)
        npt.assert_array_equal(param.mat_h1, mat_h1)

    def test_update(self):
        """Test update."""

        riskfree = .01
        mean_v = .5
        kappa = 1.5
        eta = .1
        lmbd = .01
        lmbd_v = .5
        rho = -.5

        param = HestonParam(riskfree=riskfree, lmbd=lmbd, lmbd_v=lmbd_v,
                            mean_v=mean_v, kappa=kappa, eta=eta, rho=rho)

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

        npt.assert_array_almost_equal(param.mat_k0,
                                      [riskfree, kappa_q * mean_vq])
        npt.assert_array_equal(param.mat_k1, [[0, -.5], [0, -kappa_q]])

    def test_get_theta(self):
        """Test get theta."""

        riskfree = .01
        mean_v = .5
        kappa = 1.5
        eta = .1
        lmbd = .01
        lmbd_v = .5
        rho = -.5

        param = HestonParam(riskfree=riskfree, lmbd=lmbd, lmbd_v=lmbd_v,
                            mean_v=mean_v, kappa=kappa,
                            eta=eta, rho=rho, measure='P')

        theta = np.array([mean_v, kappa, eta, rho, lmbd, lmbd_v])
        theta_vol = np.concatenate((theta[:3], theta[5:]))

        npt.assert_array_equal(param.get_theta(), theta)
        npt.assert_array_equal(param.get_theta(subset='all'), theta)
        npt.assert_array_equal(param.get_theta(subset='all', measure='PQ'),
                               theta)
        npt.assert_array_equal(param.get_theta(subset='all', measure='P'),
                               theta[:-1])
        npt.assert_array_equal(param.get_theta(subset='all', measure='Q'),
                               theta[:-1])
        npt.assert_array_equal(param.get_theta(subset='vol'), theta_vol)
        npt.assert_array_equal(param.get_theta(subset='vol', measure='PQ'),
                               theta_vol)
        npt.assert_array_equal(param.get_theta(subset='vol', measure='P'),
                               theta_vol[:-1])
        npt.assert_array_equal(param.get_theta(subset='vol', measure='Q'),
                               theta_vol[:-1])

        theta = np.arange(6)
        param.update(theta=theta)
        theta_vol = np.ones(3) * 2
        param.update(theta=theta_vol, subset='vol', measure='P')
        theta[:3] = theta_vol
        npt.assert_array_equal(param.get_theta(), theta)

    def test_bounds(self):
        """Test bounds."""

        param = HestonParam()
        self.assertEqual(len(param.get_bounds()), 6)
        self.assertEqual(len(param.get_bounds(subset='all')), 6)
        self.assertEqual(len(param.get_bounds(subset='all', measure='PQ')), 6)
        self.assertEqual(len(param.get_bounds(subset='all', measure='P')), 5)
        self.assertEqual(len(param.get_bounds(subset='all', measure='Q')), 5)
        self.assertEqual(len(param.get_bounds(subset='vol')), 4)
        self.assertEqual(len(param.get_bounds(subset='vol', measure='PQ')), 4)
        self.assertEqual(len(param.get_bounds(subset='vol', measure='P')), 3)
        self.assertEqual(len(param.get_bounds(subset='vol', measure='Q')), 3)

    def test_validity(self):
        """Test validity."""

        riskfree = .01
        mean_v = .5
        kappa = 1.5
        eta = .1
        lmbd = .01
        lmbd_v = .5
        rho = -.5

        param = HestonParam(riskfree=riskfree, lmbd=lmbd, lmbd_v=lmbd_v,
                            mean_v=mean_v, kappa=kappa,
                            eta=eta, rho=rho, measure='P')

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
