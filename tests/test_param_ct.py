#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Test suite for CT parameter class.

"""
from __future__ import print_function, division

import warnings

import unittest as ut
import numpy as np
import numpy.testing as npt

from diffusions import CentTendParam


class SDEParameterTestCase(ut.TestCase):
    """Test parameter classes."""

    def test_init(self):
        """Test initialization."""

        riskfree = .01
        lmbd = .01
        mean_v = .5
        kappa_s = 1.5
        kappa_y = .5
        eta_s = .1
        eta_y = .01
        rho = -.5

        param = CentTendParam(riskfree=riskfree, lmbd=lmbd,
                              mean_v=mean_v, kappa_s=kappa_s, kappa_y=kappa_y,
                              eta_s=eta_s, eta_y=eta_y, rho=rho)

        names = ['mean_v', 'kappa_s', 'kappa_y', 'eta_s', 'eta_y',
                 'rho', 'lmbd', 'lmbd_s', 'lmbd_y']

        self.assertEqual(param.measure, 'P')
        self.assertEqual(param.get_model_name(), 'Central Tendency')
        self.assertEqual(param.get_names(), names)
        self.assertEqual(param.get_names(subset='all'), names)
        self.assertEqual(param.get_names(subset='vol'),
                          names[:5] + names[-2:])
        self.assertEqual(param.get_names(subset='vol', measure='P'),
                          names[:5])
        self.assertEqual(param.get_names(subset='vol', measure='Q'),
                          names[:5])
        self.assertEqual(param.get_names(subset='all', measure='P'),
                          names[:-2])
        self.assertEqual(param.get_names(subset='all', measure='Q'),
                          names[:-2])

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

    def test_constraints(self):
        """Test constraints."""

        riskfree = .01
        lmbd = .01
        mean_v = .5
        kappa_s = 1.5
        kappa_y = .5
        eta_s = .1
        eta_y = .01
        rho = -.5

        param = CentTendParam(riskfree=riskfree, lmbd=lmbd,
                              mean_v=mean_v, kappa_s=kappa_s, kappa_y=kappa_y,
                              eta_s=eta_s, eta_y=eta_y, rho=rho)

        cons = param.get_constraints()
        self.assertTrue(cons[0]['fun'](param.get_theta()) > 0)
        self.assertTrue(cons[1]['fun'](param.get_theta()) > 0)

        riskfree = .01
        lmbd = .01
        mean_v = .5
        kappa_s = .5
        kappa_y = 1.5
        eta_s = .01
        eta_y = .1
        rho = -.5

        param = CentTendParam(riskfree=riskfree, lmbd=lmbd,
                              mean_v=mean_v, kappa_s=kappa_s, kappa_y=kappa_y,
                              eta_s=eta_s, eta_y=eta_y, rho=rho)

        cons = param.get_constraints()
        self.assertFalse(cons[0]['fun'](param.get_theta()) > 0)
        self.assertFalse(cons[1]['fun'](param.get_theta()) > 0)


    def test_init_q(self):
        """Test initialization under Q."""

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

        param = CentTendParam(riskfree=riskfree,
                              lmbd=lmbd, lmbd_s=lmbd_s, lmbd_y=lmbd_y,
                              mean_v=mean_v, kappa_s=kappa_s, kappa_y=kappa_y,
                              eta_s=eta_s, eta_y=eta_y, rho=rho, measure='Q')

        kappa_sq = kappa_s - lmbd_s * eta_s
        kappa_yq = kappa_y - lmbd_y * eta_y
        scale = kappa_s / kappa_sq

        self.assertEqual(param.measure, 'Q')
        self.assertEqual(param.riskfree, riskfree)
        self.assertEqual(param.lmbd, 0)
        self.assertEqual(param.lmbd_s, lmbd_s)
        self.assertEqual(param.lmbd_y, lmbd_y)
        self.assertEqual(param.mean_v, mean_v * kappa_y / kappa_yq * scale)
        self.assertEqual(param.kappa_s, kappa_sq)
        self.assertEqual(param.kappa_y, kappa_yq)
        self.assertEqual(param.eta_s, eta_s)
        self.assertEqual(param.eta_y, eta_y * scale**.5)
        self.assertEqual(param.rho, rho)
        self.assertTrue(param.is_valid())

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            param.convert_to_q()

    def test_from_theta(self):
        """Test from theta."""

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

        theta = [riskfree, mean_v, kappa_s, kappa_y, eta_s, eta_y,
                 rho, lmbd, lmbd_s, lmbd_y]
        param = CentTendParam.from_theta(theta, measure='P')

        self.assertEqual(param.measure, 'P')
        self.assertEqual(param.riskfree, riskfree)
        self.assertEqual(param.lmbd, lmbd)
        self.assertEqual(param.lmbd_s, lmbd_s)
        self.assertEqual(param.lmbd_y, lmbd_y)
        self.assertEqual(param.mean_v, mean_v)
        self.assertEqual(param.kappa_s, kappa_s)
        self.assertEqual(param.kappa_y, kappa_y)
        self.assertEqual(param.eta_s, eta_s)
        self.assertEqual(param.eta_y, eta_y)
        self.assertEqual(param.rho, rho)
        self.assertTrue(param.is_valid())

    def test_from_theta_q(self):
        """Test from theta under Q."""

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

        theta = [riskfree, mean_v, kappa_s, kappa_y, eta_s, eta_y,
                 rho, lmbd, lmbd_s, lmbd_y]
        param = CentTendParam.from_theta(theta, measure='Q')

        kappa_sq = kappa_s - lmbd_s * eta_s
        kappa_yq = kappa_y - lmbd_y * eta_y
        scale = kappa_s / kappa_sq

        self.assertEqual(param.measure, 'Q')
        self.assertEqual(param.riskfree, riskfree)
        self.assertEqual(param.lmbd, 0)
        self.assertEqual(param.lmbd_s, lmbd_s)
        self.assertEqual(param.lmbd_y, lmbd_y)
        self.assertEqual(param.mean_v, mean_v * kappa_y / kappa_yq * scale)
        self.assertEqual(param.kappa_s, kappa_sq)
        self.assertEqual(param.kappa_y, kappa_yq)
        self.assertEqual(param.eta_s, eta_s)
        self.assertEqual(param.eta_y, eta_y * scale**.5)
        self.assertEqual(param.rho, rho)
        self.assertTrue(param.is_valid())

    def test_convert_to_q(self):
        """Test conversion to Q."""

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

        theta = [riskfree, mean_v, kappa_s, kappa_y, eta_s, eta_y,
                 rho, lmbd, lmbd_s, lmbd_y]
        param = CentTendParam.from_theta(theta)
        param.convert_to_q()

        kappa_sq = kappa_s - lmbd_s * eta_s
        kappa_yq = kappa_y - lmbd_y * eta_y
        scale = kappa_s / kappa_sq

        self.assertEqual(param.measure, 'Q')
        self.assertEqual(param.riskfree, riskfree)
        self.assertEqual(param.lmbd, 0)
        self.assertEqual(param.lmbd_s, lmbd_s)
        self.assertEqual(param.lmbd_y, lmbd_y)
        self.assertEqual(param.mean_v, mean_v * kappa_y / kappa_yq * scale)
        self.assertEqual(param.kappa_s, kappa_sq)
        self.assertEqual(param.kappa_y, kappa_yq)
        self.assertEqual(param.eta_s, eta_s)
        self.assertEqual(param.eta_y, eta_y * scale**.5)
        self.assertEqual(param.rho, rho)
        self.assertTrue(param.is_valid())

    def test_ajd_matrices(self):
        """Test AJD matrices."""

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

        param = CentTendParam(riskfree=riskfree,
                              lmbd=lmbd, lmbd_s=lmbd_s, lmbd_y=lmbd_y,
                              mean_v=mean_v, kappa_s=kappa_s, kappa_y=kappa_y,
                              eta_s=eta_s, eta_y=eta_y, rho=rho)

        kappa_sq = kappa_s - lmbd_s * eta_s
        kappa_yq = kappa_y - lmbd_y * eta_y
        scale = kappa_s / kappa_sq
        mean_vq = mean_v * kappa_y / kappa_yq * scale
        eta_yq = eta_y * scale**.5

        mat_k0 = [riskfree, 0., kappa_y * mean_v]
        mat_k1 = [[0, lmbd-.5, 0],
                  [0, -kappa_s, kappa_s],
                  [0, 0, -kappa_y]]
        mat_h0 = np.zeros((3, 3))
        mat_h1 = np.zeros((3, 3, 3))
        mat_h1[1, 0] = [1, eta_s*param.rho, 0]
        mat_h1[1, 1] = [eta_s*param.rho, eta_s**2, 0]
        mat_h1[2, 2, 2] = eta_y**2

        npt.assert_array_almost_equal(param.mat_k0, mat_k0)
        npt.assert_array_equal(param.mat_k1, mat_k1)
        npt.assert_array_equal(param.mat_h0, mat_h0)
        npt.assert_array_equal(param.mat_h1, mat_h1)

        param.convert_to_q()

        mat_k0 = [riskfree, 0., kappa_yq * mean_vq]
        mat_k1 = [[0, -.5, 0],
                  [0, -kappa_sq, kappa_sq],
                  [0, 0, -kappa_yq]]
        mat_h0 = np.zeros((3, 3))
        mat_h1 = np.zeros((3, 3, 3))
        mat_h1[1, 0] = [1, eta_s*param.rho, 0]
        mat_h1[1, 1] = [eta_s*param.rho, eta_s**2, 0]
        mat_h1[2, 2, 2] = eta_yq**2

        npt.assert_array_almost_equal(param.mat_k0, mat_k0)
        npt.assert_array_equal(param.mat_k1, mat_k1)
        npt.assert_array_equal(param.mat_h0, mat_h0)
        npt.assert_array_equal(param.mat_h1, mat_h1)

        theta = np.ones(9)
        theta_vol = np.concatenate((theta[:5], theta[-2:]))
        param = CentTendParam()
        param.update(theta=theta)
        npt.assert_array_equal(param.get_theta(), theta)
        npt.assert_array_equal(param.get_theta(subset='vol'), theta_vol)

        mat_k0 = [param.riskfree, 0., param.kappa_y * param.mean_v]
        mat_k1 = [[0, param.lmbd-.5, 0],
                  [0, -param.kappa_s, param.kappa_s],
                  [0, 0, -param.kappa_y]]
        mat_h0 = np.zeros((3, 3))
        mat_h1 = np.zeros((3, 3, 3))
        mat_h1[1, 0] = [1, param.eta_s*param.rho, 0]
        mat_h1[1, 1] = [param.eta_s*param.rho, param.eta_s**2, 0]
        mat_h1[2, 2, 2] = param.eta_y**2

        npt.assert_array_equal(param.mat_k0, mat_k0)
        npt.assert_array_equal(param.mat_k1, mat_k1)
        npt.assert_array_equal(param.mat_h0, mat_h0)
        npt.assert_array_equal(param.mat_h1, mat_h1)

    def test_update(self):
        """Test update."""

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

        param = CentTendParam(riskfree=riskfree,
                              lmbd=lmbd, lmbd_s=lmbd_s, lmbd_y=lmbd_y,
                              mean_v=mean_v, kappa_s=kappa_s, kappa_y=kappa_y,
                              eta_s=eta_s, eta_y=eta_y, rho=rho)

        mean_v, kappa_s, kappa_y= .6, 1.7, .6
        eta_s, eta_y, rho, lmbd = .2, .02, -.6, .05
        theta = np.array([mean_v, kappa_s, kappa_y, eta_s, eta_y, rho, lmbd])
        param.update(theta=theta, measure='Q')

        kappa_sq = kappa_s - lmbd_s * eta_s
        kappa_yq = kappa_y - lmbd_y * eta_y
        scale = kappa_s / kappa_sq
        mean_vq = mean_v * kappa_y / kappa_yq * scale
        eta_yq = eta_y * scale**.5

        self.assertEqual(param.measure, 'Q')
        self.assertEqual(param.riskfree, riskfree)
        self.assertEqual(param.lmbd, 0)
        self.assertEqual(param.lmbd_s, lmbd_s)
        self.assertEqual(param.lmbd_y, lmbd_y)
        self.assertAlmostEqual(param.mean_v, mean_vq)
        self.assertEqual(param.kappa_s, kappa_sq)
        self.assertEqual(param.kappa_y, kappa_yq)
        self.assertEqual(param.eta_s, eta_s)
        self.assertEqual(param.eta_y, eta_yq)
        self.assertEqual(param.rho, rho)
        self.assertTrue(param.is_valid())

        mat_k0 = [riskfree, 0., kappa_yq * mean_vq]
        mat_k1 = [[0, -.5, 0],
                  [0, -kappa_sq, kappa_sq],
                  [0, 0, -kappa_yq]]

        npt.assert_array_almost_equal(param.mat_k0, mat_k0)
        npt.assert_array_equal(param.mat_k1, mat_k1)

    def test_get_theta(self):
        """Test get theta."""

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

        param = CentTendParam(riskfree=riskfree,
                              lmbd=lmbd, lmbd_s=lmbd_s, lmbd_y=lmbd_y,
                              mean_v=mean_v, kappa_s=kappa_s, kappa_y=kappa_y,
                              eta_s=eta_s, eta_y=eta_y, rho=rho, measure='P')

        theta = [mean_v, kappa_s, kappa_y, eta_s, eta_y,
                 rho, lmbd, lmbd_s, lmbd_y]
        theta_vol = np.concatenate((theta[:5], theta[-2:]))

        npt.assert_array_equal(param.get_theta(), theta)
        npt.assert_array_equal(param.get_theta(subset='all'), theta)
        npt.assert_array_equal(param.get_theta(subset='all', measure='PQ'),
                               theta)
        npt.assert_array_equal(param.get_theta(subset='all', measure='P'),
                               theta[:-2])
        npt.assert_array_equal(param.get_theta(subset='all', measure='Q'),
                               theta[:-2])
        npt.assert_array_equal(param.get_theta(subset='vol'), theta_vol)
        npt.assert_array_equal(param.get_theta(subset='vol', measure='PQ'),
                               theta_vol)
        npt.assert_array_equal(param.get_theta(subset='vol', measure='P'),
                               theta_vol[:-2])
        npt.assert_array_equal(param.get_theta(subset='vol', measure='Q'),
                               theta_vol[:-2])

        theta = np.arange(9)
        param.update(theta=theta)
        theta_vol = np.ones(5) * 2
        param.update(theta=theta_vol, subset='vol', measure='P')
        theta[:5] = theta_vol
        npt.assert_array_equal(param.get_theta(), theta)

    def test_bounds(self):
        """Test bounds."""

        param = CentTendParam()
        self.assertEqual(len(param.get_bounds()), 9)
        self.assertEqual(len(param.get_bounds(subset='all')), 9)
        self.assertEqual(len(param.get_bounds(subset='all', measure='PQ')), 9)
        self.assertEqual(len(param.get_bounds(subset='all', measure='P')), 7)
        self.assertEqual(len(param.get_bounds(subset='all', measure='Q')), 7)
        self.assertEqual(len(param.get_bounds(subset='vol')), 7)
        self.assertEqual(len(param.get_bounds(subset='vol', measure='PQ')), 7)
        self.assertEqual(len(param.get_bounds(subset='vol', measure='P')), 5)
        self.assertEqual(len(param.get_bounds(subset='vol', measure='Q')), 5)

    def test_validity(self):
        """Test validity."""

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

        param = CentTendParam(riskfree=riskfree,
                              lmbd=lmbd, lmbd_s=lmbd_s, lmbd_y=lmbd_y,
                              mean_v=mean_v, kappa_s=kappa_s, kappa_y=kappa_y,
                              eta_s=eta_s, eta_y=eta_y, rho=rho, measure='P')

        self.assertTrue(param.is_valid())

        param = CentTendParam(riskfree=riskfree, lmbd=lmbd,
                              mean_v=-mean_v, kappa_s=kappa_s, kappa_y=kappa_y,
                              eta_s=eta_s, eta_y=eta_y, rho=rho)

        self.assertFalse(param.is_valid())

        param = CentTendParam(riskfree=riskfree, lmbd=lmbd,
                              mean_v=mean_v, kappa_s=-kappa_s, kappa_y=kappa_y,
                              eta_s=eta_s, eta_y=eta_y, rho=rho)

        self.assertFalse(param.is_valid())

        param = CentTendParam(riskfree=riskfree, lmbd=lmbd,
                              mean_v=mean_v, kappa_s=kappa_s, kappa_y=-kappa_y,
                              eta_s=eta_s, eta_y=eta_y, rho=rho)

        self.assertFalse(param.is_valid())

        param = CentTendParam(riskfree=riskfree, lmbd=lmbd,
                              mean_v=mean_v, kappa_s=kappa_s, kappa_y=kappa_y,
                              eta_s=-eta_s, eta_y=eta_y, rho=rho)

        self.assertFalse(param.is_valid())

        param = CentTendParam(riskfree=riskfree, lmbd=lmbd,
                              mean_v=mean_v, kappa_s=kappa_s, kappa_y=kappa_y,
                              eta_s=eta_s, eta_y=-eta_y, rho=rho)

        self.assertFalse(param.is_valid())


if __name__ == '__main__':

    ut.main()
