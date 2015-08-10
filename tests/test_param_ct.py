#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Test suite for CT parameter class.

"""
from __future__ import print_function, division

import unittest as ut
import numpy as np
import numpy.testing as npt

from diffusions import CentTendParam


class SDEParameterTestCase(ut.TestCase):
    """Test parameter classes."""

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

        npt.assert_array_almost_equal(param.mat_k0, mat_k0)
        npt.assert_array_equal(param.mat_k1, mat_k1)
        npt.assert_array_equal(param.mat_h0, mat_h0)
        npt.assert_array_equal(param.mat_h1, mat_h1)

        param = CentTendParam(riskfree=riskfree, lmbd=lmbd,
                              mean_v=mean_v, kappa_s=kappa_s, kappa_y=kappa_y,
                              eta_s=eta_s, eta_y=eta_y, rho=rho)

        theta = np.array([mean_v, kappa_s, kappa_y, eta_s, eta_y, lmbd, rho])
        npt.assert_array_equal(param.get_theta(), theta)

        theta = np.ones(7)
        param = CentTendParam()
        param.update(theta=theta)
        npt.assert_array_equal(param.get_theta(), theta)

        mat_k0 = [param.riskfree, 0., param.kappa_y * param.mean_v]
        mat_k1 = [[0, param.lmbd - .5, 0],
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

        theta = np.arange(7)
        param.update(theta=theta)
        theta_vol = np.ones(5) * 2
        param.update(theta=theta_vol, subset='vol')
        theta[:5] = theta_vol
        npt.assert_array_equal(param.get_theta(), theta)

        self.assertEqual(len(param.get_bounds()), 7)
        self.assertEqual(len(param.get_bounds(subset='vol')), 5)

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
