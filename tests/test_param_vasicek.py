#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Test suite for Vasicek parameter class.

"""
from __future__ import print_function, division

import unittest as ut
import numpy as np

from diffusions import VasicekParam


class SDEParameterTestCase(ut.TestCase):
    """Test parameter classes."""

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

        self.assertTrue(param.is_valid())
        param = VasicekParam(mean, -kappa, eta)
        self.assertFalse(param.is_valid())
        param = VasicekParam(mean, kappa, -eta)
        self.assertFalse(param.is_valid())


if __name__ == '__main__':
    ut.main()
