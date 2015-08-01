#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Test suite for GBM parameter class.

"""
from __future__ import print_function, division

import unittest as ut
import numpy as np

from diffusions import GBMparam


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

        self.assertTrue(param.is_valid())
        param = GBMparam(mean, -sigma)
        self.assertFalse(param.is_valid())


if __name__ == '__main__':
    ut.main()
