#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Test suite for diffusions package.

"""

from __future__ import print_function, division

import unittest as ut
import numpy as np

from diffusions.gbm import GBM, GBMparam
from diffusions.generic_model import SDE
#from diffusions.helper_functions import nice_errors
from diffusions import nice_errors


class DiffusionsTestCase(ut.TestCase):
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

        errors = np.random.normal(size=(2, 3, 4))
        treated_errors = nice_errors(errors, -1)
        np.testing.assert_array_equal(treated_errors.mean(-1), 0)
        np.testing.assert_almost_equal(treated_errors.std(-1), np.ones((2, 3)))


if __name__ == '__main__':
    ut.main()
