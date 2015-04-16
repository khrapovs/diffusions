#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Test suite for diffusions package.

"""
from __future__ import print_function, division

import unittest as ut
import numpy as np

from diffusions import columnwise_prod, rolling_window, nice_errors


class HelperFunctionTestCase(ut.TestCase):
    """Test helper functions."""

    def test_columnwise_prod(self):
        """Test columnwise product."""
        left = np.arange(6).reshape((3, 2))
        right = np.arange(9).reshape((3, 3))
        prod = []
        for i in range(right.shape[1]):
            prod.append(left.T * right[:, i])
        prod = np.vstack(prod).T
        expected = columnwise_prod(left, right)

        np.testing.assert_array_equal(prod, expected)

    def test_rolling_window(self):
        """test riolling window apply."""

        mat = rolling_window(np.sum, np.ones(5), window=2)

        np.testing.assert_array_equal(mat, np.ones(4) * 2)
        mat = np.arange(10).reshape((2,5))
        mat = rolling_window(np.mean, mat, window=2)
        expect = np.array([[ 0.5,  1.5,  2.5,  3.5], [ 5.5,  6.5,  7.5,  8.5]])

        np.testing.assert_array_equal(mat, expect)

    def test_nice_errors(self):
        """Test nice errors function."""

        nvars, nobs, nsim = 2, 3, 11
        sim = 1
        size = (nobs, nsim, nvars)
        new_size = (nobs, nsim*2, nvars)
        errors = np.random.normal(size=size)
        treated_errors = nice_errors(errors, sim)

        self.assertEqual(treated_errors.shape, new_size)
        np.testing.assert_almost_equal(treated_errors.mean(sim), 0)
        np.testing.assert_almost_equal(treated_errors.std(sim),
                                       np.ones((nobs, nvars)))


if __name__ == '__main__':
    ut.main()
