#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Test suite for helper functions.

"""
from __future__ import print_function, division

import unittest as ut
import numpy as np
from statsmodels.tsa.tsatools import lagmat

from diffusions import (columnwise_prod, rolling_window, nice_errors,
                        poly_coef, instruments)


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
        """Test rolling window apply."""

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

    def test_poly_coef(self):
        """Test polynomial coefficients."""

        roots = [2, 3]
        coefs = [1, -np.sum(roots), np.prod(roots)]
        self.assertEqual(poly_coef(roots), coefs)

        roots = np.array([2, 3, 4])

        coefs = [1, -np.sum(roots), 0, -np.prod(roots)]
        coefs[2] = np.prod(roots[:2]) + np.prod(roots[1:]) \
            + np.prod(roots[[0, 2]])
        self.assertEqual(poly_coef(roots), coefs)

    def test_instruments(self):
        """Test instruments."""
        nperiods = 10
        instrlag = 2

        instrmnts = instruments(nobs=nperiods)
        np.testing.assert_array_equal(instrmnts, np.ones((nperiods, 1)))

        ninstr = 2
        data = np.arange(nperiods*ninstr).reshape((ninstr, nperiods))
        instrmnts = instruments(data=data, instr_choice='const')
        np.testing.assert_array_equal(instrmnts, np.ones((nperiods, 1)))

        instrmnts = instruments(data=data, instr_choice='var')
        expect = np.hstack([np.ones((nperiods, 1)), lagmat(data.T, maxlag=1)])
        np.testing.assert_array_equal(instrmnts, expect)

        instrmnts = instruments(data[0], instrlag=instrlag, instr_choice='var')
        ninstr = 1
        shape = (nperiods, ninstr*instrlag + 1)
        # Test the shape of instruments
        self.assertEqual(instrmnts.shape, shape)


if __name__ == '__main__':
    ut.main()
