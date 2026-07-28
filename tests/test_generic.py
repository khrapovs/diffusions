#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Test suite for generic classes."""

from __future__ import division, print_function

import unittest as ut

from affidiff import GBM, GBMparam


class GenericModelTestCase(ut.TestCase):
    """Test generic model."""

    def test_update_theta(self) -> None:
        """Test update of true parameter."""
        mean, sigma = 1.5, 0.2
        param = GBMparam(mean, sigma)
        gbm = GBM(param)
        param_new = GBMparam(2 * mean, 2 * sigma)
        gbm.update_theta(param_new)

        self.assertEqual(gbm.param, param_new)


if __name__ == "__main__":
    ut.main()
