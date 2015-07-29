#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Test suite for generic classes.

"""
from __future__ import print_function, division

import unittest as ut
import numpy as np

from diffusions import (GBM, GBMparam, Vasicek, VasicekParam,
                        CIR, CIRparam, Heston, HestonParam,
                        CentTend, CentTendParam)


class GenericModelTestCase(ut.TestCase):
    """Test generic model."""

    def test_update_theta(self):
        """Test update of true parameter."""

        mean, sigma = 1.5, .2
        param = GBMparam(mean, sigma)
        gbm = GBM(param)
        param_new = GBMparam(2*mean, 2*sigma)
        gbm.update_theta(param_new)

        self.assertEqual(gbm.theta_true, param_new)


if __name__ == '__main__':
    ut.main()
