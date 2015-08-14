#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Test suite for realized moments of GBM.

"""
from __future__ import print_function, division

import unittest as ut
import numpy as np

from diffusions import GBM, GBMparam


class RealizedMomentsGBMTestCase(ut.TestCase):
    """Test realized moments for GBM."""

    def test_gbm_relized_mom(self):
        """Test realized moments of GBM model."""
        mean, sigma = 1.5, .2
        param = GBMparam(mean, sigma)
        gbm = GBM(param)
        gbm.nsub = 2

        nperiods = 10
        data = np.ones((2, nperiods))
        instrlag = 2

        depvar = gbm.realized_depvar(data)
        # Test shape of dependent variables
        self.assertEqual(depvar.shape, (3, nperiods))

        const = gbm.realized_const(param.get_theta())
        # Test shape of the intercept
        self.assertEqual(const.shape, (3, ))

        instr = gbm.instruments(data, instrlag=instrlag)
        ninstr = 1 + data.shape[0] * instrlag
        # Test shape of instrument matrix
        self.assertEqual(instr.shape, (ninstr, nperiods - instrlag))

        rmom, drmom = gbm.integrated_mom(param.get_theta(), data=data,
                                         instrlag=instrlag)
        nmoms = 3 * ninstr
        # Test shape of moments and gradients
        self.assertEqual(rmom.shape, (nperiods - instrlag, nmoms))
        self.assertEqual(drmom.shape, (nmoms, np.size(param.get_theta())))


if __name__ == '__main__':

    ut.main()
