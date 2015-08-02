#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Test suite for realized moments.

"""
from __future__ import print_function, division

import unittest as ut
import numpy as np
import numpy.testing as npt

from statsmodels.tsa.tsatools import lagmat

from diffusions import GBM, GBMparam
from diffusions import Heston, HestonParam
from diffusions import CentTend, CentTendParam


class RealizedMomentsTestCase(ut.TestCase):
    """Test realized moments."""

    def test_gbm_relized_mom(self):
        """Test realized moments of GBM model."""
        mean, sigma = 1.5, .2
        param = GBMparam(mean, sigma)
        gbm = GBM(param)
        gbm.interval = .5

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

    def test_heston_relized_mom(self):
        """Test realized moments of Heston model."""

        riskfree = 0.
        lmbd, mean_v, kappa, eta, rho = .01, .2, 1.5, .2**.5, -.5
        lmbd_v = .2
        param = HestonParam(riskfree=riskfree, lmbd=lmbd, mean_v=mean_v,
                            kappa=kappa, eta=eta, rho=rho, lmbd_v=lmbd_v)
        heston = Heston(param)
        heston.interval = .5
        nmoms = 4

        nperiods = 5
        ret = np.ones(nperiods) * (lmbd - .5) * mean_v
        rvar = np.ones(nperiods) * mean_v
        data = np.vstack([ret, rvar])
        instrlag = 2
        theta = param.get_theta(subset='all', measure='P')

        depvar = heston.realized_depvar(data)
        instr_data = np.vstack([rvar, rvar**2])
        ninstr = instr_data.shape[0]

        # Test shape of dependent variables
        self.assertEqual(depvar.shape, (nperiods, 3 * 4))

        mom, dmom = heston.integrated_mom(theta, instr_data=instr_data,
                                          instr_choice='var',
                                          data=data, instrlag=instrlag)
        nmoms_all = nmoms * (ninstr*instrlag + 1)
        mom_shape = (nperiods - instrlag, nmoms_all)

        # Test the shape of moment functions
        self.assertEqual(mom.shape, mom_shape)
        self.assertIsNone(dmom)

        mom, dmom = heston.integrated_mom(theta, instr_choice='const',
                                          data=data, instrlag=instrlag)
        nmoms_all = nmoms
        mom_shape = (nperiods - instrlag, nmoms_all)

        aggh = 2
        means = [heston.mean_vol(param, aggh),
                 heston.mean_vol2(param, aggh),
                 heston.mean_ret(param, aggh),
                 heston.mean_cross(param, aggh)]

        depvar = np.ones((nperiods - instrlag, 4)) * means
        depvar = np.tile(depvar, 3)
        error = depvar.dot(heston.mat_a(param, None).T) \
                - heston.realized_const(param, aggh, None)

        npt.assert_array_almost_equal(error, np.zeros(mom_shape))

        # Test the shape of moment functions
        self.assertEqual(mom.shape, mom_shape)

        subset = 'vol'
        theta = param.get_theta(subset=subset)
        mom, dmom = heston.integrated_mom(theta, subset=subset,
                                          instr_choice='const',
                                          data=data, instrlag=instrlag)
        nmoms = 2
        mom_shape = (nperiods - instrlag, nmoms)

        # Test the shape of moment functions
        self.assertEqual(mom.shape, mom_shape)

        depvar = np.ones((nperiods - instrlag, 4)) * means
        depvar = np.tile(depvar, 3)
        subset_sl = slice(2)
        error = depvar.dot(heston.mat_a(param, subset_sl).T) \
                - heston.realized_const(param, aggh, subset_sl)

        npt.assert_array_almost_equal(error, np.zeros(mom_shape))

        subset = 'vol'
        measure = 'Q'
        theta = param.get_theta(subset=subset, measure=measure)
        mom, dmom = heston.integrated_mom(theta, subset=subset,
                                          measure=measure,
                                          instr_choice='const',
                                          data=data, instrlag=instrlag)
        nmoms = 2
        mom_shape = (nperiods - instrlag, nmoms)

        # Test the shape of moment functions
        self.assertEqual(mom.shape, mom_shape)

        subset_sl = slice(2)
        aggh = 2
        means = [heston.mean_vol(param, aggh),
                 heston.mean_vol2(param, aggh),
                 heston.mean_ret(param, aggh),
                 heston.mean_cross(param, aggh)]
        depvar = np.ones((nperiods - instrlag, 4)) * means
        depvar = np.tile(depvar, 3)
        error = depvar.dot(heston.mat_a(param, subset_sl).T) \
                - heston.realized_const(param, aggh, subset_sl)

        npt.assert_array_almost_equal(error, np.zeros(mom_shape))

        subset = 'vol'
        measure = 'P'
        theta = param.get_theta(subset=subset, measure=measure)
        mom, dmom = heston.integrated_mom(theta, subset=subset,
                                          measure=measure,
                                          instr_choice='const',
                                          data=data, instrlag=instrlag)
        nmoms = 2
        mom_shape = (nperiods - instrlag, nmoms)

        # Test the shape of moment functions
        self.assertEqual(mom.shape, mom_shape)

        subset_sl = slice(2)
        aggh = 2
        means = [heston.mean_vol(param, aggh),
                 heston.mean_vol2(param, aggh),
                 heston.mean_ret(param, aggh),
                 heston.mean_cross(param, aggh)]
        depvar = np.ones((nperiods - instrlag, 4)) * means
        depvar = np.tile(depvar, 3)
        error = depvar.dot(heston.mat_a(param, subset_sl).T) \
                - heston.realized_const(param, aggh, subset_sl)

        npt.assert_array_almost_equal(error, np.zeros(mom_shape))

        param = HestonParam(riskfree=riskfree, lmbd=lmbd, mean_v=mean_v,
                            kappa=kappa, eta=eta, rho=rho, lmbd_v=lmbd_v)
        heston = Heston(param)
        subset = 'vol'
        measure = 'PQ'
        theta = param.get_theta(subset=subset, measure=measure)
        mom, dmom = heston.integrated_mom(theta, subset=subset,
                                          measure=measure,
                                          instr_choice='const', aggh=[1, 1],
                                          data=[data, data], instrlag=instrlag)
        nmoms = 2
        mom_shape = (nperiods - instrlag, nmoms*2)

        # Test the shape of moment functions
        self.assertEqual(mom.shape, mom_shape)

        subset_sl = slice(2)
        aggh = 2
        means = [heston.mean_vol(param, aggh),
                 heston.mean_vol2(param, aggh),
                 heston.mean_ret(param, aggh),
                 heston.mean_cross(param, aggh)]
        depvar = np.ones((nperiods - instrlag, 4)) * means
        depvar = np.tile(depvar, 3)
        error = depvar.dot(heston.mat_a(param, subset_sl).T) \
                - heston.realized_const(param, aggh, subset_sl)

        npt.assert_array_almost_equal(np.hstack((error, error)),
                                      np.zeros(mom_shape))

        param = HestonParam(riskfree=riskfree, lmbd=lmbd, mean_v=mean_v,
                            kappa=kappa, eta=eta, rho=rho, lmbd_v=2*lmbd_v)
        heston = Heston(param)
        theta = param.get_theta(subset=subset, measure=measure)
        mom2, dmom = heston.integrated_mom(theta, subset=subset,
                                          measure=measure,
                                          instr_choice='const', aggh=[1, 1],
                                          data=[data, data], instrlag=instrlag)

        self.assertFalse(np.allclose(mom, mom2))

        subset = 'all'
        measure = 'Q'
        theta = param.get_theta(subset=subset, measure=measure)
        mom, dmom = heston.integrated_mom(theta, subset=subset,
                                          measure=measure,
                                          instr_choice='const',
                                          data=data, instrlag=instrlag)
        nmoms = 4
        mom_shape = (nperiods - instrlag, nmoms)

        # Test the shape of moment functions
        self.assertEqual(mom.shape, mom_shape)

        subset_sl = None
        aggh = 2
        means = [heston.mean_vol(param, aggh),
                 heston.mean_vol2(param, aggh),
                 heston.mean_ret(param, aggh),
                 heston.mean_cross(param, aggh)]
        depvar = np.ones((nperiods - instrlag, 4)) * means
        depvar = np.tile(depvar, 3)
        error = depvar.dot(heston.mat_a(param, subset_sl).T) \
                - heston.realized_const(param, aggh, subset_sl)

        npt.assert_array_almost_equal(error, np.zeros(mom_shape))

        subset = 'all'
        measure = 'P'
        theta = param.get_theta(subset=subset, measure=measure)
        mom, dmom = heston.integrated_mom(theta, subset=subset,
                                          measure=measure,
                                          instr_choice='const',
                                          data=data, instrlag=instrlag)
        nmoms = 4
        mom_shape = (nperiods - instrlag, nmoms)

        # Test the shape of moment functions
        self.assertEqual(mom.shape, mom_shape)

        subset_sl = None
        aggh = 2
        means = [heston.mean_vol(param, aggh),
                 heston.mean_vol2(param, aggh),
                 heston.mean_ret(param, aggh),
                 heston.mean_cross(param, aggh)]
        depvar = np.ones((nperiods - instrlag, 4)) * means
        depvar = np.tile(depvar, 3)
        error = depvar.dot(heston.mat_a(param, subset_sl).T) \
                - heston.realized_const(param, aggh, subset_sl)

        npt.assert_array_almost_equal(error, np.zeros(mom_shape))

        subset = 'all'
        measure = 'PQ'
        theta = param.get_theta(subset=subset, measure=measure)
        mom, dmom = heston.integrated_mom(theta, subset=subset,
                                          measure=measure,
                                          instr_choice='const', aggh=[1, 1],
                                          data=[data, data], instrlag=instrlag)
        nmoms = 4
        mom_shape = (nperiods - instrlag, nmoms*2)

        # Test the shape of moment functions
        self.assertEqual(mom.shape, mom_shape)

        subset_sl = None
        aggh = 2
        means = [heston.mean_vol(param, aggh),
                 heston.mean_vol2(param, aggh),
                 heston.mean_ret(param, aggh),
                 heston.mean_cross(param, aggh)]
        depvar = np.ones((nperiods - instrlag, 4)) * means
        depvar = np.tile(depvar, 3)
        error = depvar.dot(heston.mat_a(param, subset_sl).T) \
                - heston.realized_const(param, aggh, subset_sl)

        npt.assert_array_almost_equal(np.hstack((error, error)),
                                      np.zeros(mom_shape))

    def test_heston_coefs(self):
        """Test coefficients in descretization of Heston model.

        """
        riskfree, lmbd, mean_v, kappa, eta, rho = 0., .01, .2, 1.5, .2**.5, -.5
        param = HestonParam(riskfree=riskfree, lmbd=lmbd,
                            mean_v=mean_v, kappa=kappa,
                            eta=eta, rho=rho)
        heston = Heston(param)
        heston.interval = .1
        aggh = 2

        self.assertIsInstance(heston.coef_big_a(param, aggh), float)
        self.assertIsInstance(heston.coef_small_a(param, aggh), float)
        self.assertIsInstance(heston.coef_big_c(param, aggh), float)
        self.assertIsInstance(heston.coef_small_c(param, aggh), float)

        self.assertEqual(heston.mat_a0(param, aggh).shape, (4, 4))
        self.assertEqual(heston.mat_a1(param, aggh).shape, (4, 4))
        self.assertEqual(heston.mat_a2(param, aggh).shape, (4, 4))

        self.assertEqual(heston.mat_a(param).shape, (4, 3*4))

        self.assertEqual(heston.realized_const(param, aggh).shape, (4, ))
        self.assertEqual(heston.realized_const(param, aggh)[2], 0)

        means = [heston.mean_vol(param, aggh),
                 heston.mean_vol2(param, aggh),
                 heston.mean_ret(param, aggh),
                 heston.mean_cross(param, aggh)]

        npt.assert_array_equal(heston.depvar_unc_mean(param, aggh), means)

        res = heston.mean_vol(param, aggh) \
            * (1 - heston.coef_big_a(param, 1))

        self.assertEqual(heston.realized_const(param, aggh)[0], res)

        res = heston.mean_vol2(param, aggh) \
            * (1 - heston.coef_big_a(param, 1)) \
            * (1 - heston.coef_big_a(param, 1)**2)

        self.assertEqual(heston.realized_const(param, aggh)[1], res)

        res = heston.mean_ret(param, aggh) \
            + heston.mean_vol(param, aggh) * (.5 - lmbd)

        self.assertEqual(heston.realized_const(param, aggh)[2], res)

        res = heston.mean_vol2(param, aggh) * (.5 - lmbd) \
            * (1 - heston.coef_big_a(param, 1)) \
            + heston.mean_cross(param, aggh) \
            * (1 - heston.coef_big_a(param, 1))

        self.assertAlmostEqual(heston.realized_const(param, aggh)[3], res)

    def test_ct_relized_mom(self):
        """Test realized moments of Central Tendency model."""
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

        centtend = CentTend(param)
        centtend.interval = .5
        nmoms = 4

        nperiods = 10
        ret = np.arange(nperiods)
        rvar = ret ** 2
        data = np.vstack([ret, rvar])
        instrlag = 2

        depvar = centtend.realized_depvar(data)
        instr_data = np.vstack([rvar, rvar**2])
        ninstr = instr_data.shape[0]

        # Test shape of dependent variables
        self.assertEqual(depvar.shape, (nperiods, 6 * 4))

        mom, dmom = centtend.integrated_mom(param.get_theta(),
                                            instr_data=instr_data,
                                            instr_choice='var',
                                            data=data, instrlag=instrlag)
        nmoms_all = nmoms * (ninstr*instrlag + 1)
        mom_shape = (nperiods - instrlag, nmoms_all)

        # Test the shape of moment functions
        self.assertEqual(mom.shape, mom_shape)
        self.assertIsNone(dmom)

        mom, dmom = centtend.integrated_mom(param.get_theta(),
                                            instr_choice='const',
                                            data=data, instrlag=instrlag)
        nmoms_all = nmoms
        mom_shape = (nperiods - instrlag, nmoms_all)

        # Test the shape of moment functions
        self.assertEqual(mom.shape, mom_shape)
        self.assertIsNone(dmom)

        subset = 'vol'
        mom, dmom = centtend.integrated_mom(param.get_theta(subset=subset),
                                            subset=subset,
                                            instr_choice='const',
                                            data=data, instrlag=instrlag)
        nmoms = 2
        mom_shape = (nperiods - instrlag, nmoms)

        # Test the shape of moment functions
        self.assertEqual(mom.shape, mom_shape)
        self.assertIsNone(dmom)

    def test_ct_coefs(self):
        """Test coefficients in descretization of CT model.

        """
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

        centtend = CentTend(param)
        centtend.interval = .1
        aggh = 2

        self.assertIsInstance(centtend.coef_big_as(param, aggh), float)
        self.assertIsInstance(centtend.coef_big_bs(param, aggh), float)
        self.assertIsInstance(centtend.coef_big_cs(param, aggh), float)
        self.assertIsInstance(centtend.coef_big_ay(param, aggh), float)
        self.assertIsInstance(centtend.coef_big_cy(param, aggh), float)
        self.assertIsInstance(centtend.coef_small_as(param, aggh), float)
        self.assertIsInstance(centtend.coef_small_bs(param, aggh), float)
        self.assertIsInstance(centtend.coef_small_cs(param, aggh), float)

        self.assertEqual(len(centtend.roots(param, aggh)), 5)
        roots = [centtend.coef_big_as(param, aggh),
                  centtend.coef_big_ay(param, aggh),
                  centtend.coef_big_as(param, aggh)**2,
                  centtend.coef_big_ay(param, aggh)**2,
                  centtend.coef_big_as(param, aggh)
                  * centtend.coef_big_ay(param, aggh)]
        self.assertEqual(centtend.roots(param, aggh), roots)

        self.assertEqual(len(centtend.depvar_unc_mean(param, aggh)), 4)

        self.assertEqual(centtend.mat_a0(param, aggh).shape, (4, 4))
        self.assertEqual(centtend.mat_a0(param, aggh)[1, 1], 1.)

        self.assertEqual(centtend.mat_a1(param, aggh).shape, (4, 4))
        expect = -np.sum(centtend.roots(param, aggh))
        self.assertEqual(centtend.mat_a1(param, aggh)[1, 1], expect)

        self.assertEqual(centtend.mat_a2(param, aggh).shape, (4, 4))

        self.assertEqual(centtend.mat_a3(param, aggh).shape, (4, 4))
        self.assertEqual(centtend.mat_a3(param, aggh)[0, 0], 1.)
        self.assertEqual(centtend.mat_a3(param, aggh)[3, 1], .5 - param.lmbd)

        self.assertEqual(centtend.mat_a4(param, aggh).shape, (4, 4))
        expect = -np.sum(centtend.roots(param, aggh)[:2])
        self.assertEqual(centtend.mat_a4(param, aggh)[0, 0], expect)
        self.assertEqual(centtend.mat_a4(param, aggh)[3, 1],
                         (.5 - param.lmbd) * expect)

        self.assertEqual(centtend.mat_a5(param, aggh).shape, (4, 4))
        expect = np.prod(centtend.roots(param, aggh)[:2])
        self.assertEqual(centtend.mat_a5(param, aggh)[0, 0], expect)
        self.assertEqual(centtend.mat_a5(param, aggh)[3, 1],
                         (.5 - param.lmbd) * expect)
        expect = -np.prod(centtend.roots(param, aggh))
        self.assertEqual(centtend.mat_a5(param, aggh)[1, 1], expect)
        self.assertEqual(centtend.mat_a5(param, aggh)[2, 2], 1.)
        self.assertEqual(centtend.mat_a5(param, aggh)[2, 0], .5 - param.lmbd)

        self.assertEqual(centtend.mat_a(param).shape, (4, 6*4))

        self.assertEqual(centtend.realized_const(param, aggh).shape, (4, ))
        self.assertEqual(centtend.realized_const(param, aggh)[2], 0)

        roots = [centtend.coef_big_as(param, 1),
                  centtend.coef_big_ay(param, 1),
                  centtend.coef_big_as(param, 1)**2,
                  centtend.coef_big_ay(param, 1)**2,
                  centtend.coef_big_as(param, 1)
                  * centtend.coef_big_ay(param, 1)]

        res = centtend.depvar_unc_mean(param, aggh)[0] \
            * (1 - roots[0]) * (1 - roots[1])

        self.assertAlmostEqual(centtend.realized_const(param, aggh)[0], res)

        res = centtend.depvar_unc_mean(param, aggh)[1] \
            * (1 - roots[0]) * (1 - roots[1]) * (1 - roots[2]) \
            * (1 - roots[3]) * (1 - roots[4])

        self.assertAlmostEqual(centtend.realized_const(param, aggh)[1], res)

        res = (centtend.depvar_unc_mean(param, aggh)[1] * (.5 - param.lmbd) \
            + centtend.depvar_unc_mean(param, aggh)[3]) \
            * (1 - roots[0]) * (1 - roots[1])

        self.assertAlmostEqual(centtend.realized_const(param, aggh)[3], res)


if __name__ == '__main__':
    ut.main()
