#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Test suite for realized moments of Central Tendency.

"""
from __future__ import print_function, division

import unittest as ut
import numpy as np
import numpy.testing as npt

from diffusions import CentTend, CentTendParam


class RealizedMomentsCTTestCase(ut.TestCase):
    """Test realized moments for CT."""

    def test_ct_depvar(self):
        """Test dependent varibales of Central Tendency model."""

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
        centtend.nsub = 2

        nperiods = 10
        ret = np.arange(nperiods)
        rvar = ret ** 2
        data = np.vstack([ret, rvar])
        depvar = centtend.realized_depvar(data)

        # Test shape of dependent variables
        self.assertEqual(depvar.shape, (nperiods, 6 * 4))

    def test_ct_var_instr(self):
        """Test variable instruments of Central Tendency model."""

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
        centtend.nsub = 2
        nmoms = 4

        nperiods = 10
        ret = np.arange(nperiods)
        rvar = ret ** 2
        data = np.vstack([ret, rvar])
        instrlag = 2

        instr_data = np.vstack([rvar, rvar**2])
        ninstr = instr_data.shape[0]

        mom, dmom = centtend.integrated_mom(param.get_theta(),
                                            instr_data=instr_data,
                                            instr_choice='var',
                                            data=data, instrlag=instrlag)
        nmoms_all = nmoms * (ninstr*instrlag + 1)
        mom_shape = (nperiods - instrlag, nmoms_all)

        # Test the shape of moment functions
        self.assertEqual(mom.shape, mom_shape)
        self.assertIsNone(dmom)

    def test_const_instr(self):
        """Test constant instrument of Central Tendency model."""

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
        centtend.nsub = 2
        nmoms = 4

        nperiods = 10
        ret = np.arange(nperiods)
        rvar = ret ** 2
        data = np.vstack([ret, rvar])
        instrlag = 2
        depvar = centtend.realized_depvar(data)

        mom, dmom = centtend.integrated_mom(param.get_theta(),
                                            instr_choice='const',
                                            data=data, instrlag=instrlag)
        nmoms_all = nmoms
        mom_shape = (nperiods - instrlag, nmoms_all)

        aggh = 2
        means = [centtend.mean_vol(param, aggh),
                 centtend.mean_vol2(param, aggh),
                 centtend.mean_ret(param, aggh),
                 centtend.mean_cross(param, aggh)]

        depvar = np.ones((nperiods - instrlag, 4)) * means
        depvar = np.tile(depvar, 6)
        error = depvar.dot(centtend.mat_a(param, None).T) \
                - centtend.realized_const(param, aggh, None)

        npt.assert_array_almost_equal(error, np.zeros(mom_shape))
        # Test the shape of moment functions
        self.assertEqual(mom.shape, mom_shape)
        self.assertIsNone(dmom)

    def test_vol_p(self):
        """Test vol P realized moments of Central Tendency model."""

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
        centtend.nsub = 2
        nmoms = 4

        nperiods = 10
        ret = np.arange(nperiods)
        rvar = ret ** 2
        data = np.vstack([ret, rvar])
        instrlag = 2
        depvar = centtend.realized_depvar(data)

        aggh = 2
        means = [centtend.mean_vol(param, aggh),
                 centtend.mean_vol2(param, aggh),
                 centtend.mean_ret(param, aggh),
                 centtend.mean_cross(param, aggh)]

        subset = 'vol'
        mom, dmom = centtend.integrated_mom(param.get_theta(subset=subset),
                                            subset=subset,
                                            instr_choice='const',
                                            data=data, instrlag=instrlag)
        nmoms = 2
        mom_shape = (nperiods - instrlag, nmoms)

        # Test the shape of moment functions
        self.assertEqual(mom.shape, mom_shape)

        depvar = np.ones((nperiods - instrlag, 4)) * means
        depvar = np.tile(depvar, 6)
        subset_sl = slice(2)
        error = depvar.dot(centtend.mat_a(param, subset_sl).T) \
                - centtend.realized_const(param, aggh, subset_sl)

        npt.assert_array_almost_equal(error, np.zeros(mom_shape))

    def test_ct_relized_mom(self):
        """Test realized moments of Central Tendency model."""

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

        centtend = CentTend(param)
        centtend.nsub = 2
        nmoms = 4

        nperiods = 10
        ret = np.arange(nperiods)
        rvar = ret ** 2
        data = np.vstack([ret, rvar])
        instrlag = 2

        subset = 'vol'
        measure = 'Q'
        theta = param.get_theta(subset=subset, measure=measure)
        mom, dmom = centtend.integrated_mom(theta, subset=subset,
                                            measure=measure,
                                            instr_choice='const',
                                            data=data, instrlag=instrlag)
        nmoms = 2
        mom_shape = (nperiods - instrlag, nmoms)

        # Test the shape of moment functions
        self.assertEqual(mom.shape, mom_shape)

        subset_sl = slice(2)
        aggh = 2
        means = [centtend.mean_vol(param, aggh),
                 centtend.mean_vol2(param, aggh),
                 centtend.mean_ret(param, aggh),
                 centtend.mean_cross(param, aggh)]
        depvar = np.ones((nperiods - instrlag, 4)) * means
        depvar = np.tile(depvar, 6)
        error = depvar.dot(centtend.mat_a(param, subset_sl).T) \
                - centtend.realized_const(param, aggh, subset_sl)

        npt.assert_array_almost_equal(error, np.zeros(mom_shape))

        subset = 'vol'
        measure = 'P'
        theta = param.get_theta(subset=subset, measure=measure)
        mom, dmom = centtend.integrated_mom(theta, subset=subset,
                                            measure=measure,
                                            instr_choice='const',
                                            data=data, instrlag=instrlag)
        nmoms = 2
        mom_shape = (nperiods - instrlag, nmoms)

        # Test the shape of moment functions
        self.assertEqual(mom.shape, mom_shape)

        subset_sl = slice(2)
        aggh = 2
        means = [centtend.mean_vol(param, aggh),
                 centtend.mean_vol2(param, aggh),
                 centtend.mean_ret(param, aggh),
                 centtend.mean_cross(param, aggh)]
        depvar = np.ones((nperiods - instrlag, 4)) * means
        depvar = np.tile(depvar, 6)
        error = depvar.dot(centtend.mat_a(param, subset_sl).T) \
                - centtend.realized_const(param, aggh, subset_sl)

        npt.assert_array_almost_equal(error, np.zeros(mom_shape))

        param = CentTendParam(riskfree=riskfree, lmbd=lmbd,
                              mean_v=mean_v, kappa_s=kappa_s, kappa_y=kappa_y,
                              eta_s=eta_s, eta_y=eta_y, rho=rho)
        centtend = CentTend(param)
        subset = 'vol'
        measure = 'PQ'
        theta = param.get_theta(subset=subset, measure=measure)
        mom, dmom = centtend.integrated_mom(theta, subset=subset,
                                            measure=measure,
                                            instr_choice='const',
                                            aggh=[aggh, aggh],
                                            data=[data, data],
                                            instrlag=instrlag)
        nmoms = 2
        mom_shape = (nperiods - instrlag, nmoms*2)

        # Test the shape of moment functions
        self.assertEqual(mom.shape, mom_shape)

        subset_sl = slice(2)
        aggh = 10
        means = [centtend.mean_vol(param, aggh),
                 centtend.mean_vol2(param, aggh),
                 centtend.mean_ret(param, aggh),
                 centtend.mean_cross(param, aggh)]
        depvar = np.ones((nperiods - instrlag, 4)) * means
        depvar = np.tile(depvar, 6)
        error_q = depvar.dot(centtend.mat_a(param, subset_sl).T) \
                - centtend.realized_const(param, aggh, subset_sl)

        param = CentTendParam(riskfree=riskfree, lmbd=lmbd,
                              mean_v=mean_v, kappa_s=kappa_s, kappa_y=kappa_y,
                              eta_s=eta_s, eta_y=eta_y, rho=rho)

        means = [centtend.mean_vol(param, aggh),
                 centtend.mean_vol2(param, aggh),
                 centtend.mean_ret(param, aggh),
                 centtend.mean_cross(param, aggh)]
        depvar = np.ones((nperiods - instrlag, 4)) * means
        depvar = np.tile(depvar, 6)
        error_p = depvar.dot(centtend.mat_a(param, subset_sl).T) \
                - centtend.realized_const(param, aggh, subset_sl)

        npt.assert_array_almost_equal(np.hstack((error_p, error_q)),
                                      np.zeros(mom_shape))

        param = CentTendParam(riskfree=riskfree,
                              lmbd=lmbd, lmbd_s=2*lmbd_s, lmbd_y=3*lmbd_y,
                              mean_v=mean_v, kappa_s=kappa_s, kappa_y=kappa_y,
                              eta_s=eta_s, eta_y=eta_y, rho=rho)

        centtend = CentTend(param)
        theta = param.get_theta(subset=subset, measure=measure)
        mom2, dmom = centtend.integrated_mom(theta, subset=subset,
                                             measure=measure,
                                             instr_choice='const',
                                             aggh=[aggh, aggh],
                                             data=[data, data],
                                             instrlag=instrlag)

        self.assertFalse(np.allclose(mom, mom2))

    def test_ct_relized_mom_all(self):
        """Test realized moments of Central Tendency model."""

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

        centtend = CentTend(param)
        centtend.nsub = 2
        nmoms = 4

        nperiods = 10
        ret = np.arange(nperiods)
        rvar = ret ** 2
        data = np.vstack([ret, rvar])
        instrlag = 2

        subset = 'all'
        measure = 'Q'
        param = CentTendParam(riskfree=riskfree,
                              lmbd=lmbd, lmbd_s=lmbd_s, lmbd_y=lmbd_y,
                              mean_v=mean_v, kappa_s=kappa_s, kappa_y=kappa_y,
                              eta_s=eta_s, eta_y=eta_y, rho=rho)
        theta = param.get_theta(subset=subset, measure=measure)
        mom, dmom = centtend.integrated_mom(theta, subset=subset,
                                            measure=measure,
                                            instr_choice='const',
                                            data=data, instrlag=instrlag)
        nmoms = 4
        mom_shape = (nperiods - instrlag, nmoms)

        # Test the shape of moment functions
        self.assertEqual(mom.shape, mom_shape)

        subset_sl = None
        aggh = 2
        means = [centtend.mean_vol(param, aggh),
                 centtend.mean_vol2(param, aggh),
                 centtend.mean_ret(param, aggh),
                 centtend.mean_cross(param, aggh)]
        depvar = np.ones((nperiods - instrlag, 4)) * means
        depvar = np.tile(depvar, 6)
        error = depvar.dot(centtend.mat_a(param, subset_sl).T) \
                - centtend.realized_const(param, aggh, subset_sl)

        npt.assert_array_almost_equal(error, np.zeros(mom_shape))

        subset = 'all'
        measure = 'P'
        param = CentTendParam(riskfree=riskfree,
                              lmbd=lmbd, lmbd_s=lmbd_s, lmbd_y=lmbd_y,
                              mean_v=mean_v, kappa_s=kappa_s, kappa_y=kappa_y,
                              eta_s=eta_s, eta_y=eta_y, rho=rho)
        theta = param.get_theta(subset=subset, measure=measure)
        mom, dmom = centtend.integrated_mom(theta, subset=subset,
                                            measure=measure,
                                            instr_choice='const',
                                            data=data, instrlag=instrlag)
        nmoms = 4
        mom_shape = (nperiods - instrlag, nmoms)

        # Test the shape of moment functions
        self.assertEqual(mom.shape, mom_shape)

        subset_sl = None
        aggh = 2
        means = [centtend.mean_vol(param, aggh),
                 centtend.mean_vol2(param, aggh),
                 centtend.mean_ret(param, aggh),
                 centtend.mean_cross(param, aggh)]
        depvar = np.ones((nperiods - instrlag, 4)) * means
        depvar = np.tile(depvar, 6)
        error = depvar.dot(centtend.mat_a(param, subset_sl).T) \
                - centtend.realized_const(param, aggh, subset_sl)

        npt.assert_array_almost_equal(error, np.zeros(mom_shape))

        subset = 'all'
        measure = 'PQ'
        param = CentTendParam(riskfree=riskfree,
                              lmbd=lmbd, lmbd_s=lmbd_s, lmbd_y=lmbd_y,
                              mean_v=mean_v, kappa_s=kappa_s, kappa_y=kappa_y,
                              eta_s=eta_s, eta_y=eta_y, rho=rho)
        theta = param.get_theta(subset=subset, measure=measure)
        mom, dmom = centtend.integrated_mom(theta, subset=subset,
                                            measure=measure,
                                            instr_choice='const',
                                            aggh=[aggh, aggh],
                                            data=[data, data],
                                            instrlag=instrlag)
        nmoms = 4
        mom_shape = (nperiods - instrlag, nmoms*2)

        # Test the shape of moment functions
        self.assertEqual(mom.shape, mom_shape)

        subset_sl = None
        aggh = 2
        means = [centtend.mean_vol(param, aggh),
                 centtend.mean_vol2(param, aggh),
                 centtend.mean_ret(param, aggh),
                 centtend.mean_cross(param, aggh)]
        depvar = np.ones((nperiods - instrlag, 4)) * means
        depvar = np.tile(depvar, 6)
        error_q = depvar.dot(centtend.mat_a(param, subset_sl).T) \
                - centtend.realized_const(param, aggh, subset_sl)

        param = CentTendParam(riskfree=riskfree, lmbd=lmbd,
                              mean_v=mean_v, kappa_s=kappa_s, kappa_y=kappa_y,
                              eta_s=eta_s, eta_y=eta_y, rho=rho)
        means = [centtend.mean_vol(param, aggh),
                 centtend.mean_vol2(param, aggh),
                 centtend.mean_ret(param, aggh),
                 centtend.mean_cross(param, aggh)]
        depvar = np.ones((nperiods - instrlag, 4)) * means
        depvar = np.tile(depvar, 6)
        error_p = depvar.dot(centtend.mat_a(param, subset_sl).T) \
                - centtend.realized_const(param, aggh, subset_sl)

        npt.assert_array_almost_equal(np.hstack((error_p, error_q)),
                                      np.zeros(mom_shape))

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
        centtend.nsub = 10
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
