#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Test suite for realized moments of Heston.

"""
from __future__ import print_function, division

import unittest as ut
import numpy as np
import numpy.testing as npt

from diffusions import Heston, HestonParam


class RealizedMomentsHestonTestCase(ut.TestCase):
    """Test realized moments for Heston."""

    def test_heston_depvar(self):
        """Test dependent variables of Heston model."""

        riskfree = 0.
        lmbd, mean_v, kappa, eta, rho = .01, .2, 1.5, .2**.5, -.5
        lmbd_v = .2
        param = HestonParam(riskfree=riskfree, lmbd=lmbd, mean_v=mean_v,
                            kappa=kappa, eta=eta, rho=rho, lmbd_v=lmbd_v)
        heston = Heston(param)
        heston.nsub = 2

        nperiods = 5
        ret = np.ones(nperiods) * (lmbd - .5) * mean_v
        rvar = np.ones(nperiods) * mean_v
        data = np.vstack([ret, rvar])
        depvar = heston.realized_depvar(data)

        # Test shape of dependent variables
        self.assertEqual(depvar.shape, (nperiods, 3 * 4))

    def test_heston_var_instr(self):
        """Test realized moments with variable instruments of Heston model."""

        riskfree = 0.
        lmbd, mean_v, kappa, eta, rho = .01, .2, 1.5, .2**.5, -.5
        lmbd_v = .2
        param = HestonParam(riskfree=riskfree, lmbd=lmbd, mean_v=mean_v,
                            kappa=kappa, eta=eta, rho=rho, lmbd_v=lmbd_v)
        heston = Heston(param)
        heston.nsub = 2
        nmoms = 4

        nperiods = 5
        ret = np.ones(nperiods) * (lmbd - .5) * mean_v
        rvar = np.ones(nperiods) * mean_v
        data = np.vstack([ret, rvar])
        instrlag = 2
        theta = param.get_theta(subset='all', measure='P')

        instr_data = np.vstack([rvar, rvar**2])
        ninstr = instr_data.shape[0]

        mom, dmom = heston.integrated_mom(theta, instr_data=instr_data,
                                          instr_choice='var',
                                          data=data, instrlag=instrlag)
        nmoms_all = nmoms * (ninstr*instrlag + 1)
        mom_shape = (nperiods - instrlag, nmoms_all)

        # Test the shape of moment functions
        self.assertEqual(mom.shape, mom_shape)
        self.assertIsNone(dmom)

    def test_const_instr(self):
        """Test constant instrument of Heston model."""

        riskfree = 0.
        lmbd, mean_v, kappa, eta, rho = .01, .2, 1.5, .2**.5, -.5
        lmbd_v = .2
        param = HestonParam(riskfree=riskfree, lmbd=lmbd, mean_v=mean_v,
                            kappa=kappa, eta=eta, rho=rho, lmbd_v=lmbd_v)
        heston = Heston(param)
        heston.nsub = 2
        nmoms = 4

        nperiods = 5
        ret = np.ones(nperiods) * (lmbd - .5) * mean_v
        rvar = np.ones(nperiods) * mean_v
        data = np.vstack([ret, rvar])
        instrlag = 2
        theta = param.get_theta(subset='all', measure='P')

        depvar = heston.realized_depvar(data)

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

    def test_vol_p(self):
        """Test vol P realized moments of Heston model."""

        riskfree = 0.
        lmbd, mean_v, kappa, eta, rho = .01, .2, 1.5, .2**.5, -.5
        lmbd_v = .2
        param = HestonParam(riskfree=riskfree, lmbd=lmbd, mean_v=mean_v,
                            kappa=kappa, eta=eta, rho=rho, lmbd_v=lmbd_v)
        heston = Heston(param)
        heston.nsub = 2
        nmoms = 4

        nperiods = 5
        ret = np.ones(nperiods) * (lmbd - .5) * mean_v
        rvar = np.ones(nperiods) * mean_v
        data = np.vstack([ret, rvar])
        instrlag = 2
        theta = param.get_theta(subset='all', measure='P')

        depvar = heston.realized_depvar(data)

        aggh = 2
        means = [heston.mean_vol(param, aggh),
                 heston.mean_vol2(param, aggh),
                 heston.mean_ret(param, aggh),
                 heston.mean_cross(param, aggh)]

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

    def test_heston_relized_mom(self):
        """Test realized moments of Heston model."""

        riskfree = 0.
        lmbd, mean_v, kappa, eta, rho = .01, .2, 1.5, .2**.5, -.5
        lmbd_v = .2
        param = HestonParam(riskfree=riskfree, lmbd=lmbd, mean_v=mean_v,
                            kappa=kappa, eta=eta, rho=rho, lmbd_v=lmbd_v)
        heston = Heston(param)
        heston.nsub = 2
        nmoms = 4

        nperiods = 5
        ret = np.ones(nperiods) * (lmbd - .5) * mean_v
        rvar = np.ones(nperiods) * mean_v
        data = np.vstack([ret, rvar])
        instrlag = 2
        theta = param.get_theta(subset='all', measure='P')

        depvar = heston.realized_depvar(data)

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
                                          instr_choice='const',
                                          aggh=[aggh, aggh],
                                          data=[data, data], instrlag=instrlag)
        nmoms = 2
        mom_shape = (nperiods - instrlag, nmoms*2)

        # Test the shape of moment functions
        self.assertEqual(mom.shape, mom_shape)

        subset_sl = slice(2)
        aggh = 10
        means = [heston.mean_vol(param, aggh),
                 heston.mean_vol2(param, aggh),
                 heston.mean_ret(param, aggh),
                 heston.mean_cross(param, aggh)]
        depvar = np.ones((nperiods - instrlag, 4)) * means
        depvar = np.tile(depvar, 3)
        error_q = depvar.dot(heston.mat_a(param, subset_sl).T) \
                - heston.realized_const(param, aggh, subset_sl)

        param = HestonParam(riskfree=riskfree, lmbd=lmbd, mean_v=mean_v,
                            kappa=kappa, eta=eta, rho=rho, lmbd_v=lmbd_v)

        means = [heston.mean_vol(param, aggh),
                 heston.mean_vol2(param, aggh),
                 heston.mean_ret(param, aggh),
                 heston.mean_cross(param, aggh)]
        depvar = np.ones((nperiods - instrlag, 4)) * means
        depvar = np.tile(depvar, 3)
        error_p = depvar.dot(heston.mat_a(param, subset_sl).T) \
                - heston.realized_const(param, aggh, subset_sl)

        npt.assert_array_almost_equal(np.hstack((error_p, error_q)),
                                      np.zeros(mom_shape))

        param = HestonParam(riskfree=riskfree, lmbd=lmbd, mean_v=mean_v,
                            kappa=kappa, eta=eta, rho=rho, lmbd_v=2*lmbd_v)
        heston = Heston(param)
        theta = param.get_theta(subset=subset, measure=measure)
        mom2, dmom = heston.integrated_mom(theta, subset=subset,
                                          measure=measure,
                                          instr_choice='const',
                                          aggh=[aggh, aggh],
                                          data=[data, data], instrlag=instrlag)

        self.assertFalse(np.allclose(mom, mom2))

    def test_heston_relized_mom_all(self):
        """Test realized moments of Heston model."""

        riskfree = 0.
        lmbd, mean_v, kappa, eta, rho = .01, .2, 1.5, .2**.5, -.5
        lmbd_v = .2

        nperiods = 5
        instrlag = 2
        ret = np.ones(nperiods) * (lmbd - .5) * mean_v
        rvar = np.ones(nperiods) * mean_v
        data = np.vstack([ret, rvar])

        subset = 'all'
        measure = 'Q'
        param = HestonParam(riskfree=riskfree, lmbd=lmbd, mean_v=mean_v,
                            kappa=kappa, eta=eta, rho=rho, lmbd_v=lmbd_v)
        heston = Heston(param)
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
        param = HestonParam(riskfree=riskfree, lmbd=lmbd, mean_v=mean_v,
                            kappa=kappa, eta=eta, rho=rho, lmbd_v=lmbd_v)
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
        param = HestonParam(riskfree=riskfree, lmbd=lmbd, mean_v=mean_v,
                            kappa=kappa, eta=eta, rho=rho, lmbd_v=lmbd_v)
        theta = param.get_theta(subset=subset, measure=measure)
        mom, dmom = heston.integrated_mom(theta, subset=subset,
                                          measure=measure,
                                          instr_choice='const',
                                          aggh=[aggh, aggh],
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
        error_q = depvar.dot(heston.mat_a(param, subset_sl).T) \
                - heston.realized_const(param, aggh, subset_sl)

        param = HestonParam(riskfree=riskfree, lmbd=lmbd, mean_v=mean_v,
                            kappa=kappa, eta=eta, rho=rho, lmbd_v=lmbd_v)
        means = [heston.mean_vol(param, aggh),
                 heston.mean_vol2(param, aggh),
                 heston.mean_ret(param, aggh),
                 heston.mean_cross(param, aggh)]
        depvar = np.ones((nperiods - instrlag, 4)) * means
        depvar = np.tile(depvar, 3)
        error_p = depvar.dot(heston.mat_a(param, subset_sl).T) \
                - heston.realized_const(param, aggh, subset_sl)

        npt.assert_array_almost_equal(np.hstack((error_p, error_q)),
                                      np.zeros(mom_shape))

    def test_heston_coefs(self):
        """Test coefficients in descretization of Heston model.

        """
        riskfree, lmbd, mean_v, kappa, eta, rho = 0., .01, .2, 1.5, .2**.5, -.5
        param = HestonParam(riskfree=riskfree, lmbd=lmbd,
                            mean_v=mean_v, kappa=kappa,
                            eta=eta, rho=rho)
        heston = Heston(param)
        heston.nsub = 10
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


if __name__ == '__main__':

    ut.main()
