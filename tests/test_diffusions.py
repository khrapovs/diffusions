#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Test suite for diffusions package.

"""

from __future__ import print_function, division

import unittest as ut
import numpy as np

from diffusions import GBM, GBMparam
from diffusions import Vasicek, VasicekParam
from diffusions import CIR, CIRparam
from diffusions import Heston, HestonParam
from diffusions import (nice_errors, ajd_drift, ajd_diff, columnwise_prod,
                        rolling_window)


class SDEParameterTestCase(ut.TestCase):
    """Test SDE, GBM classes."""

    def test_gbmparam_class(self):
        """Test GBM parameter class."""

        mean, sigma = 1.5, .2
        param = GBMparam(mean, sigma)

        self.assertEqual(param.mean, mean)
        self.assertEqual(param.sigma, sigma)
        np.testing.assert_array_equal(param.get_theta(),
                                      np.array([mean, sigma]))
        np.testing.assert_array_equal(param.mat_k0,
                                      np.array([mean - sigma**2/2]))
        np.testing.assert_array_equal(param.mat_k1, np.array([[0]]))
        np.testing.assert_array_equal(param.mat_h0, np.array([[sigma**2]]))
        np.testing.assert_array_equal(param.mat_h1, np.array([[[0]]]))

    def test_hestonparam_class(self):
        """Test Heston parameter class."""

        riskfree = .01
        lmbd = .01
        mean_v = .5
        kappa = 1.5
        eta = .1
        rho = -.5
        param = HestonParam(riskfree=riskfree, lmbd=lmbd,
                            mean_v=mean_v, kappa=kappa,
                            eta=eta, rho=rho)

        self.assertEqual(param.riskfree, riskfree)
        self.assertEqual(param.lmbd, lmbd)
        self.assertEqual(param.mean_v, mean_v)
        self.assertEqual(param.kappa, kappa)
        self.assertEqual(param.eta, eta)
        self.assertEqual(param.rho, rho)

        theta = np.array([lmbd, mean_v, kappa, eta, rho])
        np.testing.assert_array_equal(param.get_theta(), theta)

        theta = np.ones(5)
        param = HestonParam()
        param.update(theta=theta)
        np.testing.assert_array_equal(param.get_theta(), theta)
        # TODO : test AJD representation


class HelperFunctionsTestCase(ut.TestCase):
    """Test helper functions."""

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

    def test_ajd_drift_gbm(self):
        """Test AJD drift function for GBM model."""

        mean, sigma = 1.5, .2
        param = GBMparam(mean, sigma)
        nvars, nsim = 1, 2
        size = (nsim, nvars)
        state = np.ones(size)
        drift = state * (mean - sigma**2/2)

        self.assertEqual(ajd_drift(state, param).shape, size)
        np.testing.assert_array_equal(ajd_drift(state, param), drift)

    def test_ajd_diff_gbm(self):
        """Test AJD diffusion function for GBM model."""

        mean, sigma = 1.5, .2
        param = GBMparam(mean, sigma)
        nvars, nsim = 1, 2
        size = (nsim, nvars)
        state = np.ones(size)
        diff = np.ones((nsim, nvars, nvars)) * sigma

        self.assertEqual(ajd_diff(state, param).shape, (nsim, nvars, nvars))
        np.testing.assert_array_equal(ajd_diff(state, param), diff)

    def test_ajd_drift_vasicek(self):
        """Test AJD drift function for Vasicek model."""

        mean, kappa, eta = 1.5, 1, .2
        param = VasicekParam(mean, kappa, eta)
        nvars, nsim = 1, 2
        size = (nsim, nvars)
        state = np.ones(size)
        drift = kappa * (mean - state)

        self.assertEqual(ajd_drift(state, param).shape, size)
        np.testing.assert_array_equal(ajd_drift(state, param), drift)

    def test_ajd_diff_vasicek(self):
        """Test AJD diffusion function for Vasicek model."""

        mean, kappa, eta = 1.5, 1, .2
        param = VasicekParam(mean, kappa, eta)
        nvars, nsim = 1, 2
        size = (nsim, nvars)
        state = np.ones(size)
        diff = np.ones((nsim, nvars, nvars)) * eta

        self.assertEqual(ajd_diff(state, param).shape, (nsim, nvars, nvars))
        np.testing.assert_array_equal(ajd_diff(state, param), diff)

    def test_ajd_drift_cir(self):
        """Test AJD drift function for CIR model."""

        mean, kappa, eta = 1.5, 1, .2
        param = CIRparam(mean, kappa, eta)
        nvars, nsim = 1, 2
        size = (nsim, nvars)
        state = np.ones(size)
        drift = kappa * (mean - state)

        self.assertEqual(ajd_drift(state, param).shape, size)
        np.testing.assert_array_equal(ajd_drift(state, param), drift)

    def test_ajd_diff_cir(self):
        """Test AJD diffusion function for CIR model."""

        mean, kappa, eta = 1.5, 1, .2
        param = CIRparam(mean, kappa, eta)
        nvars, nsim = 1, 2
        size = (nsim, nvars)
        state_val = 4
        state = np.ones(size)*state_val
        diff = eta * state_val**.5 * np.ones((nsim, nvars, nvars))

        self.assertEqual(ajd_diff(state, param).shape, (nsim, nvars, nvars))
        np.testing.assert_array_equal(ajd_diff(state, param), diff)

    def test_ajd_drift_heston(self):
        """Test AJD drift function for Heston model."""

        riskfree, lmbd, mean_v, kappa, eta, rho = 0., .01, .2, 1.5, .2, -.5
        param = HestonParam(riskfree=riskfree, lmbd=lmbd,
                            mean_v=mean_v, kappa=kappa,
                            eta=eta, rho=rho)
        nvars, nsim = 2, 3
        size = (nsim, nvars)
        state = np.ones(size)
        drift = np.ones(size)
        drift_r = riskfree + state[:, 1]**2 * (lmbd - .5)
        drift_v = kappa * (mean_v - state[:, 1])
        drift = np.vstack([drift_r, drift_v]).T

        self.assertEqual(ajd_drift(state, param).shape, drift.shape)
        np.testing.assert_almost_equal(ajd_drift(state, param), drift)

    def test_ajd_diff_heston(self):
        """Test AJD diffusion function for Heston model."""

        riskfree, lmbd, mean_v, kappa, eta, rho = 0., .01, .2, 1.5, .2, -.0
        param = HestonParam(riskfree=riskfree, lmbd=lmbd,
                            mean_v=mean_v, kappa=kappa,
                            eta=eta, rho=rho)
        nvars, nsim = 2, 3
        size = (nsim, nvars)
        state = np.ones(size)
        diff = np.ones((nsim, nvars, nvars))
        var = np.array([[1, eta*rho], [eta*rho, eta**2]])
        var = ((np.ones((nsim, nvars, nvars)) * var).T * state[:, 1]).T
        diff = np.linalg.cholesky(var)

        self.assertEqual(ajd_diff(state, param).shape, diff.shape)
        np.testing.assert_array_equal(ajd_diff(state, param), diff)

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


class SimulationTestCase(ut.TestCase):
    """Test simulation capabilities."""

    def test_gbm_simupdate(self):
        """Test simulation update of the GBM model."""

        mean, sigma = 1.5, .2
        param = GBMparam(mean, sigma)
        gbm = GBM(param)
        gbm.ndiscr, gbm.interval = 2, .5
        nvars, nsim = 1, 2
        size = (nsim, nvars)
        state = np.ones(size)
        error = np.zeros(size)

        new_state = gbm.update(state, error)
        loc = state * (mean - sigma**2/2)
        scale = np.ones((nsim, nvars, nvars)) * sigma
        delta = gbm.interval / gbm.ndiscr
        new_state_compute = loc * delta + (scale * error).sum(1) * delta**.5

        self.assertEqual(new_state.shape, size)
        np.testing.assert_array_equal(new_state, new_state_compute)

    def test_vasicek_simupdate(self):
        """Test simulation update of the Vasicek model."""

        mean, kappa, eta = 1.5, 1, .2
        param = VasicekParam(mean, kappa, eta)
        vasicek = Vasicek(param)
        vasicek.ndiscr, vasicek.interval = 2, .5
        nvars, nsim = 1, 2
        size = (nsim, nvars)
        state = np.ones(size)
        error = np.zeros(size)

        new_state = vasicek.update(state, error)
        loc = kappa * (mean - state)
        scale = np.ones((nsim, nvars, nvars)) * eta
        delta = vasicek.interval / vasicek.ndiscr
        new_state_compute = loc * delta + (scale * error).sum(1) * delta**.5

        self.assertEqual(new_state.shape, size)
        np.testing.assert_array_equal(new_state, new_state_compute)

    def test_cir_simupdate(self):
        """Test simulation update of the CIR model."""

        mean, kappa, eta = 1.5, 1, .2
        param = CIRparam(mean, kappa, eta)
        cir = CIR(param)
        cir.ndiscr, cir.interval = 2, .5
        nvars, nsim = 1, 2
        size = (nsim, nvars)
        state_val = 4
        state = np.ones(size) * state_val
        error = np.zeros(size)

        new_state = cir.update(state, error)
        loc = kappa * (mean - state)
        scale = np.ones((nsim, nvars, nvars)) * eta * state**.5
        delta = cir.interval / cir.ndiscr
        new_state_compute = loc * delta + (scale * error).sum(1) * delta**.5

        self.assertEqual(new_state.shape, size)
        np.testing.assert_array_equal(new_state, new_state_compute)

    def test_heston_simupdate(self):
        """Test simulation update of the Heston model."""

        riskfree, lmbd, mean_v, kappa, eta, rho = 0., .01, .2, 1.5, .2**.5, -.5
        param = HestonParam(riskfree=riskfree, lmbd=lmbd,
                            mean_v=mean_v, kappa=kappa,
                            eta=eta, rho=rho)
        heston = Heston(param)
        heston.ndiscr, heston.interval = 2, .5
        nvars, nsim = 2, 3
        size = (nsim, nvars)
        state = np.ones(size)
        error = np.vstack([np.zeros(nsim), np.ones(nsim)]).T

        new_state = heston.update(state, error)
        drift_r = riskfree + state[:, 1]**2 * (lmbd - .5)
        drift_v = kappa * (mean_v - state[:, 1])
        loc = np.vstack([drift_r, drift_v]).T

        var = np.array([[1, eta*rho], [eta*rho, eta**2]])
        var = ((np.ones((nsim, nvars, nvars)) * var).T * state[:, 1]).T
        scale = np.linalg.cholesky(var)

        delta = heston.interval / heston.ndiscr
        new_state_compute = loc * delta
        for i in range(nsim):
            new_state_compute[i] += (scale[i] * error[i]).sum(1) * delta**.5

        self.assertEqual(new_state.shape, size)
        np.testing.assert_almost_equal(new_state, new_state_compute)

    def test_gbm_simulation(self):
        """Test simulation of the GBM model."""

        nvars = 1
        mean, sigma = 1.5, .2
        param = GBMparam(mean, sigma)
        gbm = GBM(param)
        start, nperiods, interval, ndiscr, nsim = 1, 5, .5, 3, 4
        nobs = int(nperiods / interval)
        paths = gbm.simulate(start, interval, ndiscr, nobs, nsim, diff=0)

        self.assertEqual(paths.shape, (nobs, 2*nsim, nvars))

        nsim = 1
        paths = gbm.simulate(start, interval, ndiscr, nobs, nsim, diff=0)

        self.assertEqual(paths.shape, (nobs, 2*nsim, nvars))

        paths = gbm.simulate(start, interval, ndiscr, nobs, nsim)

        self.assertEqual(paths.shape, (nobs, 2*nsim, nvars))

        fun = lambda: gbm.simulate([1, 1], interval, ndiscr,
                                   nobs, nsim, diff=0)
        self.assertRaises(ValueError, fun)

    def test_vasicek_simulation(self):
        """Test simulation of the Vasicek model."""

        nvars = 1
        mean, kappa, eta = 1.5, .1, .2
        param = VasicekParam(mean, kappa, eta)
        vasicek = Vasicek(param)
        start, nperiods, interval, ndiscr, nsim = 1, 5, .5, 3, 4
        nobs = int(nperiods / interval)
        paths = vasicek.simulate(start, interval, ndiscr, nobs, nsim)

        self.assertEqual(paths.shape, (nobs, 2*nsim, nvars))

        fun = lambda: vasicek.simulate([1, 1], interval, ndiscr,
                                       nobs, nsim, diff=0)

        self.assertRaises(ValueError, fun)

    def test_cir_simulation(self):
        """Test simulation of the CIR model."""

        nvars = 1
        mean, kappa, eta = 1.5, .1, .2
        param = CIRparam(mean, kappa, eta)
        cir = CIR(param)
        start, nperiods, interval, ndiscr, nsim = 1, 5, .5, 3, 4
        nobs = int(nperiods / interval)
        paths = cir.simulate(start, interval, ndiscr, nobs, nsim)

        self.assertEqual(paths.shape, (nobs, 2*nsim, nvars))

        fun = lambda: cir.simulate([1, 1], interval, ndiscr,
                                   nobs, nsim, diff=0)

        self.assertRaises(ValueError, fun)

    def test_heston_simulation(self):
        """Test simulation of the Heston model."""

        nvars = 2
        riskfree, lmbd, mean_v, kappa, eta, rho = 0., .01, .2, 1.5, .2**.5, -.5
        param = HestonParam(riskfree=riskfree, lmbd=lmbd,
                            mean_v=mean_v, kappa=kappa,
                            eta=eta, rho=rho)
        heston = Heston(param)
        start, nperiods, interval, ndiscr, nsim = [1, mean_v], 5, .5, 3, 4
        nobs = int(nperiods / interval)
        paths = heston.simulate(start, interval, ndiscr, nobs, nsim, diff=0)

        self.assertEqual(paths.shape, (nobs, 2*nsim, nvars))

        paths = heston.simulate(start, interval, ndiscr, nobs, nsim)

        self.assertEqual(paths.shape, (nobs, 2*nsim, nvars))

        fun = lambda: heston.simulate(0, interval, ndiscr, nobs, nsim, diff=0)

        self.assertRaises(ValueError, fun)

    def test_gbm_sim_realized(self):
        """Test simulation of realized values of the GBM model."""

        mean, sigma = 1.5, .2
        param = GBMparam(mean, sigma)
        gbm = GBM(param)
        start, nperiods, interval, ndiscr, nsim = 1, 5, 1/80, 3, 4
        aggh = 2
        returns, rvol = gbm.sim_realized(start, interval=interval, aggh=aggh,
                                         ndiscr=ndiscr, nperiods=nperiods,
                                         nsim=nsim, diff=0)

        self.assertEqual(returns.shape, (nperiods-aggh+1, ))
        self.assertEqual(rvol.shape, (nperiods-aggh+1, ))

    def test_vasicek_sim_realized(self):
        """Test simulation of realized values of the Vasicek model."""

        mean, kappa, eta = 1.5, .1, .2
        param = VasicekParam(mean, kappa, eta)
        vasicek = Vasicek(param)
        start, nperiods, interval, ndiscr, nsim = 1, 5, .5, 3, 4
        aggh = 2
        returns, rvol = vasicek.sim_realized(start, interval=interval,
                                             ndiscr=ndiscr, nperiods=nperiods,
                                             nsim=nsim, aggh=aggh, diff=0)

        self.assertEqual(returns.shape, (nperiods-aggh+1, ))
        self.assertEqual(rvol.shape, (nperiods-aggh+1, ))

    def test_cir_sim_realized(self):
        """Test simulation of realized values of the CIR model."""

        mean, kappa, eta = 1.5, .1, .2
        param = CIRparam(mean, kappa, eta)
        cir = CIR(param)
        start, nperiods, interval, ndiscr, nsim = 1, 5, .5, 3, 4
        aggh = 2
        returns, rvol = cir.sim_realized(start, interval=interval,
                                         ndiscr=ndiscr, nperiods=nperiods,
                                         nsim=nsim, aggh=aggh, diff=0)

        self.assertEqual(returns.shape, (nperiods-aggh+1, ))
        self.assertEqual(rvol.shape, (nperiods-aggh+1, ))

    def test_heston_sim_realized(self):
        """Test simulation of realized values of the Heston model."""

        riskfree, lmbd, mean_v, kappa, eta, rho = 0., .01, .2, 1.5, .2**.5, -.5
        param = HestonParam(riskfree=riskfree, lmbd=lmbd,
                            mean_v=mean_v, kappa=kappa,
                            eta=eta, rho=rho)
        heston = Heston(param)
        start, nperiods, interval, ndiscr, nsim = [1, mean_v], 5, .5, 3, 4
        aggh = 2
        returns, rvol = heston.sim_realized(start, interval=interval,
                                            ndiscr=ndiscr, nperiods=nperiods,
                                            nsim=nsim, aggh=aggh, diff=0)

        self.assertEqual(returns.shape, (nperiods-aggh+1, ))
        self.assertEqual(rvol.shape, (nperiods-aggh+1, ))


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

    def test_heston_instruments(self):
        """Test realized moments of Heston model."""
        riskfree, lmbd, mean_v, kappa, eta, rho = 0., .01, .2, 1.5, .2**.5, -.5
        param = HestonParam(riskfree=riskfree, lmbd=lmbd,
                            mean_v=mean_v, kappa=kappa,
                            eta=eta, rho=rho)
        heston = Heston(param)
        heston.interval = .5

        nperiods = 10
        data = np.ones((2, nperiods))
        instrlag = 2

        instruments = heston.instruments(data[0], instrlag=instrlag)
        ninstr = 1
        shape = (nperiods, ninstr*instrlag + 1)

        # Test the shape of instruments
        self.assertEqual(instruments.shape, shape)

        instruments = heston.instruments(nobs=nperiods)
        ninstr = 0
        shape = (nperiods, ninstr*instrlag + 1)

        # Test the shape of instruments
        self.assertEqual(instruments.shape, shape)

    def test_heston_relized_mom(self):
        """Test realized moments of Heston model."""
        riskfree, lmbd, mean_v, kappa, eta, rho = 0., .01, .2, 1.5, .2**.5, -.5
        param = HestonParam(riskfree=riskfree, lmbd=lmbd,
                            mean_v=mean_v, kappa=kappa,
                            eta=eta, rho=rho)
        heston = Heston(param)
        heston.interval = .5
        nparams = param.get_theta().size
        nmoms = 4

        nperiods = 5
        ret = np.arange(nperiods)
        rvar = ret ** 2
        data = np.vstack([ret, rvar])
        instrlag = 2

        depvar = heston.realized_depvar(data)
        instr_data = np.vstack([rvar, rvar**2])
        ninstr = instr_data.shape[0]

        # Test shape of dependent variables
        self.assertEqual(depvar.shape, (nperiods, 3 * 4))

        mom, dmom = heston.integrated_mom(param.get_theta(),
                                          instr_data=instr_data,
                                          instr_choice='var', exact_jacob=True,
                                          data=data, instrlag=instrlag)
        nmoms_all = nmoms * (ninstr*instrlag + 1)
        mom_shape = (nperiods - instrlag, nmoms_all)

        # Test the shape of moment functions
        self.assertEqual(mom.shape, mom_shape)

        dmom_shape = (nmoms_all, nparams)

        # Test the shape of the Jacobian
        self.assertEqual(dmom.shape, dmom_shape)

        mom, dmom = heston.integrated_mom(param.get_theta(),
                                          instr_choice='const',
                                          data=data, instrlag=instrlag)
        nmoms_all = nmoms
        mom_shape = (nperiods - instrlag, nmoms_all)

        # Test the shape of moment functions
        self.assertEqual(mom.shape, mom_shape)

    def test_heston_coefs(self):
        """Test coefficients in descretization of Heston model.

        """
        riskfree, lmbd, mean_v, kappa, eta, rho = 0., .01, .2, 1.5, .2**.5, -.5
        param = HestonParam(riskfree=riskfree, lmbd=lmbd,
                            mean_v=mean_v, kappa=kappa,
                            eta=eta, rho=rho)
        heston = Heston(param)
        heston.interval = .1
        theta = param.get_theta()
        nparams = theta.size
        nmoms = 4

        self.assertIsInstance(heston.coef_big_a(theta), float)
        self.assertIsInstance(heston.coef_small_a(theta), float)

        self.assertEqual(heston.mat_a0(theta).shape, (4, 4))
        self.assertEqual(heston.mat_a1(theta).shape, (4, 4))
        self.assertEqual(heston.mat_a2(theta).shape, (4, 4))

        self.assertEqual(heston.mat_a(theta).shape, (4, 3*4))

        self.assertEqual(heston.realized_const(theta).shape, (4, ))
        self.assertEqual(heston.realized_const(theta)[0], 0)

        res = heston.depvar_unc_mean(theta)[1] * (1 - heston.coef_big_a(theta))

        self.assertEqual(heston.realized_const(theta)[1], res)

        res = heston.depvar_unc_mean(theta)[2] \
            * (1 - heston.coef_big_a(theta)) \
            * (1 - heston.coef_big_a(theta)**2)

        self.assertEqual(heston.realized_const(theta)[2], res)

        res = (heston.depvar_unc_mean(theta)[2] * (.5 - param.lmbd) \
            + heston.depvar_unc_mean(theta)[3]) \
            * (1 - heston.coef_big_a(theta))

        self.assertAlmostEqual(heston.realized_const(theta)[3], res)

        dconst = heston.drealized_const(param.get_theta())

        # Test derivative of intercept
        self.assertEqual(dconst.shape, (nmoms, nparams))

        dmat_a = heston.diff_mat_a(param.get_theta())

        # Test derivative of matrix A
        self.assertEqual(len(dmat_a), nmoms)
        for mat in dmat_a:
            self.assertEqual(mat.shape, (3*nmoms, nparams))


if __name__ == '__main__':
    ut.main()
