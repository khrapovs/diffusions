#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Test suite for diffusion simulations.

"""
from __future__ import print_function, division

import unittest as ut
import numpy as np

from diffusions import (GBM, GBMparam, Vasicek, VasicekParam,
                        CIR, CIRparam, Heston, HestonParam,
                        CentTend, CentTendParam)


class GBMTestCase(ut.TestCase):
    """Test simupdate."""

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

    def test_ct_simupdate(self):
        """Test simulation update of the Central Tendency model."""

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
        centend = CentTend(param)
        centend.ndiscr, centend.interval = 2, .5
        nvars, nsim = 3, 3
        size = (nsim, nvars)
        state = np.ones(size)
        error = np.vstack([np.zeros(nsim), np.ones(nsim), np.ones(nsim)]).T

        new_state = centend.update(state, error)
        drift_r = riskfree + state[:, 1]**2 * (lmbd - .5)
        drift_s = kappa_s * (state[:, 2] - state[:, 1])
        drift_v = kappa_y * (mean_v - state[:, 2])
        loc = np.vstack([drift_r, drift_s, drift_v]).T

        var_s = np.zeros((3, 3))
        var_s[:2, :2] = np.array([[1, eta_s*rho], [eta_s*rho, eta_s**2]])
        var_v = np.zeros((3, 3))
        var_v[2, 2] = eta_y**2
        var = ((np.ones((nsim, nvars, nvars)) * var_s).T * state[:, 1]).T \
            + ((np.ones((nsim, nvars, nvars)) * var_v).T * state[:, 2]).T
        scale = np.linalg.cholesky(var)

        delta = centend.interval / centend.ndiscr
        new_state_compute = loc * delta
        for i in range(nsim):
            new_state_compute[i] += (scale[i] * error[i]).sum(1) * delta**.5

        self.assertEqual(new_state.shape, size)
        np.testing.assert_almost_equal(new_state, new_state_compute)


class SimulationTestCase(ut.TestCase):
    """Test simulation."""

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

    def test_ct_simulation(self):
        """Test simulation of the Central Tendency model."""

        nvars = 3
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
        start = [1, mean_v, mean_v]
        nperiods, interval, ndiscr, nsim = 5, .5, 3, 4
        nobs = int(nperiods / interval)
        paths = centtend.simulate(start, interval, ndiscr, nobs, nsim, diff=0)

        self.assertEqual(paths.shape, (nobs, 2*nsim, nvars))

        paths = centtend.simulate(start, interval, ndiscr, nobs, nsim)

        self.assertEqual(paths.shape, (nobs, 2*nsim, nvars))

        fun = lambda: centtend.simulate(0, interval, ndiscr, nobs,
                                        nsim, diff=0)

        self.assertRaises(ValueError, fun)


class RealizedSimTestCase(ut.TestCase):
    """Test Realized data simulation."""

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

    def test_ct_sim_realized(self):
        """Test simulation of realized values of the Central Tendency model."""

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
        start = [1, mean_v, mean_v]
        nperiods, interval, ndiscr, nsim = 5, .5, 3, 4
        aggh = 2
        returns, rvol = centtend.sim_realized(start, interval=interval,
                                              ndiscr=ndiscr, nperiods=nperiods,
                                              nsim=nsim, aggh=aggh, diff=0)

        self.assertEqual(returns.shape, (nperiods-aggh+1, ))
        self.assertEqual(rvol.shape, (nperiods-aggh+1, ))


if __name__ == '__main__':
    ut.main()
