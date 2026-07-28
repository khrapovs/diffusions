"""Test suite for realized moments of Heston."""

import numpy as np
import numpy.testing as npt
import pytest

from affidiff import Heston, HestonParam


class TestRealizedMomentsHeston:
    """Test realized moments for Heston."""

    def test_heston_depvar(self) -> None:
        """Test dependent variables of Heston model."""
        riskfree = 0.0
        lmbd, mean_v, kappa, eta, rho = 0.01, 0.2, 1.5, 0.2**0.5, -0.5
        lmbd_v = 0.2
        param = HestonParam(riskfree=riskfree, lmbd=lmbd, mean_v=mean_v, kappa=kappa, eta=eta, rho=rho, lmbd_v=lmbd_v)
        heston = Heston(param)
        heston.nsub = 2

        nperiods = 5
        ret = np.ones(nperiods) * (lmbd - 0.5) * mean_v
        rvar = np.ones(nperiods) * mean_v
        data = np.vstack([ret, rvar])
        depvar = heston.realized_depvar(data=data)

        # Test shape of dependent variables
        assert depvar.shape == (nperiods, 3 * 4)

    def test_heston_var_instr(self) -> None:
        """Test realized moments with variable instruments of Heston model."""
        riskfree = 0.0
        lmbd, mean_v, kappa, eta, rho = 0.01, 0.2, 1.5, 0.2**0.5, -0.5
        lmbd_v = 0.2
        param = HestonParam(riskfree=riskfree, lmbd=lmbd, mean_v=mean_v, kappa=kappa, eta=eta, rho=rho, lmbd_v=lmbd_v)
        heston = Heston(param)
        heston.nsub = 2
        nmoms = 4

        nperiods = 5
        ret = np.ones(nperiods) * (lmbd - 0.5) * mean_v
        rvar = np.ones(nperiods) * mean_v
        data = np.vstack([ret, rvar])
        instrlag = 2
        theta = param.get_theta(subset="all", measure="P")

        instr_data = np.vstack([rvar, rvar**2])
        ninstr = instr_data.shape[0]

        mom, dmom = heston.integrated_mom(
            theta=theta, instr_data=instr_data, instr_choice="var", data=data, instrlag=instrlag
        )
        nmoms_all = nmoms * (ninstr * instrlag + 1)
        mom_shape = (nperiods - instrlag, nmoms_all)

        # Test the shape of moment functions
        assert mom.shape == mom_shape
        assert dmom is None

    def test_const_instr(self) -> None:
        """Test constant instrument of Heston model."""
        riskfree = 0.0
        lmbd, mean_v, kappa, eta, rho = 0.01, 0.2, 1.5, 0.2**0.5, -0.5
        lmbd_v = 0.2
        param = HestonParam(riskfree=riskfree, lmbd=lmbd, mean_v=mean_v, kappa=kappa, eta=eta, rho=rho, lmbd_v=lmbd_v)
        heston = Heston(param)
        heston.nsub = 2
        nmoms = 4

        nperiods = 5
        ret = np.ones(nperiods) * (lmbd - 0.5) * mean_v
        rvar = np.ones(nperiods) * mean_v
        data = np.vstack([ret, rvar])
        instrlag = 2
        theta = param.get_theta(subset="all", measure="P")

        depvar = heston.realized_depvar(data=data)

        mom, dmom = heston.integrated_mom(theta=theta, instr_choice="const", data=data, instrlag=instrlag)
        nmoms_all = nmoms
        mom_shape = (nperiods - instrlag, nmoms_all)

        aggh = 2
        means = [
            heston.mean_vol(param=param, aggh=aggh),
            heston.mean_vol2(param=param, aggh=aggh),
            heston.mean_ret(param=param, aggh=aggh),
            heston.mean_cross(param=param, aggh=aggh),
        ]

        depvar = np.ones((nperiods - instrlag, 4)) * means
        depvar = np.tile(depvar, 3)
        error = depvar.dot(heston.mat_a(param=param, subset=None).T) - heston.realized_const(
            param=param, aggh=aggh, subset=None
        )

        npt.assert_array_almost_equal(error, np.zeros(mom_shape))

        # Test the shape of moment functions
        assert mom.shape == mom_shape

    def test_vol_p(self) -> None:
        """Test vol P realized moments of Heston model."""
        riskfree = 0.0
        lmbd, mean_v, kappa, eta, rho = 0.01, 0.2, 1.5, 0.2**0.5, -0.5
        lmbd_v = 0.2
        param = HestonParam(riskfree=riskfree, lmbd=lmbd, mean_v=mean_v, kappa=kappa, eta=eta, rho=rho, lmbd_v=lmbd_v)
        heston = Heston(param)
        heston.nsub = 2

        nperiods = 5
        ret = np.ones(nperiods) * (lmbd - 0.5) * mean_v
        rvar = np.ones(nperiods) * mean_v
        data = np.vstack([ret, rvar])
        instrlag = 2

        aggh = 2
        means = [
            heston.mean_vol(param=param, aggh=aggh),
            heston.mean_vol2(param=param, aggh=aggh),
            heston.mean_ret(param=param, aggh=aggh),
            heston.mean_cross(param=param, aggh=aggh),
        ]

        subset = "vol"
        theta = param.get_theta(subset=subset)
        mom, dmom = heston.integrated_mom(
            theta=theta, subset=subset, instr_choice="const", data=data, instrlag=instrlag
        )
        nmoms = 2
        mom_shape = (nperiods - instrlag, nmoms)

        # Test the shape of moment functions
        assert mom.shape == mom_shape

        depvar = np.ones((nperiods - instrlag, 4)) * means
        depvar = np.tile(depvar, 3)
        subset_sl = slice(2)
        error = depvar.dot(heston.mat_a(param=param, subset=subset_sl).T) - heston.realized_const(
            param=param, aggh=aggh, subset=subset_sl
        )

        npt.assert_array_almost_equal(error, np.zeros(mom_shape))

    def test_heston_relized_mom(self) -> None:
        """Test realized moments of Heston model."""
        riskfree = 0.0
        lmbd, mean_v, kappa, eta, rho = 0.01, 0.2, 1.5, 0.2**0.5, -0.5
        lmbd_v = 0.2
        param = HestonParam(riskfree=riskfree, lmbd=lmbd, mean_v=mean_v, kappa=kappa, eta=eta, rho=rho, lmbd_v=lmbd_v)
        heston = Heston(param)
        heston.nsub = 2

        nperiods = 5
        ret = np.ones(nperiods) * (lmbd - 0.5) * mean_v
        rvar = np.ones(nperiods) * mean_v
        data = np.vstack([ret, rvar])
        instrlag = 2

        subset = "vol"
        measure = "Q"
        theta = param.get_theta(subset=subset, measure=measure)
        mom, dmom = heston.integrated_mom(
            theta=theta, subset=subset, measure=measure, instr_choice="const", data=data, instrlag=instrlag
        )
        nmoms = 2
        mom_shape = (nperiods - instrlag, nmoms)

        # Test the shape of moment functions
        assert mom.shape == mom_shape

        subset_sl = slice(2)
        aggh = 2
        means = [
            heston.mean_vol(param=param, aggh=aggh),
            heston.mean_vol2(param=param, aggh=aggh),
            heston.mean_ret(param=param, aggh=aggh),
            heston.mean_cross(param=param, aggh=aggh),
        ]
        depvar = np.ones((nperiods - instrlag, 4)) * means
        depvar = np.tile(depvar, 3)
        error = depvar.dot(heston.mat_a(param=param, subset=subset_sl).T) - heston.realized_const(
            param=param, aggh=aggh, subset=subset_sl
        )

        npt.assert_array_almost_equal(error, np.zeros(mom_shape))

        subset = "vol"
        measure = "P"
        theta = param.get_theta(subset=subset, measure=measure)
        mom, dmom = heston.integrated_mom(
            theta=theta, subset=subset, measure=measure, instr_choice="const", data=data, instrlag=instrlag
        )
        nmoms = 2
        mom_shape = (nperiods - instrlag, nmoms)

        # Test the shape of moment functions
        assert mom.shape == mom_shape

        subset_sl = slice(2)
        aggh = 2
        means = [
            heston.mean_vol(param=param, aggh=aggh),
            heston.mean_vol2(param=param, aggh=aggh),
            heston.mean_ret(param=param, aggh=aggh),
            heston.mean_cross(param=param, aggh=aggh),
        ]
        depvar = np.ones((nperiods - instrlag, 4)) * means
        depvar = np.tile(depvar, 3)
        error = depvar.dot(heston.mat_a(param=param, subset=subset_sl).T) - heston.realized_const(
            param=param, aggh=aggh, subset=subset_sl
        )

        npt.assert_array_almost_equal(error, np.zeros(mom_shape))

        param = HestonParam(riskfree=riskfree, lmbd=lmbd, mean_v=mean_v, kappa=kappa, eta=eta, rho=rho, lmbd_v=lmbd_v)
        heston = Heston(param)
        subset = "vol"
        measure = "PQ"
        theta = param.get_theta(subset=subset, measure=measure)
        mom, dmom = heston.integrated_mom(
            theta=theta,
            subset=subset,
            measure=measure,
            instr_choice="const",
            aggh=[aggh, aggh],
            data=[data, data],
            instrlag=instrlag,
        )
        nmoms = 2
        mom_shape = (nperiods - instrlag, nmoms * 2)

        # Test the shape of moment functions
        assert mom.shape == mom_shape

        subset_sl = slice(2)
        aggh = 10
        means = [
            heston.mean_vol(param=param, aggh=aggh),
            heston.mean_vol2(param=param, aggh=aggh),
            heston.mean_ret(param=param, aggh=aggh),
            heston.mean_cross(param=param, aggh=aggh),
        ]
        depvar = np.ones((nperiods - instrlag, 4)) * means
        depvar = np.tile(depvar, 3)
        error_q = depvar.dot(heston.mat_a(param=param, subset=subset_sl).T) - heston.realized_const(
            param=param, aggh=aggh, subset=subset_sl
        )

        param = HestonParam(riskfree=riskfree, lmbd=lmbd, mean_v=mean_v, kappa=kappa, eta=eta, rho=rho, lmbd_v=lmbd_v)

        means = [
            heston.mean_vol(param=param, aggh=aggh),
            heston.mean_vol2(param=param, aggh=aggh),
            heston.mean_ret(param=param, aggh=aggh),
            heston.mean_cross(param=param, aggh=aggh),
        ]
        depvar = np.ones((nperiods - instrlag, 4)) * means
        depvar = np.tile(depvar, 3)
        error_p = depvar.dot(heston.mat_a(param=param, subset=subset_sl).T) - heston.realized_const(
            param=param, aggh=aggh, subset=subset_sl
        )

        npt.assert_array_almost_equal(np.hstack((error_p, error_q)), np.zeros(mom_shape))

        param = HestonParam(
            riskfree=riskfree, lmbd=lmbd, mean_v=mean_v, kappa=kappa, eta=eta, rho=rho, lmbd_v=2 * lmbd_v
        )
        heston = Heston(param)
        theta = param.get_theta(subset=subset, measure=measure)
        mom2, dmom = heston.integrated_mom(
            theta=theta,
            subset=subset,
            measure=measure,
            instr_choice="const",
            aggh=[aggh, aggh],
            data=[data, data],
            instrlag=instrlag,
        )

        assert not np.allclose(mom, mom2)

    def test_heston_relized_mom_all(self) -> None:
        """Test realized moments of Heston model."""
        riskfree = 0.0
        lmbd, mean_v, kappa, eta, rho = 0.01, 0.2, 1.5, 0.2**0.5, -0.5
        lmbd_v = 0.2

        nperiods = 5
        instrlag = 2
        ret = np.ones(nperiods) * (lmbd - 0.5) * mean_v
        rvar = np.ones(nperiods) * mean_v
        data = np.vstack([ret, rvar])

        subset = "all"
        measure = "Q"
        param = HestonParam(riskfree=riskfree, lmbd=lmbd, mean_v=mean_v, kappa=kappa, eta=eta, rho=rho, lmbd_v=lmbd_v)
        heston = Heston(param)
        theta = param.get_theta(subset=subset, measure=measure)
        mom, dmom = heston.integrated_mom(
            theta=theta, subset=subset, measure=measure, instr_choice="const", data=data, instrlag=instrlag
        )
        nmoms = 4
        mom_shape = (nperiods - instrlag, nmoms)

        # Test the shape of moment functions
        assert mom.shape == mom_shape

        subset_sl = None
        aggh = 2
        means = [
            heston.mean_vol(param=param, aggh=aggh),
            heston.mean_vol2(param=param, aggh=aggh),
            heston.mean_ret(param=param, aggh=aggh),
            heston.mean_cross(param=param, aggh=aggh),
        ]
        depvar = np.ones((nperiods - instrlag, 4)) * means
        depvar = np.tile(depvar, 3)
        error = depvar.dot(heston.mat_a(param=param, subset=subset_sl).T) - heston.realized_const(
            param=param, aggh=aggh, subset=subset_sl
        )

        npt.assert_array_almost_equal(error, np.zeros(mom_shape))

        subset = "all"
        measure = "P"
        param = HestonParam(riskfree=riskfree, lmbd=lmbd, mean_v=mean_v, kappa=kappa, eta=eta, rho=rho, lmbd_v=lmbd_v)
        theta = param.get_theta(subset=subset, measure=measure)
        mom, dmom = heston.integrated_mom(
            theta=theta, subset=subset, measure=measure, instr_choice="const", data=data, instrlag=instrlag
        )
        nmoms = 4
        mom_shape = (nperiods - instrlag, nmoms)

        # Test the shape of moment functions
        assert mom.shape == mom_shape

        subset_sl = None
        aggh = 2
        means = [
            heston.mean_vol(param=param, aggh=aggh),
            heston.mean_vol2(param=param, aggh=aggh),
            heston.mean_ret(param=param, aggh=aggh),
            heston.mean_cross(param=param, aggh=aggh),
        ]
        depvar = np.ones((nperiods - instrlag, 4)) * means
        depvar = np.tile(depvar, 3)
        error = depvar.dot(heston.mat_a(param=param, subset=subset_sl).T) - heston.realized_const(
            param=param, aggh=aggh, subset=subset_sl
        )

        npt.assert_array_almost_equal(error, np.zeros(mom_shape))

        subset = "all"
        measure = "PQ"
        param = HestonParam(riskfree=riskfree, lmbd=lmbd, mean_v=mean_v, kappa=kappa, eta=eta, rho=rho, lmbd_v=lmbd_v)
        theta = param.get_theta(subset=subset, measure=measure)
        mom, dmom = heston.integrated_mom(
            theta=theta,
            subset=subset,
            measure=measure,
            instr_choice="const",
            aggh=[aggh, aggh],
            data=[data, data],
            instrlag=instrlag,
        )
        nmoms = 4
        mom_shape = (nperiods - instrlag, nmoms * 2)

        # Test the shape of moment functions
        assert mom.shape == mom_shape

        subset_sl = None
        aggh = 2
        means = [
            heston.mean_vol(param=param, aggh=aggh),
            heston.mean_vol2(param=param, aggh=aggh),
            heston.mean_ret(param=param, aggh=aggh),
            heston.mean_cross(param=param, aggh=aggh),
        ]
        depvar = np.ones((nperiods - instrlag, 4)) * means
        depvar = np.tile(depvar, 3)
        error_q = depvar.dot(heston.mat_a(param=param, subset=subset_sl).T) - heston.realized_const(
            param=param, aggh=aggh, subset=subset_sl
        )

        param = HestonParam(riskfree=riskfree, lmbd=lmbd, mean_v=mean_v, kappa=kappa, eta=eta, rho=rho, lmbd_v=lmbd_v)
        means = [
            heston.mean_vol(param=param, aggh=aggh),
            heston.mean_vol2(param=param, aggh=aggh),
            heston.mean_ret(param=param, aggh=aggh),
            heston.mean_cross(param=param, aggh=aggh),
        ]
        depvar = np.ones((nperiods - instrlag, 4)) * means
        depvar = np.tile(depvar, 3)
        error_p = depvar.dot(heston.mat_a(param=param, subset=subset_sl).T) - heston.realized_const(
            param=param, aggh=aggh, subset=subset_sl
        )

        npt.assert_array_almost_equal(np.hstack((error_p, error_q)), np.zeros(mom_shape))

    def test_heston_coefs(self) -> None:
        """Test coefficients in descretization of Heston model."""
        riskfree, lmbd, mean_v, kappa, eta, rho = 0.0, 0.01, 0.2, 1.5, 0.2**0.5, -0.5
        param = HestonParam(riskfree=riskfree, lmbd=lmbd, mean_v=mean_v, kappa=kappa, eta=eta, rho=rho)
        heston = Heston(param)
        heston.nsub = 10
        aggh = 2

        assert isinstance(heston.coef_big_a(param=param, aggh=aggh), float)
        assert isinstance(heston.coef_small_a(param=param, aggh=aggh), float)
        assert isinstance(heston.coef_big_c(param=param, aggh=aggh), float)
        assert isinstance(heston.coef_small_c(param=param, aggh=aggh), float)

        assert heston.mat_a0(param=param, aggh=aggh).shape == (4, 4)
        assert heston.mat_a1(param=param, aggh=aggh).shape == (4, 4)
        assert heston.mat_a2(param=param, aggh=aggh).shape == (4, 4)

        assert heston.mat_a(param=param).shape == (4, 3 * 4)

        assert heston.realized_const(param=param, aggh=aggh).shape == (4,)
        assert heston.realized_const(param=param, aggh=aggh)[2] == 0

        means = [
            heston.mean_vol(param=param, aggh=aggh),
            heston.mean_vol2(param=param, aggh=aggh),
            heston.mean_ret(param=param, aggh=aggh),
            heston.mean_cross(param=param, aggh=aggh),
        ]

        npt.assert_array_equal(heston.depvar_unc_mean(param=param, aggh=aggh), means)

        res = heston.mean_vol(param=param, aggh=aggh) * (1 - heston.coef_big_a(param=param, aggh=1))

        assert heston.realized_const(param=param, aggh=aggh)[0] == res

        res = (
            heston.mean_vol2(param=param, aggh=aggh)
            * (1 - heston.coef_big_a(param=param, aggh=1))
            * (1 - heston.coef_big_a(param=param, aggh=1) ** 2)
        )

        assert heston.realized_const(param=param, aggh=aggh)[1] == res

        res = heston.mean_ret(param=param, aggh=aggh) + heston.mean_vol(param=param, aggh=aggh) * (0.5 - lmbd)

        assert heston.realized_const(param=param, aggh=aggh)[2] == res

        res = heston.mean_vol2(param=param, aggh=aggh) * (0.5 - lmbd) * (
            1 - heston.coef_big_a(param=param, aggh=1)
        ) + heston.mean_cross(param=param, aggh=aggh) * (1 - heston.coef_big_a(param=param, aggh=1))

        assert heston.realized_const(param=param, aggh=aggh)[3] == pytest.approx(res)
