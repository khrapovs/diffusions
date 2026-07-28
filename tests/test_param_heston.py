"""Test suite for Heston parameter class."""

import warnings

import numpy as np
import numpy.testing as npt
import pytest

from affidiff import HestonParam


class TestSDEParameter:
    """Test parameter classes."""

    def test_init(self) -> None:
        """Test initialization."""
        riskfree = 0.01
        mean_v = 0.5
        kappa = 1.5
        eta = 0.1
        lmbd = 0.01
        rho = -0.5

        param = HestonParam(riskfree=riskfree, lmbd=lmbd, mean_v=mean_v, kappa=kappa, eta=eta, rho=rho)
        names = ["mean_v", "kappa", "eta", "rho", "lmbd", "lmbd_v"]

        assert param.measure == "P"
        assert param.get_model_name() == "Heston"
        assert param.get_names() == names
        assert param.get_names(subset="all") == names
        assert param.get_names(subset="vol") == names[:3] + names[5:]
        assert param.get_names(subset="vol", measure="P") == names[:3]
        assert param.get_names(subset="vol", measure="Q") == names[:3]
        assert param.get_names(subset="all", measure="P") == names[:-1]
        assert param.get_names(subset="all", measure="Q") == names[:-1]

        assert param.riskfree == riskfree
        assert param.lmbd == lmbd
        assert param.lmbd_v == 0.0
        assert param.mean_v == mean_v
        assert param.kappa == kappa
        assert param.eta == eta
        assert param.rho == rho
        assert param.is_valid()

    def test_constraints(self) -> None:
        """Test constraints."""
        riskfree = 0.01
        mean_v = 0.5
        kappa = 1.5
        eta = 0.1
        lmbd = 0.01
        rho = -0.5

        param = HestonParam(riskfree=riskfree, lmbd=lmbd, mean_v=mean_v, kappa=kappa, eta=eta, rho=rho)

        assert param.get_constraints() == ()

    def test_init_q(self) -> None:
        """Test initialization under Q."""
        riskfree = 0.01
        mean_v = 0.5
        kappa = 1.5
        eta = 0.1
        lmbd = 0.01
        lmbd_v = 0.5
        rho = -0.5

        param = HestonParam(
            riskfree=riskfree, lmbd=lmbd, lmbd_v=lmbd_v, mean_v=mean_v, kappa=kappa, eta=eta, rho=rho, measure="Q"
        )

        assert param.measure == "Q"
        assert param.riskfree == riskfree
        assert param.lmbd == 0
        assert param.lmbd_v == lmbd_v
        assert param.mean_v == mean_v * kappa / param.kappa
        assert param.kappa == kappa - lmbd_v * eta
        assert param.eta == eta
        assert param.rho == rho
        assert param.is_valid()

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            param.convert_to_q()

    def test_from_theta(self) -> None:
        """Test from theta."""
        riskfree = 0.01
        mean_v = 0.5
        kappa = 1.5
        eta = 0.1
        lmbd = 0.01
        lmbd_v = 0.5
        rho = -0.5

        theta = [riskfree, mean_v, kappa, eta, rho, lmbd, lmbd_v]
        param = HestonParam.from_theta(theta, measure="P")

        assert param.measure == "P"
        assert param.riskfree == riskfree
        assert param.lmbd == lmbd
        assert param.lmbd_v == lmbd_v
        assert param.mean_v == mean_v
        assert param.kappa == kappa
        assert param.eta == eta
        assert param.rho == rho
        assert param.is_valid()

    def test_from_theta_q(self) -> None:
        """Test from theta under Q."""
        riskfree = 0.01
        mean_v = 0.5
        kappa = 1.5
        eta = 0.1
        lmbd = 0.01
        lmbd_v = 0.5
        rho = -0.5

        theta = [riskfree, mean_v, kappa, eta, rho, lmbd, lmbd_v]
        param = HestonParam.from_theta(theta, measure="Q")

        assert param.measure == "Q"
        assert param.riskfree == riskfree
        assert param.lmbd == 0
        assert param.lmbd_v == lmbd_v
        assert param.mean_v == mean_v * kappa / param.kappa
        assert param.kappa == kappa - lmbd_v * eta
        assert param.eta == eta
        assert param.rho == rho
        assert param.is_valid()

    def test_convert_to_q(self) -> None:
        """Test conversion to Q."""
        riskfree = 0.01
        mean_v = 0.5
        kappa = 1.5
        eta = 0.1
        lmbd = 0.01
        lmbd_v = 0.5
        rho = -0.5

        theta = [riskfree, mean_v, kappa, eta, rho, lmbd, lmbd_v]
        param = HestonParam.from_theta(theta)
        param.convert_to_q()

        assert param.measure == "Q"
        assert param.riskfree == riskfree
        assert param.lmbd == 0
        assert param.lmbd_v == lmbd_v
        assert param.mean_v == mean_v * kappa / param.kappa
        assert param.kappa == kappa - lmbd_v * eta
        assert param.eta == eta
        assert param.rho == rho
        assert param.is_valid()

    def test_ajd_matrices(self) -> None:
        """Test AJD matrices."""
        riskfree = 0.01
        mean_v = 0.5
        kappa = 1.5
        eta = 0.1
        lmbd = 0.01
        lmbd_v = 0.5
        rho = -0.5

        param = HestonParam(riskfree=riskfree, lmbd=lmbd, lmbd_v=lmbd_v, mean_v=mean_v, kappa=kappa, eta=eta, rho=rho)

        mat_k0 = [riskfree, kappa * mean_v]
        mat_k1 = [[0, lmbd - 0.5], [0, -kappa]]
        mat_h0 = np.zeros((2, 2))
        mat_h1 = [np.zeros((2, 2)), [[1, eta * rho], [eta * rho, eta**2]]]

        npt.assert_array_equal(param.mat_k0, mat_k0)
        npt.assert_array_equal(param.mat_k1, mat_k1)
        npt.assert_array_equal(param.mat_h0, mat_h0)
        npt.assert_array_equal(param.mat_h1, mat_h1)

        param.convert_to_q()

        kappa_q = kappa - lmbd_v * eta
        mean_v_q = mean_v * kappa / kappa_q

        mat_k0 = [riskfree, kappa_q * mean_v_q]
        mat_k1 = [[0, -0.5], [0, -kappa_q]]
        mat_h0 = np.zeros((2, 2))
        mat_h1 = [np.zeros((2, 2)), [[1, eta * rho], [eta * rho, eta**2]]]

        npt.assert_array_equal(param.mat_k0, mat_k0)
        npt.assert_array_equal(param.mat_k1, mat_k1)
        npt.assert_array_equal(param.mat_h0, mat_h0)
        npt.assert_array_equal(param.mat_h1, mat_h1)

        theta = np.ones(6)
        theta_vol = np.concatenate((theta[:3], theta[5:]))
        param = HestonParam()
        param.update(theta=theta)
        npt.assert_array_equal(param.get_theta(), theta)
        npt.assert_array_equal(param.get_theta(subset="vol"), theta_vol)

        mat_k0 = [param.riskfree, param.kappa * param.mean_v]
        mat_k1 = [[0, param.lmbd - 0.5], [0, -param.kappa]]
        mat_h0 = np.zeros((2, 2))
        mat_h1 = [np.zeros((2, 2)), [[1, param.eta * param.rho], [param.eta * param.rho, param.eta**2]]]

        npt.assert_array_equal(param.mat_k0, mat_k0)
        npt.assert_array_equal(param.mat_k1, mat_k1)
        npt.assert_array_equal(param.mat_h0, mat_h0)
        npt.assert_array_equal(param.mat_h1, mat_h1)

    def test_update(self) -> None:
        """Test update."""
        riskfree = 0.01
        mean_v = 0.5
        kappa = 1.5
        eta = 0.1
        lmbd = 0.01
        lmbd_v = 0.5
        rho = -0.5

        param = HestonParam(riskfree=riskfree, lmbd=lmbd, lmbd_v=lmbd_v, mean_v=mean_v, kappa=kappa, eta=eta, rho=rho)

        mean_v, kappa, eta, rho, lmbd = 0.6, 1.7, 0.2, -0.6, 0.3
        theta = np.array([mean_v, kappa, eta, rho, lmbd])
        param.update(theta=theta, measure="Q")
        mean_vq = mean_v * kappa / param.kappa
        kappa_q = kappa - lmbd_v * eta

        assert param.measure == "Q"
        assert param.riskfree == riskfree
        assert param.lmbd == 0
        assert param.lmbd_v == lmbd_v
        assert param.mean_v == pytest.approx(mean_vq)
        assert param.kappa == kappa_q
        assert param.eta == eta
        assert param.rho == rho
        assert param.is_valid()

        npt.assert_array_almost_equal(param.mat_k0, [riskfree, kappa_q * mean_vq])
        npt.assert_array_equal(param.mat_k1, [[0, -0.5], [0, -kappa_q]])

    def test_get_theta(self) -> None:
        """Test get theta."""
        riskfree = 0.01
        mean_v = 0.5
        kappa = 1.5
        eta = 0.1
        lmbd = 0.01
        lmbd_v = 0.5
        rho = -0.5

        param = HestonParam(
            riskfree=riskfree, lmbd=lmbd, lmbd_v=lmbd_v, mean_v=mean_v, kappa=kappa, eta=eta, rho=rho, measure="P"
        )

        theta = np.array([mean_v, kappa, eta, rho, lmbd, lmbd_v])
        theta_vol = np.concatenate((theta[:3], theta[5:]))

        npt.assert_array_equal(param.get_theta(), theta)
        npt.assert_array_equal(param.get_theta(subset="all"), theta)
        npt.assert_array_equal(param.get_theta(subset="all", measure="PQ"), theta)
        npt.assert_array_equal(param.get_theta(subset="all", measure="P"), theta[:-1])
        npt.assert_array_equal(param.get_theta(subset="all", measure="Q"), theta[:-1])
        npt.assert_array_equal(param.get_theta(subset="vol"), theta_vol)
        npt.assert_array_equal(param.get_theta(subset="vol", measure="PQ"), theta_vol)
        npt.assert_array_equal(param.get_theta(subset="vol", measure="P"), theta_vol[:-1])
        npt.assert_array_equal(param.get_theta(subset="vol", measure="Q"), theta_vol[:-1])

        theta = np.arange(6)
        param.update(theta=theta)
        theta_vol = np.ones(3) * 2
        param.update(theta=theta_vol, subset="vol", measure="P")
        theta[:3] = theta_vol
        npt.assert_array_equal(param.get_theta(), theta)

    def test_bounds(self) -> None:
        """Test bounds."""
        param = HestonParam()
        bounds = param.get_bounds()
        assert bounds is not None and len(bounds) == 6
        bounds_all = param.get_bounds(subset="all")
        assert bounds_all is not None and len(bounds_all) == 6
        bounds_pq = param.get_bounds(subset="all", measure="PQ")
        assert bounds_pq is not None and len(bounds_pq) == 6
        bounds_p = param.get_bounds(subset="all", measure="P")
        assert bounds_p is not None and len(bounds_p) == 5
        bounds_q = param.get_bounds(subset="all", measure="Q")
        assert bounds_q is not None and len(bounds_q) == 5
        bounds_vol = param.get_bounds(subset="vol")
        assert bounds_vol is not None and len(bounds_vol) == 4
        bounds_vol_pq = param.get_bounds(subset="vol", measure="PQ")
        assert bounds_vol_pq is not None and len(bounds_vol_pq) == 4
        bounds_vol_p = param.get_bounds(subset="vol", measure="P")
        assert bounds_vol_p is not None and len(bounds_vol_p) == 3
        bounds_vol_q = param.get_bounds(subset="vol", measure="Q")
        assert bounds_vol_q is not None and len(bounds_vol_q) == 3

    def test_validity(self) -> None:
        """Test validity."""
        riskfree = 0.01
        mean_v = 0.5
        kappa = 1.5
        eta = 0.1
        lmbd = 0.01
        lmbd_v = 0.5
        rho = -0.5

        param = HestonParam(
            riskfree=riskfree, lmbd=lmbd, lmbd_v=lmbd_v, mean_v=mean_v, kappa=kappa, eta=eta, rho=rho, measure="P"
        )

        assert param.is_valid()
        param = HestonParam(riskfree=riskfree, lmbd=lmbd, mean_v=-mean_v, kappa=kappa, eta=eta, rho=rho)
        assert not param.is_valid()
        param = HestonParam(riskfree=riskfree, lmbd=lmbd, mean_v=mean_v, kappa=-kappa, eta=eta, rho=rho)
        assert not param.is_valid()
        param = HestonParam(riskfree=riskfree, lmbd=lmbd, mean_v=mean_v, kappa=-kappa, eta=-eta, rho=rho)
        assert not param.is_valid()
