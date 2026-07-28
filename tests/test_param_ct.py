"""Test suite for CT parameter class."""

import warnings

import numpy as np
import numpy.testing as npt
import pytest

from affidiff import CentTendParam


class TestSDEParameter:
    """Test parameter classes."""

    def test_init(self) -> None:
        """Test initialization."""
        riskfree = 0.01
        lmbd = 0.01
        mean_v = 0.5
        kappa_s = 1.5
        kappa_y = 0.5
        eta_s = 0.1
        eta_y = 0.01
        rho = -0.5

        param = CentTendParam(
            riskfree=riskfree,
            lmbd=lmbd,
            mean_v=mean_v,
            kappa_s=kappa_s,
            kappa_y=kappa_y,
            eta_s=eta_s,
            eta_y=eta_y,
            rho=rho,
        )

        names = ["mean_v", "kappa_s", "kappa_y", "eta_s", "eta_y", "rho", "lmbd", "lmbd_s", "lmbd_y"]

        assert param.measure == "P"
        assert param.get_model_name() == "Central Tendency"
        assert param.get_names() == names
        assert param.get_names(subset="all") == names
        assert param.get_names(subset="vol") == names[:5] + names[-2:]
        assert param.get_names(subset="vol", measure="P") == names[:5]
        assert param.get_names(subset="vol", measure="Q") == names[:5]
        assert param.get_names(subset="all", measure="P") == names[:-2]
        assert param.get_names(subset="all", measure="Q") == names[:-2]

        assert param.riskfree == riskfree
        assert param.lmbd == lmbd
        assert param.lmbd_s == 0.0
        assert param.lmbd_y == 0.0
        assert param.mean_v == mean_v
        assert param.kappa_s == kappa_s
        assert param.kappa_y == kappa_y
        assert param.eta_s == eta_s
        assert param.eta_y == eta_y
        assert param.rho == rho
        assert param.is_valid()

    def test_constraints(self) -> None:
        """Test constraints."""
        riskfree = 0.01
        lmbd = 0.01
        mean_v = 0.5
        kappa_s = 1.5
        kappa_y = 0.5
        eta_s = 0.1
        eta_y = 0.01
        rho = -0.5

        param = CentTendParam(
            riskfree=riskfree,
            lmbd=lmbd,
            mean_v=mean_v,
            kappa_s=kappa_s,
            kappa_y=kappa_y,
            eta_s=eta_s,
            eta_y=eta_y,
            rho=rho,
        )

        cons = param.get_constraints()
        assert cons[0]["fun"](param.get_theta()) > 0
        assert cons[1]["fun"](param.get_theta()) > 0

        riskfree = 0.01
        lmbd = 0.01
        mean_v = 0.5
        kappa_s = 0.5
        kappa_y = 1.5
        eta_s = 0.01
        eta_y = 0.1
        rho = -0.5

        param = CentTendParam(
            riskfree=riskfree,
            lmbd=lmbd,
            mean_v=mean_v,
            kappa_s=kappa_s,
            kappa_y=kappa_y,
            eta_s=eta_s,
            eta_y=eta_y,
            rho=rho,
        )

        cons = param.get_constraints()
        assert not (cons[0]["fun"](param.get_theta()) > 0)
        assert not (cons[1]["fun"](param.get_theta()) > 0)

    def test_init_q(self) -> None:
        """Test initialization under Q."""
        riskfree = 0.01
        lmbd = 0.01
        lmbd_s = 0.5
        lmbd_y = 0.5
        mean_v = 0.5
        kappa_s = 1.5
        kappa_y = 0.5
        eta_s = 0.1
        eta_y = 0.01
        rho = -0.5

        param = CentTendParam(
            riskfree=riskfree,
            lmbd=lmbd,
            lmbd_s=lmbd_s,
            lmbd_y=lmbd_y,
            mean_v=mean_v,
            kappa_s=kappa_s,
            kappa_y=kappa_y,
            eta_s=eta_s,
            eta_y=eta_y,
            rho=rho,
            measure="Q",
        )

        kappa_sq = kappa_s - lmbd_s * eta_s
        kappa_yq = kappa_y - lmbd_y * eta_y
        scale = kappa_s / kappa_sq

        assert param.measure == "Q"
        assert param.riskfree == riskfree
        assert param.lmbd == 0
        assert param.lmbd_s == lmbd_s
        assert param.lmbd_y == lmbd_y
        assert param.mean_v == mean_v * kappa_y / kappa_yq * scale
        assert param.kappa_s == kappa_sq
        assert param.kappa_y == kappa_yq
        assert param.eta_s == eta_s
        assert param.eta_y == eta_y * scale**0.5
        assert param.rho == rho
        assert param.is_valid()

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            param.convert_to_q()

    def test_from_theta(self) -> None:
        """Test from theta."""
        riskfree = 0.01
        lmbd = 0.01
        lmbd_s = 0.5
        lmbd_y = 0.5
        mean_v = 0.5
        kappa_s = 1.5
        kappa_y = 0.5
        eta_s = 0.1
        eta_y = 0.01
        rho = -0.5

        theta = [riskfree, mean_v, kappa_s, kappa_y, eta_s, eta_y, rho, lmbd, lmbd_s, lmbd_y]
        param = CentTendParam.from_theta(theta, measure="P")

        assert param.measure == "P"
        assert param.riskfree == riskfree
        assert param.lmbd == lmbd
        assert param.lmbd_s == lmbd_s
        assert param.lmbd_y == lmbd_y
        assert param.mean_v == mean_v
        assert param.kappa_s == kappa_s
        assert param.kappa_y == kappa_y
        assert param.eta_s == eta_s
        assert param.eta_y == eta_y
        assert param.rho == rho
        assert param.is_valid()

    def test_from_theta_q(self) -> None:
        """Test from theta under Q."""
        riskfree = 0.01
        lmbd = 0.01
        lmbd_s = 0.5
        lmbd_y = 0.5
        mean_v = 0.5
        kappa_s = 1.5
        kappa_y = 0.5
        eta_s = 0.1
        eta_y = 0.01
        rho = -0.5

        theta = [riskfree, mean_v, kappa_s, kappa_y, eta_s, eta_y, rho, lmbd, lmbd_s, lmbd_y]
        param = CentTendParam.from_theta(theta, measure="Q")

        kappa_sq = kappa_s - lmbd_s * eta_s
        kappa_yq = kappa_y - lmbd_y * eta_y
        scale = kappa_s / kappa_sq

        assert param.measure == "Q"
        assert param.riskfree == riskfree
        assert param.lmbd == 0
        assert param.lmbd_s == lmbd_s
        assert param.lmbd_y == lmbd_y
        assert param.mean_v == mean_v * kappa_y / kappa_yq * scale
        assert param.kappa_s == kappa_sq
        assert param.kappa_y == kappa_yq
        assert param.eta_s == eta_s
        assert param.eta_y == eta_y * scale**0.5
        assert param.rho == rho
        assert param.is_valid()

    def test_convert_to_q(self) -> None:
        """Test conversion to Q."""
        riskfree = 0.01
        lmbd = 0.01
        lmbd_s = 0.5
        lmbd_y = 0.5
        mean_v = 0.5
        kappa_s = 1.5
        kappa_y = 0.5
        eta_s = 0.1
        eta_y = 0.01
        rho = -0.5

        theta = [riskfree, mean_v, kappa_s, kappa_y, eta_s, eta_y, rho, lmbd, lmbd_s, lmbd_y]
        param = CentTendParam.from_theta(theta)
        param.convert_to_q()

        kappa_sq = kappa_s - lmbd_s * eta_s
        kappa_yq = kappa_y - lmbd_y * eta_y
        scale = kappa_s / kappa_sq

        assert param.measure == "Q"
        assert param.riskfree == riskfree
        assert param.lmbd == 0
        assert param.lmbd_s == lmbd_s
        assert param.lmbd_y == lmbd_y
        assert param.mean_v == mean_v * kappa_y / kappa_yq * scale
        assert param.kappa_s == kappa_sq
        assert param.kappa_y == kappa_yq
        assert param.eta_s == eta_s
        assert param.eta_y == eta_y * scale**0.5
        assert param.rho == rho
        assert param.is_valid()

    def test_ajd_matrices(self) -> None:
        """Test AJD matrices."""
        riskfree = 0.01
        lmbd = 0.01
        lmbd_s = 0.5
        lmbd_y = 0.5
        mean_v = 0.5
        kappa_s = 1.5
        kappa_y = 0.5
        eta_s = 0.1
        eta_y = 0.01
        rho = -0.5

        param = CentTendParam(
            riskfree=riskfree,
            lmbd=lmbd,
            lmbd_s=lmbd_s,
            lmbd_y=lmbd_y,
            mean_v=mean_v,
            kappa_s=kappa_s,
            kappa_y=kappa_y,
            eta_s=eta_s,
            eta_y=eta_y,
            rho=rho,
        )

        kappa_sq = kappa_s - lmbd_s * eta_s
        kappa_yq = kappa_y - lmbd_y * eta_y
        scale = kappa_s / kappa_sq
        mean_vq = mean_v * kappa_y / kappa_yq * scale
        eta_yq = eta_y * scale**0.5

        mat_k0 = [riskfree, 0.0, kappa_y * mean_v]
        mat_k1 = [[0, lmbd - 0.5, 0], [0, -kappa_s, kappa_s], [0, 0, -kappa_y]]
        mat_h0 = np.zeros((3, 3))
        mat_h1 = np.zeros((3, 3, 3))
        mat_h1[1, 0] = [1, eta_s * param.rho, 0]
        mat_h1[1, 1] = [eta_s * param.rho, eta_s**2, 0]
        mat_h1[2, 2, 2] = eta_y**2

        npt.assert_array_almost_equal(param.mat_k0, mat_k0)
        npt.assert_array_equal(param.mat_k1, mat_k1)
        npt.assert_array_equal(param.mat_h0, mat_h0)
        npt.assert_array_equal(param.mat_h1, mat_h1)

        param.convert_to_q()

        mat_k0 = [riskfree, 0.0, kappa_yq * mean_vq]
        mat_k1 = [[0, -0.5, 0], [0, -kappa_sq, kappa_sq], [0, 0, -kappa_yq]]
        mat_h0 = np.zeros((3, 3))
        mat_h1 = np.zeros((3, 3, 3))
        mat_h1[1, 0] = [1, eta_s * param.rho, 0]
        mat_h1[1, 1] = [eta_s * param.rho, eta_s**2, 0]
        mat_h1[2, 2, 2] = eta_yq**2

        npt.assert_array_almost_equal(param.mat_k0, mat_k0)
        npt.assert_array_equal(param.mat_k1, mat_k1)
        npt.assert_array_equal(param.mat_h0, mat_h0)
        npt.assert_array_equal(param.mat_h1, mat_h1)

        theta = np.ones(9)
        theta_vol = np.concatenate((theta[:5], theta[-2:]))
        param = CentTendParam()
        param.update(theta=theta)
        npt.assert_array_equal(param.get_theta(), theta)
        npt.assert_array_equal(param.get_theta(subset="vol"), theta_vol)

        mat_k0 = [param.riskfree, 0.0, param.kappa_y * param.mean_v]
        mat_k1 = [[0, param.lmbd - 0.5, 0], [0, -param.kappa_s, param.kappa_s], [0, 0, -param.kappa_y]]
        mat_h0 = np.zeros((3, 3))
        mat_h1 = np.zeros((3, 3, 3))
        mat_h1[1, 0] = [1, param.eta_s * param.rho, 0]
        mat_h1[1, 1] = [param.eta_s * param.rho, param.eta_s**2, 0]
        mat_h1[2, 2, 2] = param.eta_y**2

        npt.assert_array_equal(param.mat_k0, mat_k0)
        npt.assert_array_equal(param.mat_k1, mat_k1)
        npt.assert_array_equal(param.mat_h0, mat_h0)
        npt.assert_array_equal(param.mat_h1, mat_h1)

    def test_update(self) -> None:
        """Test update."""
        riskfree = 0.01
        lmbd = 0.01
        lmbd_s = 0.5
        lmbd_y = 0.5
        mean_v = 0.5
        kappa_s = 1.5
        kappa_y = 0.5
        eta_s = 0.1
        eta_y = 0.01
        rho = -0.5

        param = CentTendParam(
            riskfree=riskfree,
            lmbd=lmbd,
            lmbd_s=lmbd_s,
            lmbd_y=lmbd_y,
            mean_v=mean_v,
            kappa_s=kappa_s,
            kappa_y=kappa_y,
            eta_s=eta_s,
            eta_y=eta_y,
            rho=rho,
        )

        mean_v, kappa_s, kappa_y = 0.6, 1.7, 0.6
        eta_s, eta_y, rho, lmbd = 0.2, 0.02, -0.6, 0.05
        theta = np.array([mean_v, kappa_s, kappa_y, eta_s, eta_y, rho, lmbd])
        param.update(theta=theta, measure="Q")

        kappa_sq = kappa_s - lmbd_s * eta_s
        kappa_yq = kappa_y - lmbd_y * eta_y
        scale = kappa_s / kappa_sq
        mean_vq = mean_v * kappa_y / kappa_yq * scale
        eta_yq = eta_y * scale**0.5

        assert param.measure == "Q"
        assert param.riskfree == riskfree
        assert param.lmbd == 0
        assert param.lmbd_s == lmbd_s
        assert param.lmbd_y == lmbd_y
        assert param.mean_v == pytest.approx(mean_vq)
        assert param.kappa_s == kappa_sq
        assert param.kappa_y == kappa_yq
        assert param.eta_s == eta_s
        assert param.eta_y == eta_yq
        assert param.rho == rho
        assert param.is_valid()

        mat_k0 = [riskfree, 0.0, kappa_yq * mean_vq]
        mat_k1 = [[0, -0.5, 0], [0, -kappa_sq, kappa_sq], [0, 0, -kappa_yq]]

        npt.assert_array_almost_equal(param.mat_k0, mat_k0)
        npt.assert_array_equal(param.mat_k1, mat_k1)

    def test_get_theta(self) -> None:
        """Test get theta."""
        riskfree = 0.01
        lmbd = 0.01
        lmbd_s = 0.5
        lmbd_y = 0.5
        mean_v = 0.5
        kappa_s = 1.5
        kappa_y = 0.5
        eta_s = 0.1
        eta_y = 0.01
        rho = -0.5

        param = CentTendParam(
            riskfree=riskfree,
            lmbd=lmbd,
            lmbd_s=lmbd_s,
            lmbd_y=lmbd_y,
            mean_v=mean_v,
            kappa_s=kappa_s,
            kappa_y=kappa_y,
            eta_s=eta_s,
            eta_y=eta_y,
            rho=rho,
            measure="P",
        )

        theta = [mean_v, kappa_s, kappa_y, eta_s, eta_y, rho, lmbd, lmbd_s, lmbd_y]
        theta_vol = np.concatenate((theta[:5], theta[-2:]))

        npt.assert_array_equal(param.get_theta(), theta)
        npt.assert_array_equal(param.get_theta(subset="all"), theta)
        npt.assert_array_equal(param.get_theta(subset="all", measure="PQ"), theta)
        npt.assert_array_equal(param.get_theta(subset="all", measure="P"), theta[:-2])
        npt.assert_array_equal(param.get_theta(subset="all", measure="Q"), theta[:-2])
        npt.assert_array_equal(param.get_theta(subset="vol"), theta_vol)
        npt.assert_array_equal(param.get_theta(subset="vol", measure="PQ"), theta_vol)
        npt.assert_array_equal(param.get_theta(subset="vol", measure="P"), theta_vol[:-2])
        npt.assert_array_equal(param.get_theta(subset="vol", measure="Q"), theta_vol[:-2])

        theta = np.arange(9)
        param.update(theta=theta)
        theta_vol = np.ones(5) * 2
        param.update(theta=theta_vol, subset="vol", measure="P")
        theta[:5] = theta_vol
        npt.assert_array_equal(param.get_theta(), theta)

    def test_bounds(self) -> None:
        """Test bounds."""
        param = CentTendParam()
        bounds = param.get_bounds()
        assert bounds is not None and len(bounds) == 9
        bounds_all = param.get_bounds(subset="all")
        assert bounds_all is not None and len(bounds_all) == 9
        bounds_pq = param.get_bounds(subset="all", measure="PQ")
        assert bounds_pq is not None and len(bounds_pq) == 9
        bounds_p = param.get_bounds(subset="all", measure="P")
        assert bounds_p is not None and len(bounds_p) == 7
        bounds_q = param.get_bounds(subset="all", measure="Q")
        assert bounds_q is not None and len(bounds_q) == 7
        bounds_vol = param.get_bounds(subset="vol")
        assert bounds_vol is not None and len(bounds_vol) == 7
        bounds_vol_pq = param.get_bounds(subset="vol", measure="PQ")
        assert bounds_vol_pq is not None and len(bounds_vol_pq) == 7
        bounds_vol_p = param.get_bounds(subset="vol", measure="P")
        assert bounds_vol_p is not None and len(bounds_vol_p) == 5
        bounds_vol_q = param.get_bounds(subset="vol", measure="Q")
        assert bounds_vol_q is not None and len(bounds_vol_q) == 5

    def test_validity(self) -> None:
        """Test validity."""
        riskfree = 0.01
        lmbd = 0.01
        lmbd_s = 0.5
        lmbd_y = 0.5
        mean_v = 0.5
        kappa_s = 1.5
        kappa_y = 0.5
        eta_s = 0.1
        eta_y = 0.01
        rho = -0.5

        param = CentTendParam(
            riskfree=riskfree,
            lmbd=lmbd,
            lmbd_s=lmbd_s,
            lmbd_y=lmbd_y,
            mean_v=mean_v,
            kappa_s=kappa_s,
            kappa_y=kappa_y,
            eta_s=eta_s,
            eta_y=eta_y,
            rho=rho,
            measure="P",
        )

        assert param.is_valid()

        param = CentTendParam(
            riskfree=riskfree,
            lmbd=lmbd,
            mean_v=-mean_v,
            kappa_s=kappa_s,
            kappa_y=kappa_y,
            eta_s=eta_s,
            eta_y=eta_y,
            rho=rho,
        )

        assert not param.is_valid()

        param = CentTendParam(
            riskfree=riskfree,
            lmbd=lmbd,
            mean_v=mean_v,
            kappa_s=-kappa_s,
            kappa_y=kappa_y,
            eta_s=eta_s,
            eta_y=eta_y,
            rho=rho,
        )

        assert not param.is_valid()

        param = CentTendParam(
            riskfree=riskfree,
            lmbd=lmbd,
            mean_v=mean_v,
            kappa_s=kappa_s,
            kappa_y=-kappa_y,
            eta_s=eta_s,
            eta_y=eta_y,
            rho=rho,
        )

        assert not param.is_valid()

        param = CentTendParam(
            riskfree=riskfree,
            lmbd=lmbd,
            mean_v=mean_v,
            kappa_s=kappa_s,
            kappa_y=kappa_y,
            eta_s=-eta_s,
            eta_y=eta_y,
            rho=rho,
        )

        assert not param.is_valid()

        param = CentTendParam(
            riskfree=riskfree,
            lmbd=lmbd,
            mean_v=mean_v,
            kappa_s=kappa_s,
            kappa_y=kappa_y,
            eta_s=eta_s,
            eta_y=-eta_y,
            rho=rho,
        )

        assert not param.is_valid()
