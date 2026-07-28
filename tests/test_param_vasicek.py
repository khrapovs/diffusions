"""Test suite for Vasicek parameter class."""

import numpy as np
import numpy.testing as npt

from affidiff import VasicekParam


class TestSDEParameter:
    """Test parameter classes."""

    def test_vasicekparam_class(self) -> None:
        """Test Vasicek parameter class."""
        mean, kappa, eta = 1.5, 1.0, 0.2
        param = VasicekParam(mean, kappa, eta)

        assert param.get_model_name() == "Vasicek"
        assert param.get_names() == ["mean", "kappa", "eta"]

        assert param.mean == mean
        assert param.kappa == kappa
        assert param.eta == eta

        npt.assert_array_equal(param.get_theta(), np.array([mean, kappa, eta]))

        theta = np.ones(3)
        param = VasicekParam.from_theta(theta)
        npt.assert_array_equal(param.get_theta(), theta)

        mat_k0 = param.kappa * param.mean
        mat_k1 = -param.kappa
        mat_h0 = param.eta**2
        mat_h1 = 0

        npt.assert_array_equal(param.mat_k0, mat_k0)
        npt.assert_array_equal(param.mat_k1, mat_k1)
        npt.assert_array_equal(param.mat_h0, mat_h0)
        npt.assert_array_equal(param.mat_h1, mat_h1)

        theta *= 2
        param.update(theta)
        npt.assert_array_equal(param.get_theta(), theta)

        mat_k0 = param.kappa * param.mean
        mat_k1 = -param.kappa
        mat_h0 = param.eta**2
        mat_h1 = 0

        npt.assert_array_equal(param.mat_k0, mat_k0)
        npt.assert_array_equal(param.mat_k1, mat_k1)
        npt.assert_array_equal(param.mat_h0, mat_h0)
        npt.assert_array_equal(param.mat_h1, mat_h1)

        assert param.is_valid()
        param = VasicekParam(mean, -kappa, eta)
        assert not param.is_valid()
        param = VasicekParam(mean, kappa, -eta)
        assert not param.is_valid()
