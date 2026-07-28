"""Test suite for CIR parameter class."""

import numpy as np
import numpy.testing as npt

from affidiff import CIRparam


class TestSDEParameter:
    """Test parameter classes."""

    def test_cirparam_class(self) -> None:
        """Test CIR parameter class."""
        mean, kappa, eta = 1.5, 1.0, 0.1
        param = CIRparam(mean=mean, kappa=kappa, eta=eta)

        assert param.get_model_name() == "CIR"
        assert param.get_names() == ["mean", "kappa", "eta"]

        assert param.mean == mean
        assert param.kappa == kappa
        assert param.eta == eta

        npt.assert_array_equal(param.get_theta(), np.array([mean, kappa, eta]))

        theta = np.ones(3)
        param = CIRparam.from_theta(theta=theta)
        npt.assert_array_equal(param.get_theta(), theta)

        mat_k0 = param.kappa * param.mean
        mat_k1 = -param.kappa
        mat_h0 = 0.0
        mat_h1 = param.eta**2

        npt.assert_array_equal(param.mat_k0, mat_k0)
        npt.assert_array_equal(param.mat_k1, mat_k1)
        npt.assert_array_equal(param.mat_h0, mat_h0)
        npt.assert_array_equal(param.mat_h1, mat_h1)

        theta *= 2
        param.update(theta=theta)
        npt.assert_array_equal(param.get_theta(), theta)

        mat_k0 = param.kappa * param.mean
        mat_k1 = -param.kappa
        mat_h0 = 0.0
        mat_h1 = param.eta**2

        npt.assert_array_equal(param.mat_k0, mat_k0)
        npt.assert_array_equal(param.mat_k1, mat_k1)
        npt.assert_array_equal(param.mat_h0, mat_h0)
        npt.assert_array_equal(param.mat_h1, mat_h1)

        assert param.is_valid()
        param = CIRparam(mean=mean, kappa=-kappa, eta=eta)
        assert not param.is_valid()
        param = CIRparam(mean=mean, kappa=kappa, eta=-eta)
        assert not param.is_valid()
