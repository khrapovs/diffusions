"""Test suite for GBM parameter class."""

import numpy as np
import numpy.testing as npt

from affidiff import GBMparam


class TestSDEParameter:
    """Test parameter classes."""

    def test_gbmparam_class(self) -> None:
        """Test GBM parameter class."""
        mean, sigma = 1.5, 0.2
        param = GBMparam(mean, sigma)

        assert param.get_model_name() == "GBM"
        assert param.get_names() == ["mean", "sigma"]

        assert param.mean == mean
        assert param.sigma == sigma
        npt.assert_array_equal(param.get_theta(), np.array([mean, sigma]))

        theta = np.array([mean, sigma])
        npt.assert_array_equal(param.get_theta(), theta)

        theta = np.ones(2)
        param = GBMparam.from_theta(theta)
        npt.assert_array_equal(param.get_theta(), theta)

        mat_k0 = param.mean - param.sigma**2 / 2
        mat_k1 = 0.0
        mat_h0 = param.sigma**2
        mat_h1 = 0.0

        npt.assert_array_equal(param.mat_k0, mat_k0)
        npt.assert_array_equal(param.mat_k1, mat_k1)
        npt.assert_array_equal(param.mat_h0, mat_h0)
        npt.assert_array_equal(param.mat_h1, mat_h1)

        theta *= 2
        param.update(theta=theta)
        npt.assert_array_equal(param.get_theta(), theta)

        mat_k0 = param.mean - param.sigma**2 / 2
        mat_k1 = 0.0
        mat_h0 = param.sigma**2
        mat_h1 = 0.0

        npt.assert_array_equal(param.mat_k0, mat_k0)
        npt.assert_array_equal(param.mat_k1, mat_k1)
        npt.assert_array_equal(param.mat_h0, mat_h0)
        npt.assert_array_equal(param.mat_h1, mat_h1)

        assert param.is_valid()
        param = GBMparam(mean, -sigma)
        assert not param.is_valid()
