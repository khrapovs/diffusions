"""Test suite for generic classes."""

from affidiff import GBM, GBMparam


class TestGenericModel:
    """Test generic model."""

    def test_update_theta(self) -> None:
        """Test update of true parameter."""
        mean, sigma = 1.5, 0.2
        param = GBMparam(mean=mean, sigma=sigma)
        gbm = GBM(param)
        param_new = GBMparam(mean=2 * mean, sigma=2 * sigma)
        gbm.update_theta(param_new)

        assert gbm.param == param_new
