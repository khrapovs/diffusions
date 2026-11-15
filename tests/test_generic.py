"""Test suite for generic classes."""

import pytest

from affidiff.models.cir import CIR
from affidiff.models.ct import CentTend
from affidiff.models.gbm import GBM
from affidiff.models.generic import SDE
from affidiff.models.heston import Heston
from affidiff.models.vasicek import Vasicek
from affidiff.parameters.cir import CIRparam
from affidiff.parameters.ct import CentTendParam
from affidiff.parameters.gbm import GBMparam
from affidiff.parameters.heston import HestonParam
from affidiff.parameters.vasicek import VasicekParam


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


@pytest.mark.parametrize(
    "model",
    [GBM(GBMparam()), Vasicek(VasicekParam()), CIR(CIRparam()), Heston(HestonParam()), CentTend(CentTendParam())],
)
def test_series_names_match_state_variables(model: SDE) -> None:
    assert len(model.series_names) == model.nvars
