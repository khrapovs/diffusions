"""Test suite for generic classes."""

from functools import partial

import numpy as np
import numpy.testing as npt
import pytest
from scipy.optimize import approx_fprime

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
from affidiff.types import Measure, Subset


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


@pytest.mark.parametrize(
    "model",
    [
        Heston(HestonParam(lmbd=0.5, mean_v=0.2, kappa=1.5, eta=0.5, rho=-0.9)),
        CentTend(CentTendParam(lmbd=0.5, mean_v=0.2, kappa_s=1.5, kappa_y=0.1, eta_s=0.5, eta_y=0.1, rho=-0.9)),
    ],
)
@pytest.mark.parametrize("subset", [Subset.all, Subset.vol])
def test_integrated_mom_jacobian_matches_numerical_derivative(*, model: SDE, subset: Subset) -> None:
    assert model.param is not None
    data = model.sim_realized(nsub=20, ndiscr=1, aggh=1, nperiods=300, nsim=1, diff=0, seed=0)
    theta = model.param.get_theta(subset=subset, measure=Measure.P)
    compute_moments = partial(
        model.integrated_mom, data=data, instr_data=np.vstack([data[1]]), instr_choice="var", instrlag=2, subset=subset
    )

    def compute_mean_moments(theta_x: np.ndarray) -> np.ndarray:
        return compute_moments(theta_x)[0].mean(0)

    _, dmom = compute_moments(theta)

    npt.assert_allclose(dmom, approx_fprime(theta, compute_mean_moments), rtol=1e-4, atol=1e-6)
