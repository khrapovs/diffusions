"""Test suite for Central Tendency model simulations under P and Q measures."""

from __future__ import annotations

import numpy as np
import pytest

from affidiff.model_ct import CentTend
from affidiff.param_ct import CentTendParam
from affidiff.types import Measure
from assertions import assert_realized_finite_variation, assert_realized_shape


def build_centtend_param(
    *,
    measure: Measure = Measure.P,
    riskfree: float = 0.01,
    mean_v: float = 0.5,
    kappa_s: float = 1.5,
    kappa_y: float = 0.05,
    eta_s: float = 0.02**0.5,
    eta_y: float = 0.001**0.5,
) -> CentTendParam:
    """Build Central Tendency parameters satisfying the Feller conditions.

    Parameters
    ----------
    measure : Measure
        Measure under which the parameters are defined.
    riskfree : float
        Risk-free rate.
    mean_v : float
        Long-run mean of the central tendency.
    kappa_s : float
        Mean reversion speed of volatility.
    kappa_y : float
        Mean reversion speed of the central tendency.
    eta_s : float
        Volatility of volatility.
    eta_y : float
        Volatility of the central tendency.

    Returns
    -------
    CentTendParam
        Initialized parameter instance.

    """
    return CentTendParam(
        riskfree=riskfree,
        lmbd=1.01,
        lmbd_s=0.5,
        lmbd_y=0.5,
        mean_v=mean_v,
        kappa_s=kappa_s,
        kappa_y=kappa_y,
        eta_s=eta_s,
        eta_y=eta_y,
        rho=-0.9,
        measure=measure,
    )


@pytest.mark.parametrize("measure", [Measure.P, Measure.Q])
def test_param_is_valid(*, measure: Measure) -> None:
    """Test Central Tendency parameters satisfy the Feller conditions under both measures."""
    assert build_centtend_param(measure=measure).is_valid()


def test_sim_realized_pq() -> None:
    """Test realized data simulation under P and Q measures."""
    nperiods, nsub, ndiscr, nsim = 500, 80, 10, 1
    aggh = [1, 1]

    param_true = build_centtend_param(riskfree=0.0, mean_v=0.2, kappa_s=0.1, kappa_y=0.02, eta_s=0.1, eta_y=0.01)
    centtend = CentTend(param_true)

    (ret_p, rvar_p), (ret_q, rvar_q) = centtend.sim_realized_pq(
        nsub=nsub, ndiscr=ndiscr, aggh=aggh, nperiods=nperiods, nsim=nsim, seed=42
    )

    assert_realized_shape(returns=ret_p, rvar=rvar_p, expected_length=nperiods - aggh[0] + 1)
    assert_realized_shape(returns=ret_q, rvar=rvar_q, expected_length=nperiods - aggh[1] + 1)
    assert_realized_finite_variation(returns=ret_p, rvar=rvar_p)
    assert_realized_finite_variation(returns=ret_q, rvar=rvar_q)

    assert not np.allclose(ret_p, ret_q)
