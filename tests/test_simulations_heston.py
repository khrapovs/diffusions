"""Test suite for Heston model simulations under P and Q measures."""

from __future__ import annotations

import numpy as np
import pytest

from affidiff.model_heston import Heston
from affidiff.param_heston import HestonParam
from affidiff.types import Measure
from assertions import (
    assert_finite_values,
    assert_realized_finite_variation,
    assert_realized_shape,
    assert_simulation_shape,
    assert_variation,
)


def build_heston_param(
    *, measure: Measure = Measure.P, lmbd: float = 0.0, lmbd_v: float = 0.5, kappa: float = 0.1
) -> HestonParam:
    """Build Heston parameters satisfying the Feller condition.

    Parameters
    ----------
    measure : Measure
        Measure under which the parameters are defined.
    lmbd : float
        Equity risk price.
    lmbd_v : float
        Volatility risk price.
    kappa : float
        Mean reversion speed of volatility.

    Returns
    -------
    HestonParam
        Initialized parameter instance.

    """
    return HestonParam(
        riskfree=0.0, lmbd=lmbd, lmbd_v=lmbd_v, mean_v=0.5, kappa=kappa, eta=0.15, rho=-0.9, measure=measure
    )


@pytest.mark.parametrize("measure", [Measure.P, Measure.Q])
def test_param_is_valid(*, measure: Measure) -> None:
    """Test Heston parameters satisfy the Feller condition under both measures."""
    assert build_heston_param(measure=measure).is_valid()


def test_simulate_pq_shares_innovations() -> None:
    """Test simulation under P and Q with shared innovations."""
    nperiods, nsub, ndiscr, nsim = 100, 10, 10, 3
    nobs = nperiods * nsub
    expected_nsim = 2 * nsim

    param_p = build_heston_param(measure=Measure.P)
    heston = Heston(param_p)
    start = [1, param_p.mean_v]
    paths_p = heston.simulate(start=start, nsub=nsub, ndiscr=ndiscr, nobs=nobs, nsim=nsim, diff=0)

    param_q = build_heston_param(measure=Measure.Q)
    heston.update_theta(param_q)
    start_q = [1, param_q.mean_v]
    paths_q = heston.simulate(start=start_q, nsub=nsub, ndiscr=ndiscr, nobs=nobs, nsim=nsim, diff=0, new_innov=False)

    for paths in (paths_p, paths_q):
        assert_simulation_shape(paths=paths, nobs=nobs, expected_nsim=expected_nsim, nvars=heston.nvars)
        assert_finite_values(paths=paths)
        assert_variation(paths=paths)

    assert not np.allclose(paths_p, paths_q)


def test_sim_realized_pq() -> None:
    """Test realized data simulation under P and Q measures."""
    nperiods, nsub, ndiscr, nsim = 500, 100, 1, 1
    aggh = [1, 2]

    param_true = build_heston_param(lmbd=1.5, lmbd_v=0.2, kappa=0.04)
    heston = Heston(param_true)

    (ret_p, rvar_p), (ret_q, rvar_q) = heston.sim_realized_pq(
        nsub=nsub, ndiscr=ndiscr, aggh=aggh, nperiods=nperiods, nsim=nsim, diff=0
    )

    assert_realized_shape(returns=ret_p, rvar=rvar_p, expected_length=nperiods - aggh[0] + 1)
    assert_realized_shape(returns=ret_q, rvar=rvar_q, expected_length=nperiods - aggh[1] + 1)
    assert_realized_finite_variation(returns=ret_p, rvar=rvar_p)
    assert_realized_finite_variation(returns=ret_q, rvar=rvar_q)
