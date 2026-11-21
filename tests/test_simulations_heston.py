"""Test suite for Heston model simulations under P and Q measures."""

from __future__ import annotations

import numpy as np
import pytest

from affidiff.models.heston import Heston
from affidiff.parameters.heston import HestonParam
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


@pytest.mark.parametrize("use_cython", [False, True])
def test_realized_variance_tracks_instantaneous_variance(*, use_cython: bool) -> None:
    """Test that periods of high instantaneous variance show proportionally high realized variance of log returns."""
    nperiods, nsub = 300, 100
    param = HestonParam(mean_v=0.5, kappa=2.0, eta=0.6, rho=-0.5)
    paths = Heston(param).simulate(nsub=nsub, nobs=nperiods * nsub, nsim=1, diff=0, cython=use_cython, seed=0)
    returns, variance = paths[:, 0, 0], paths[:, 0, 1]
    # Each Euler return is driven by the variance at the start of its step.
    variance_start = np.concatenate(([param.mean_v], variance[:-1]))
    realized = (returns**2).reshape(nperiods, nsub).sum(1)
    integrated = variance_start.reshape(nperiods, nsub).mean(1)

    high = integrated > np.median(integrated)
    realized_ratio = realized[high].mean() / realized[~high].mean()
    integrated_ratio = integrated[high].mean() / integrated[~high].mean()

    assert realized_ratio > 1.5
    assert realized_ratio == pytest.approx(integrated_ratio, rel=0.1)
    assert realized.mean() == pytest.approx(integrated.mean(), rel=0.05)


def test_simulate_backends_agree_when_variance_turns_negative() -> None:
    """Test that Euler excursions of variance below zero stay finite and agree across Python and Cython backends."""
    nsub, nobs = 100, 30_000
    heston = Heston(HestonParam(mean_v=0.5, kappa=1.0, eta=0.9, rho=-0.5))
    paths_py = heston.simulate(nsub=nsub, nobs=nobs, nsim=1, diff=0, cython=False, seed=1)
    paths_cy = heston.simulate(nsub=nsub, nobs=nobs, nsim=1, diff=0, cython=True, seed=1)

    assert paths_cy[..., 1].min() < 0
    np.testing.assert_allclose(paths_py, paths_cy, rtol=0, atol=1e-10)


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
