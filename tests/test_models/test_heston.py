"""Test suite for Heston model estimation."""

from __future__ import annotations

import numpy as np
import pytest

from affidiff.models.heston import Heston
from affidiff.parameters.heston import HestonParam
from affidiff.types import Measure, Subset
from assertions import assert_realized_finite_variation, assert_realized_shape


def test_integrated_gmm_single() -> None:
    """Test integrated GMM estimation of volatility parameters under P measure."""
    mean_v, kappa, eta = 0.2, 0.06, 0.15
    param_true = HestonParam(riskfree=0.0, lmbd=0.5, mean_v=mean_v, kappa=kappa, eta=eta, rho=-0.5)
    heston = Heston(param_true)

    nperiods, nsub, ndiscr, nsim = 500, 80, 1, 1
    aggh = 1
    data = heston.sim_realized(nsub=nsub, ndiscr=ndiscr, aggh=aggh, nperiods=nperiods, nsim=nsim, diff=0, seed=42)
    _, rvar = data
    instr_data = np.vstack([rvar, rvar**2])

    res = heston.integrated_gmm(
        param_start=param_true,
        data=data,
        instrlag=3,
        instr_data=instr_data,
        aggh=aggh,
        instr_choice="var",
        method="SLSQP",
        subset=Subset.vol,
        measure=Measure.P,
        iter=3,
    )

    mean_v_est, kappa_est, eta_est = res.theta
    assert np.all(np.isfinite(res.theta))
    assert np.all(res.theta > 0)
    assert np.abs(mean_v_est - mean_v) / mean_v < 0.5
    assert np.abs(eta_est - eta) / eta < 0.5
    # Mean reversion speed is weakly identified from a short sample, so only its magnitude is checked.
    assert kappa_est < 1


def test_integrated_gmm_single_rn() -> None:
    """Test integrated GMM estimation on data simulated under both P and Q measures."""
    mean_v, kappa, eta = 0.5, 0.04, 0.15
    param_true = HestonParam(riskfree=0.0, lmbd=1.5, mean_v=mean_v, kappa=kappa, eta=eta, rho=-0.9, lmbd_v=0.1)
    heston = Heston(param_true)

    nperiods, nsub, ndiscr, nsim = 500, 100, 1, 1
    aggh = [1, 1]

    data_p, data_q = heston.sim_realized_pq(
        nsub=nsub, ndiscr=ndiscr, aggh=aggh, nperiods=nperiods, nsim=nsim, diff=0, seed=42
    )
    ret_p, rvar_p = data_p
    ret_q, rvar_q = data_q
    instr_data = np.vstack([rvar_p, rvar_p**2])

    for returns, rvar, window in [(ret_p, rvar_p, aggh[0]), (ret_q, rvar_q, aggh[1])]:
        assert_realized_shape(returns=returns, rvar=rvar, expected_length=nperiods - window + 1)
        assert_realized_finite_variation(returns=returns, rvar=rvar)

    estimates = []
    for data, window in [(data_p, aggh[0]), (data_q, aggh[1])]:
        res = heston.integrated_gmm(
            param_start=param_true,
            data=data,
            instrlag=2,
            instr_data=instr_data,
            aggh=window,
            instr_choice="var",
            method="TNC",
            subset=Subset.vol,
            measure=Measure.P,
            iter=3,
        )
        assert len(res.theta) == 3
        assert np.all(np.isfinite(res.theta))
        assert np.all(res.theta >= 0)
        estimates.append(res.theta)

    # Q data carries the volatility risk premium, so its estimates must differ from the P ones.
    assert not np.allclose(estimates[0], estimates[1])


def test_integrated_gmm_joint() -> None:
    """Test joint integrated GMM estimation under P and Q measures simultaneously."""
    mean_v, kappa, eta, lmbd_v = 0.5, 0.04, 0.15, 0.1
    param_true = HestonParam(riskfree=0.0, lmbd=1.5, mean_v=mean_v, kappa=kappa, eta=eta, rho=-0.9, lmbd_v=lmbd_v)
    heston = Heston(param_true)

    nperiods, nsub, ndiscr, nsim = 500, 100, 1, 1
    aggh = [1, 1]

    data = heston.sim_realized_pq(nsub=nsub, ndiscr=ndiscr, aggh=aggh, nperiods=nperiods, nsim=nsim, diff=0, seed=42)
    rvar_p = data[0][1]
    instr_data = np.vstack([rvar_p, rvar_p**2])

    res = heston.integrated_gmm(
        param_start=param_true,
        data=data,
        instrlag=2,
        instr_data=instr_data,
        aggh=aggh,
        instr_choice="var",
        method="TNC",
        subset=Subset.vol,
        measure=Measure.PQ,
        iter=3,
    )

    # Joint estimation adds the volatility risk price on top of the three P parameters.
    assert len(res.theta) == 4
    assert np.all(np.isfinite(res.theta))
    assert np.all(res.theta > 0)
    mean_v_est = res.theta[0]
    assert np.abs(mean_v_est - mean_v) / mean_v < 0.5


@pytest.mark.parametrize("instrlag", [1, 2, 3])
@pytest.mark.parametrize("method", ["L-BFGS-B", "TNC", "SLSQP"])
def test_integrated_gmm_opt_methods(*, instrlag: int, method: str) -> None:
    """Test integrated GMM converges for every optimization method and instrument lag."""
    mean_v, kappa, eta = 0.5, 0.1, 0.15
    param_true = HestonParam(riskfree=0.0, lmbd=0.3, mean_v=mean_v, kappa=kappa, eta=eta, rho=-0.5)
    heston = Heston(param_true)

    nperiods, nsub, ndiscr, nsim = 500, 80, 1, 1
    aggh = 10
    data = heston.sim_realized(
        start=[1, mean_v], nsub=nsub, ndiscr=ndiscr, aggh=aggh, nperiods=nperiods, nsim=nsim, diff=0, seed=42
    )
    _, rvar = data
    instr_data = np.vstack([rvar, rvar**2])

    param_start = HestonParam(riskfree=0.0, lmbd=0.3, mean_v=mean_v, kappa=kappa, eta=eta, rho=-0.5)
    param_start.update(theta=param_true.get_theta() / 2)

    res = heston.integrated_gmm(
        param_start=param_start,
        data=data,
        instrlag=instrlag,
        instr_data=instr_data,
        aggh=aggh,
        instr_choice="var",
        method=method,
        subset=Subset.vol,
        iter=3,
    )

    assert len(res.theta) == 3
    assert np.all(np.isfinite(res.theta))
    assert np.all(res.theta >= 0)
