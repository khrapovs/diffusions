"""Test suite for Heston model estimation."""

from __future__ import annotations

import numpy as np

from affidiff.model_heston import Heston
from affidiff.param_heston import HestonParam
from affidiff.types import Measure, Subset


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
