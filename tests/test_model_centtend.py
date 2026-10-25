"""Test suite for Central Tendency model estimation."""

from __future__ import annotations

import numpy as np

from affidiff.model_ct import CentTend
from affidiff.param_ct import CentTendParam
from affidiff.types import Measure, Subset


def test_integrated_gmm_single() -> None:
    """Test integrated GMM estimation of volatility parameters under P measure."""
    mean_v, eta_s, eta_y = 0.2, 0.5, 0.05
    param_true = CentTendParam(
        riskfree=0.0, lmbd=0.5, mean_v=mean_v, kappa_s=1.5, kappa_y=0.008, eta_s=eta_s, eta_y=eta_y, rho=-0.9
    )
    centtend = CentTend(param_true)

    nperiods, nsub, ndiscr, nsim = 500, 80, 10, 1
    aggh = 1
    data = centtend.sim_realized(nsub=nsub, ndiscr=ndiscr, aggh=aggh, nperiods=nperiods, nsim=nsim, diff=0, seed=42)
    _, rvar = data
    instr_data = np.vstack([rvar])

    res = centtend.integrated_gmm(
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

    mean_v_est, _, _, eta_s_est, eta_y_est = res.theta
    assert np.all(np.isfinite(res.theta))
    assert np.all(res.theta > 0)
    assert np.abs(mean_v_est - mean_v) / mean_v < 0.5
    assert np.abs(eta_s_est - eta_s) / eta_s < 0.5
    assert np.abs(eta_y_est - eta_y) / eta_y < 0.5
