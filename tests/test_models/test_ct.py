"""Test suite for Central Tendency model estimation."""

from __future__ import annotations

import numpy as np
import pytest

from affidiff.models.ct import CentTend
from affidiff.parameters.ct import CentTendParam
from affidiff.types import Measure, Subset


@pytest.fixture
def param_true() -> CentTendParam:
    """Return parameters with both volatility factors identifiable from daily data."""
    return CentTendParam(riskfree=0.0, lmbd=0.5, mean_v=0.2, kappa_s=1.5, kappa_y=0.1, eta_s=0.5, eta_y=0.1, rho=-0.9)


def test_integrated_gmm_single(param_true: CentTendParam) -> None:
    """Test integrated GMM estimation of volatility parameters under P measure.

    Volatility moments depend on eta_s and eta_y only through E[RV^2], so that is checked instead of each eta.
    """
    mean_v = param_true.mean_v
    mean_vol2 = CentTend.mean_vol2(param=param_true, aggh=1)
    centtend = CentTend(param_true)

    nperiods, nsub, ndiscr, nsim = 5000, 80, 10, 1
    aggh = 1
    data = centtend.sim_realized(
        nsub=nsub, ndiscr=ndiscr, aggh=aggh, nperiods=nperiods, nsim=nsim, diff=0, cython=True, seed=42
    )
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
    param_est = CentTendParam()
    param_est.update(theta=res.theta, subset=Subset.vol, measure=Measure.P)

    assert np.all(np.isfinite(res.theta))
    assert np.all(res.theta > 0)
    assert param_est.mean_v == pytest.approx(mean_v, rel=0.2)
    assert CentTend.mean_vol2(param=param_est, aggh=aggh) == pytest.approx(mean_vol2, rel=0.5)


def test_integrated_mom_zero_mean_at_true_parameters(param_true: CentTendParam) -> None:
    """Test instrumented volatility moment conditions average to zero at the true parameters."""
    centtend = CentTend(param_true)
    data = centtend.sim_realized(nsub=80, ndiscr=10, aggh=1, nperiods=20_000, nsim=1, diff=0, cython=True, seed=42)
    _, rvar = data
    moms, _ = centtend.integrated_mom(
        param_true.get_theta(subset=Subset.vol, measure=Measure.P),
        data=data,
        instr_data=np.vstack([rvar]),
        instr_choice="var",
        aggh=1,
        subset=Subset.vol,
        instrlag=3,
        measure=Measure.P,
    )

    # Block means give standard errors robust to the moments' serial correlation.
    nblocks = 20
    block_means = moms[: len(moms) // nblocks * nblocks].reshape(nblocks, -1, moms.shape[1]).mean(1)
    tstats = block_means.mean(0) / block_means.std(0, ddof=1) * np.sqrt(nblocks)
    assert np.all(np.abs(tstats) < 4)


@pytest.mark.parametrize("method", ["L-BFGS-B", "TNC", "SLSQP"])
def test_integrated_gmm_opt_methods(*, method: str) -> None:
    """Test integrated GMM converges from a perturbed start for every optimization method."""
    param_true = CentTendParam(
        riskfree=0.0, lmbd=0.01, mean_v=0.2, kappa_s=0.1, kappa_y=0.05, eta_s=0.1, eta_y=0.03, rho=-0.9
    )
    centtend = CentTend(param_true)

    nperiods, nsub, ndiscr, nsim = 500, 80, 1, 1
    aggh = 1
    start = [1, param_true.mean_v, param_true.mean_v]
    data = centtend.sim_realized(
        start=start, nsub=nsub, ndiscr=ndiscr, aggh=aggh, nperiods=nperiods, nsim=nsim, diff=0, seed=42
    )
    _, rvar = data
    instr_data = np.vstack([rvar, rvar**2])

    param_start = CentTendParam(
        riskfree=0.0, lmbd=0.01, mean_v=0.2, kappa_s=0.1, kappa_y=0.05, eta_s=0.1, eta_y=0.03, rho=-0.9
    )
    param_start.update(theta=param_true.get_theta() / 2)

    res = centtend.integrated_gmm(
        param_start=param_start,
        data=data,
        instrlag=1,
        instr_data=instr_data,
        aggh=aggh,
        instr_choice="var",
        method=method,
        subset=Subset.vol,
        iter=3,
    )

    assert len(res.theta) == 5
    assert np.all(np.isfinite(res.theta))
    assert np.all(res.theta >= 0)
