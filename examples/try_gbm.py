"""Try Geometric Brownian Motion."""

from __future__ import annotations

import numpy as np

from affidiff.helper_functions import plot_realized
from affidiff.model_gbm import GBM
from affidiff.param_gbm import GBMparam


def try_integrated_gmm() -> None:
    """Try Integrated GMM for GBM model."""
    mean, sigma = 1.5, 0.2
    theta_true = GBMparam(mean=mean, sigma=sigma)
    gbm = GBM(theta_true)

    start, nperiods, nsub, ndiscr, nsim = 1, 500, 80, 1, 1
    aggh = 10
    returns, rvar = gbm.sim_realized(
        start=start, nsub=nsub, ndiscr=ndiscr, aggh=aggh, nperiods=nperiods, nsim=nsim, diff=0
    )
    data = np.vstack([returns, rvar])
    print(rvar.mean() ** 0.5)
    plot_realized(returns=returns, rvar=rvar)

    mean, sigma = 2.5, 0.4
    theta_start = GBMparam(mean=mean, sigma=sigma)
    res = gbm.integrated_gmm(param_start=theta_start, data=data, instrlag=2)
    print(res)


if __name__ == "__main__":
    try_integrated_gmm()
