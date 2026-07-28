"""Try Geometric Brownian Motion."""

from __future__ import annotations

import numpy as np
import seaborn as sns

from affidiff import GBM, GBMparam
from affidiff.helper_functions import plot_final_distr, plot_realized, plot_trajectories, take_time


def try_simulation() -> None:
    """Try simulating and plotting GBM model."""
    mean, sigma = 0.05, 0.2
    theta_true = GBMparam(mean, sigma)
    print(theta_true)

    gbm = GBM(theta_true)

    start, nperiods, nsub, ndiscr, nsim = 1, 500, 2, 10, 2
    nobs = nperiods * nsub
    paths = gbm.simulate(start, nsub=nsub, ndiscr=ndiscr, nobs=nobs, nsim=nsim, diff=0)
    data = paths[:, 0, 0]

    plot_trajectories(data, nsub, "returns")


def try_marginal() -> None:
    """Try marginal distribution of GBM model."""
    mean, sigma = 0.05, 0.2
    theta_true = GBMparam(mean, sigma)
    gbm = GBM(theta_true)

    start, nperiods, nsub, ndiscr, nsim = 1, 500, 2, 10, 20
    nobs = nperiods * nsub
    paths = gbm.simulate(start, nsub=nsub, ndiscr=ndiscr, nobs=nobs, nsim=nsim, diff=0)
    data = paths[:, :, 0]

    plot_final_distr(data * nsub, "returns")


def try_gmm() -> None:
    """Try GMM estimation for GBM model."""
    mean, sigma = 1.5, 0.2
    theta_true = GBMparam(mean, sigma)
    gbm = GBM(theta_true)

    start, nperiods, nsub, ndiscr, nsim = 1, 500, 2, 10, 1
    nobs = nperiods * nsub
    paths = gbm.simulate(start, nsub=nsub, ndiscr=ndiscr, nobs=nobs, nsim=nsim, diff=0)
    data = paths[:, 0, 0]

    plot_trajectories(data, nsub, "returns")

    mean, sigma = 2.5, 0.4
    theta_start = GBMparam(mean, sigma)
    res = gbm.gmmest(theta_start, data=data, instrlag=2)
    print(res)


def try_sim_realized() -> None:
    """Try simulated realized GBM model data."""
    mean, sigma = 0.05, 0.2
    theta_true = GBMparam(mean, sigma)
    gbm = GBM(theta_true)

    start, nperiods, nsub, ndiscr, nsim = 1, 500, 80, 1, 1
    aggh = 10
    returns, rvar = gbm.sim_realized(start, nsub=nsub, ndiscr=ndiscr, aggh=aggh, nperiods=nperiods, nsim=nsim, diff=0)

    plot_realized(returns, rvar)


def try_integrated_gmm() -> None:
    """Try Integrated GMM for GBM model."""
    mean, sigma = 1.5, 0.2
    theta_true = GBMparam(mean, sigma)
    gbm = GBM(theta_true)

    start, nperiods, nsub, ndiscr, nsim = 1, 500, 80, 1, 1
    aggh = 10
    returns, rvar = gbm.sim_realized(start, nsub=nsub, ndiscr=ndiscr, aggh=aggh, nperiods=nperiods, nsim=nsim, diff=0)
    data = np.vstack([returns, rvar])
    print(rvar.mean() ** 0.5)
    plot_realized(returns, rvar)

    mean, sigma = 2.5, 0.4
    theta_start = GBMparam(mean, sigma)
    res = gbm.integrated_gmm(theta_start, data=data, instrlag=2)
    print(res)


if __name__ == "__main__":
    sns.set_context("notebook")
    with take_time("Integrated GMM"):
        try_integrated_gmm()
