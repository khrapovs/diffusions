"""Try Vasicek Model."""

from __future__ import annotations

import seaborn as sns

from affidiff import Vasicek, VasicekParam
from affidiff.helper_functions import plot_final_distr, plot_realized, plot_trajectories, take_time


def try_simulation() -> None:
    """Try simulating and plotting Vasicek model."""
    mean, kappa, eta = 0.5, 0.1, 0.2
    theta_true = VasicekParam(mean, kappa, eta)
    vasicek = Vasicek(theta_true)

    x0, nperiods, nsub, ndiscr, nsim = 1, 500, 2, 10, 3
    nobs = nperiods * nsub
    paths = vasicek.simulate(x0, nsub=nsub, ndiscr=ndiscr, nobs=nobs, nsim=nsim)
    data = paths[:, 0, 0]

    plot_trajectories(data, nsub, "returns")


def try_marginal() -> None:
    """Try marginal distribution of Vasicek model."""
    mean, kappa, eta = 0.5, 0.1, 0.2
    theta_true = VasicekParam(mean, kappa, eta)
    vasicek = Vasicek(theta_true)

    x0, nperiods, nsub, ndiscr, nsim = mean, 500, 2, 10, 20
    nobs = nperiods * nsub
    paths = vasicek.simulate(x0, nsub=nsub, ndiscr=ndiscr, nobs=nobs, nsim=nsim)
    data = paths[:, :, 0]

    plot_final_distr(data, "returns")


def try_sim_realized() -> None:
    """Try simulated realized Vasicek model data."""
    mean, kappa, eta = 0.5, 0.1, 0.2
    theta_true = VasicekParam(mean, kappa, eta)
    vasicek = Vasicek(theta_true)

    start, nperiods, nsub, ndiscr, nsim = [1.0], 500, 80, 1, 1
    aggh = 10
    returns, rvar = vasicek.sim_realized(
        start, nsub=nsub, ndiscr=ndiscr, aggh=aggh, nperiods=nperiods, nsim=nsim, diff=0
    )

    plot_realized(returns, rvar)


if __name__ == "__main__":
    sns.set_context("notebook")
    with take_time("Marginal density"):
        try_marginal()
    with take_time("Simulation"):
        try_simulation()
    with take_time("Simulate realized"):
        try_sim_realized()
