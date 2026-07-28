"""Try Cox-Ingersoll-Ross Model."""

from __future__ import annotations

import seaborn as sns

from affidiff import CIR, CIRparam
from affidiff.helper_functions import plot_final_distr, plot_realized, plot_trajectories, take_time


def try_simulation() -> None:
    """Try simulating and plotting CIR model."""
    mean, kappa, eta = 0.5, 0.1, 0.2
    theta_true = CIRparam(mean, kappa, eta)
    # 2 * kappa * mean - eta**2 > 0
    print(theta_true.is_valid())
    cir = CIR(theta_true)

    x0, nperiods, nsub, ndiscr, nsim = mean, 500, 2, 10, 3
    nobs = nperiods * nsub
    paths = cir.simulate(x0, nsub=nsub, ndiscr=ndiscr, nobs=nobs, nsim=nsim)
    data = paths[:, 0, 0]

    plot_trajectories(data, nsub, "rates")


def try_marginal() -> None:
    """Try marginal distribution of CIR model."""
    mean, kappa, eta = 0.5, 0.1, 0.2
    theta_true = CIRparam(mean, kappa, eta)
    # 2 * kappa * mean - eta**2 > 0
    print(theta_true.is_valid())
    cir = CIR(theta_true)

    x0, nperiods, nsub, ndiscr, nsim = mean, 500, 2, 10, 20
    nobs = nperiods * nsub
    paths = cir.simulate(x0, nsub=nsub, ndiscr=ndiscr, nobs=nobs, nsim=nsim)
    data = paths[:, :, 0]

    plot_final_distr(data, "rates")


def try_sim_realized() -> None:
    """Try simulated realized CIR model data."""
    mean, kappa, eta = 0.5, 0.1, 0.2
    theta_true = CIRparam(mean, kappa, eta)
    cir = CIR(theta_true)

    start, nperiods, nsub, ndiscr, nsim = 1, 500, 80, 1, 1
    aggh = 10
    returns, rvar = cir.sim_realized(start, nsub=nsub, ndiscr=ndiscr, aggh=aggh, nperiods=nperiods, nsim=nsim, diff=0)

    plot_realized(returns, rvar)


if __name__ == "__main__":
    sns.set_context("notebook")
    with take_time("Simulation"):
        try_simulation()
    with take_time("Marginal density"):
        try_marginal()
    with take_time("Simulate realized"):
        try_sim_realized()
