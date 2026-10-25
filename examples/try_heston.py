"""Try Heston model."""

from __future__ import annotations

import time

import numpy as np
import seaborn as sns
from load_real_data import load_data  # type: ignore

from affidiff.helper_functions import plot_realized, take_time
from affidiff.model_heston import Heston
from affidiff.param_heston import HestonParam
from affidiff.types import Subset


def try_integrated_gmm_real() -> None:
    """Estimate Heston model parameters with real data."""
    riskfree = 0.0

    mean_v = 0.2
    kappa = 0.22
    eta = 0.12

    lmbd = 0.3
    rho = -0.5
    # 2 * kappa * mean_v - eta**2 > 0
    param_start = HestonParam(riskfree=riskfree, lmbd=lmbd, mean_v=mean_v, kappa=kappa, eta=eta, rho=rho)
    heston = Heston(param_start)

    aggh = 1

    data = load_data()
    ret, rvar = data
    plot_realized(returns=ret, rvar=rvar)

    instr_data = np.vstack([rvar, rvar**2])

    subset = Subset.vol

    time_start = time.time()
    res = heston.integrated_gmm(
        param_start=param_start,
        data=data,
        instrlag=2,
        instr_data=instr_data,
        aggh=aggh,
        instr_choice="var",
        method="TNC",
        subset=subset,
        iter=3,
    )
    print(res)
    print("Elapsed time = %.2f min" % ((time.time() - time_start) / 60))


if __name__ == "__main__":
    np.set_printoptions(precision=4, suppress=True)
    sns.set_context("notebook")

    #    with take_time('Simulation'):
    #        try_simulation()
    #    with take_time('Marginal density'):
    #        try_marginal()
    #    with take_time('Simulate realized'):
    #        try_sim_realized()
    with take_time("Integrated GMM with real data"):
        try_integrated_gmm_real()
