"""Try Central Tendency model."""

from __future__ import annotations

import itertools
import time

import numpy as np
import seaborn as sns

from affidiff.helper_functions import plot_realized, take_time
from affidiff.model_ct import CentTend
from affidiff.param_ct import CentTendParam
from affidiff.types import Subset


def try_integrated_gmm_opt_methods() -> None:
    """Simulate realized data from Central Tendency model. Estimate parameters.

    Check various optimization methods.

    """
    riskfree = 0.0

    mean_v = 0.2
    kappa_s = 0.1
    kappa_y = 0.05
    eta_s = 0.1
    eta_y = 0.03

    lmbd = 0.01
    rho = -0.9

    param_true = CentTendParam(
        riskfree=riskfree, lmbd=lmbd, mean_v=mean_v, kappa_s=kappa_s, kappa_y=kappa_y, eta_s=eta_s, eta_y=eta_y, rho=rho
    )
    centtend = CentTend(param_true)
    print(param_true)
    print(param_true.is_valid())

    start = [1, mean_v, mean_v]
    nperiods, nsub, ndiscr, nsim = 2000, 80, 1, 1
    aggh = 1
    data = centtend.sim_realized(start=start, nsub=nsub, ndiscr=ndiscr, aggh=aggh, nperiods=nperiods, nsim=nsim, diff=0)
    ret, rvar = data
    plot_realized(returns=ret, rvar=rvar)

    instr_data = np.vstack([rvar, rvar**2])

    param_start = param_true
    param_start.update(theta=param_true.get_theta() / 2)

    tasks = itertools.product(np.arange(1, 4), ["L-BFGS-B", "TNC", "SLSQP"])
    for lag, method in tasks:
        time_start = time.time()
        res = centtend.integrated_gmm(
            param_start=param_start,
            data=data,
            instrlag=int(lag),
            instr_data=instr_data,
            aggh=aggh,
            instr_choice="var",
            method=method,
            subset=Subset.vol,
            iter=3,
        )
        print(res)
        print(lag, method)
        print("Elapsed time = %.2f min" % ((time.time() - time_start) / 60))


if __name__ == "__main__":
    np.set_printoptions(precision=4, suppress=True)
    sns.set_context("notebook")

    #    with take_time('Simulation'):
    #        try_simulation()
    #    with take_time('Marginal density'):
    #        try_marginal()
    #    with take_time('Simulation realized'):
    #        try_sim_realized()
    #    with take_time('Integrated GMM with real data'):
    #        try_integrated_gmm_real()
    with take_time("Integrated GMM with real data"):
        try_integrated_gmm_opt_methods()
