"""Test suite for diffusion model simulations across all models and backends."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest

from conftest import (
    TestSimulationConfig,
    assert_finite_values,
    assert_simulation_shape,
    assert_statistical_equivalence,
    assert_variation,
    get_model_fixtures,
)

if TYPE_CHECKING:
    from affidiff.model_generic import SDE
    from affidiff.param_generic import GenericParam


@pytest.mark.parametrize("model_class,params,nvars", get_model_fixtures())
@pytest.mark.parametrize("use_cython", [False, True])
def test_simulate_all_models(*, model_class: type[SDE], params: GenericParam, nvars: int, use_cython: bool) -> None:
    """Test simulation runs successfully for all models with both backends.

    Validates that:
    - Simulation completes without error
    - Output has correct shape (nobs, 2*nsim, nvars)
    - All values are finite
    - Data shows non-zero variation

    Parameters
    ----------
    model_class : type[SDE]
        The model class to test (GBM, Vasicek, etc.)
    params : GenericParam
        Initialized parameter instance
    nvars : int
        Expected number of state variables
    use_cython : bool
        Whether to test Cython backend (True) or Python backend (False)

    """
    cfg = TestSimulationConfig()
    nobs = cfg.nobs * cfg.nsub
    expected_nsim = 2 * cfg.nsim  # antithetic sampling doubles nsim

    model = model_class(params)
    paths = model.simulate(
        nsub=cfg.nsub, ndiscr=cfg.ndiscr, nobs=nobs, nsim=cfg.nsim, diff=0, cython=use_cython, seed=cfg.seed
    )

    assert_simulation_shape(paths=paths, nobs=nobs, expected_nsim=expected_nsim, nvars=nvars)
    assert_finite_values(paths=paths)
    assert_variation(paths=paths)


@pytest.mark.parametrize("model_class,params,nvars", get_model_fixtures())
def test_simulate_python_vs_cython_equivalence(*, model_class: type[SDE], params: GenericParam, nvars: int) -> None:
    """Test Python and Cython backends produce statistically equivalent results.

    Uses identical parameters and seeds to verify both implementations
    generate samples with matching statistical properties (mean, std, quantiles).
    Allows reasonable numerical tolerance due to floating-point differences.

    Parameters
    ----------
    model_class : type[SDE]
        The model class to test (GBM, Vasicek, etc.)
    params : GenericParam
        Initialized parameter instance
    nvars : int
        Expected number of state variables

    """
    cfg = TestSimulationConfig()
    nobs = cfg.nobs * cfg.nsub
    expected_nsim = 2 * cfg.nsim  # antithetic sampling doubles nsim

    model_py = model_class(params)
    model_cy = model_class(params)

    paths_py = model_py.simulate(
        nsub=cfg.nsub, ndiscr=cfg.ndiscr, nobs=nobs, nsim=cfg.nsim, diff=0, cython=False, seed=cfg.seed
    )

    paths_cy = model_cy.simulate(
        nsub=cfg.nsub, ndiscr=cfg.ndiscr, nobs=nobs, nsim=cfg.nsim, diff=0, cython=True, seed=cfg.seed
    )

    assert_simulation_shape(paths=paths_py, nobs=nobs, expected_nsim=expected_nsim, nvars=nvars)
    assert_simulation_shape(paths=paths_cy, nobs=nobs, expected_nsim=expected_nsim, nvars=nvars)
    assert_statistical_equivalence(paths_py=paths_py, paths_cy=paths_cy)


@pytest.mark.parametrize("model_class,params,nvars", get_model_fixtures())
@pytest.mark.parametrize("use_cython", [False, True])
def test_sim_realized_all_models(*, model_class: type[SDE], params: GenericParam, nvars: int, use_cython: bool) -> None:
    """Test sim_realized runs successfully for all models with both backends.

    Validates that:
    - sim_realized completes without error
    - Output has correct shape
    - All values are finite
    - Data shows non-zero variation

    Parameters
    ----------
    model_class : type[SDE]
        The model class to test (GBM, Vasicek, etc.)
    params : GenericParam
        Initialized parameter instance
    nvars : int
        Expected number of state variables
    use_cython : bool
        Whether to test Cython backend (True) or Python backend (False)

    """
    del nvars
    nsub = 80
    ndiscr = 1
    aggh = 10
    nperiods = 500

    model = model_class(params)
    returns, rvar = model.sim_realized(
        nsub=nsub, ndiscr=ndiscr, aggh=aggh, nperiods=nperiods, nsim=1, diff=0, cython=use_cython
    )

    expected_length = nperiods - aggh + 1

    assert returns.shape == (expected_length,)
    assert rvar.shape == (expected_length,)
    assert np.all(np.isfinite(returns))
    assert np.all(np.isfinite(rvar))
    assert np.std(returns) > 0
    assert np.std(rvar) > 0


@pytest.mark.parametrize("model_class,params,nvars", get_model_fixtures())
def test_sim_realized_python_vs_cython_equivalence(*, model_class: type[SDE], params: GenericParam, nvars: int) -> None:
    """Test Python and Cython backends produce statistically equivalent realized results.

    Uses identical parameters and seeding to verify both implementations
    generate samples with matching statistical properties (mean, std, quantiles).

    Parameters
    ----------
    model_class : type[SDE]
        The model class to test (GBM, Vasicek, etc.)
    params : GenericParam
        Initialized parameter instance
    nvars : int
        Expected number of state variables

    """
    del nvars
    cfg = TestSimulationConfig()
    nsub = 80
    ndiscr = 1
    aggh = 10
    nperiods = 500

    model_py = model_class(params)
    returns_py, rvar_py = model_py.sim_realized(
        nsub=nsub, ndiscr=ndiscr, aggh=aggh, nperiods=nperiods, nsim=1, diff=0, cython=False, seed=cfg.seed
    )

    model_cy = model_class(params)
    returns_cy, rvar_cy = model_cy.sim_realized(
        nsub=nsub, ndiscr=ndiscr, aggh=aggh, nperiods=nperiods, nsim=1, diff=0, cython=True, seed=cfg.seed
    )

    expected_length = nperiods - aggh + 1

    assert returns_py.shape == (expected_length,)
    assert returns_cy.shape == (expected_length,)
    assert rvar_py.shape == (expected_length,)
    assert rvar_cy.shape == (expected_length,)

    # Check statistical equivalence for returns
    mean_py = np.mean(returns_py)
    mean_cy = np.mean(returns_cy)
    rel_diff_mean = np.abs(mean_py - mean_cy) / (np.abs(mean_py) + 1e-10)
    assert rel_diff_mean < 0.01

    std_py = np.std(returns_py)
    std_cy = np.std(returns_cy)
    rel_diff_std = np.abs(std_py - std_cy) / std_py
    assert rel_diff_std < 0.05

    # Check statistical equivalence for realized variance
    mean_rvar_py = np.mean(rvar_py)
    mean_rvar_cy = np.mean(rvar_cy)
    rel_diff_rvar_mean = np.abs(mean_rvar_py - mean_rvar_cy) / (np.abs(mean_rvar_py) + 1e-10)
    assert rel_diff_rvar_mean < 0.01

    std_rvar_py = np.std(rvar_py)
    std_rvar_cy = np.std(rvar_cy)
    rel_diff_rvar_std = np.abs(std_rvar_py - std_rvar_cy) / std_rvar_py
    assert rel_diff_rvar_std < 0.05
