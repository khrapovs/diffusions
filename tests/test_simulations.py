"""Test suite for diffusion model simulations across all models and backends."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from conftest import (
    TestSimulationConfig,
    assert_finite_values,
    assert_realized_equivalence,
    assert_realized_finite_variation,
    assert_realized_shape,
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

    assert_realized_shape(returns=returns, rvar=rvar, expected_length=expected_length)
    assert_realized_finite_variation(returns=returns, rvar=rvar)


@pytest.mark.parametrize("model_class,params,nvars", get_model_fixtures())
def test_sim_realized_python_vs_cython_equivalence(*, model_class: type[SDE], params: GenericParam, nvars: int) -> None:
    """Test Python and Cython backends produce statistically equivalent realized results.

    Uses identical parameters and seeding to verify both implementations
    generate samples with matching statistical properties (mean, std).

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

    assert_realized_shape(returns=returns_py, rvar=rvar_py, expected_length=expected_length)
    assert_realized_shape(returns=returns_cy, rvar=rvar_cy, expected_length=expected_length)
    assert_realized_equivalence(returns_py=returns_py, returns_cy=returns_cy, rvar_py=rvar_py, rvar_cy=rvar_cy)
