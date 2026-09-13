"""Test suite for diffusion model simulations across all models and backends."""

from __future__ import annotations

from typing import TYPE_CHECKING

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


@pytest.mark.parametrize(
    "model_class,params,nvars", get_model_fixtures(), ids=["GBM", "Vasicek", "Heston", "CIR", "CentTend"]
)
@pytest.mark.parametrize("use_cython", [False, True], ids=["python", "cython"])
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


@pytest.mark.parametrize(
    "model_class,params,nvars", get_model_fixtures(), ids=["GBM", "Vasicek", "Heston", "CIR", "CentTend"]
)
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
