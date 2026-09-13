"""Pytest configuration and shared fixtures for simulation tests."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from affidiff import CIR, GBM, CentTend, CentTendParam, CIRparam, GBMparam, Heston, HestonParam, Vasicek, VasicekParam

if TYPE_CHECKING:
    from affidiff.model_generic import SDE
    from affidiff.param_generic import GenericParam

# Test configuration shared across all model simulations
TEST_CONFIG = {"nobs": 500, "nsub": 2, "ndiscr": 10, "nsim": 2, "seed": 42}


def get_model_fixtures() -> list[tuple[type[SDE], GenericParam, int]]:
    """Return list of (ModelClass, param_instance, nvars) tuples.

    Returns
    -------
    list[tuple[type[SDE], GenericParam, int]]
        List of model fixtures where each tuple contains:
        - Model class (e.g., GBM, Vasicek)
        - Parameter instance (initialized with test values)
        - Number of state variables (nvars)

    """
    return [
        (GBM, GBMparam(mean=0.05, sigma=0.2), 1),
        (Vasicek, VasicekParam(mean=0.5, kappa=0.1, eta=0.2), 1),
        (Heston, HestonParam(riskfree=0.0, lmbd=0.0, mean_v=0.5, kappa=0.1, eta=0.02**0.5, rho=-0.9), 2),
        (CIR, CIRparam(mean=0.5, kappa=0.1, eta=0.2), 1),
        (
            CentTend,
            CentTendParam(
                riskfree=0.01,
                lmbd=0.01,
                mean_v=0.5,
                kappa_s=1.5,
                kappa_y=0.05,
                eta_s=0.02**0.5,
                eta_y=0.001**0.5,
                rho=-0.9,
            ),
            3,
        ),
    ]


def assert_simulation_shape(*, paths: np.ndarray, nobs: int, expected_nsim: int, nvars: int) -> None:
    """Assert that simulated paths have correct shape.

    Parameters
    ----------
    paths : np.ndarray
        Simulated paths array
    nobs : int
        Expected number of observations
    expected_nsim : int
        Expected number of simulations (after antithetic sampling)
    nvars : int
        Expected number of state variables

    Raises
    ------
    AssertionError
        If shape does not match (nobs, expected_nsim, nvars)

    """
    assert paths.shape == (nobs, expected_nsim, nvars)


def assert_finite_values(*, paths: np.ndarray) -> None:
    """Assert that all values in paths are finite.

    Parameters
    ----------
    paths : np.ndarray
        Simulated paths array

    Raises
    ------
    AssertionError
        If any NaN or infinite values are found

    """
    assert np.all(np.isfinite(paths))


def assert_variation(*, paths: np.ndarray) -> None:
    """Assert that paths show variation (non-zero standard deviation).

    Parameters
    ----------
    paths : np.ndarray
        Simulated paths array

    Raises
    ------
    AssertionError
        If standard deviation is zero

    """
    assert np.std(paths) > 0


def assert_statistical_equivalence(
    *,
    paths_py: np.ndarray,
    paths_cy: np.ndarray,
    rtol_mean: float = 0.01,
    rtol_std: float = 0.05,
    rtol_quantile: float = 0.05,
) -> None:
    """Assert that Python and Cython backends produce statistically equivalent results.

    Compares mean, standard deviation, and quantiles with specified tolerances.

    Parameters
    ----------
    paths_py : np.ndarray
        Simulated paths from Python backend
    paths_cy : np.ndarray
        Simulated paths from Cython backend
    rtol_mean : float, optional
        Relative tolerance for mean comparison (default 0.01 = 1%)
    rtol_std : float, optional
        Relative tolerance for standard deviation comparison (default 0.05 = 5%)
    rtol_quantile : float, optional
        Relative tolerance for quantile comparison (default 0.05 = 5%)

    Raises
    ------
    AssertionError
        If statistical properties differ beyond tolerance

    """
    data_py = paths_py[:, :, :].flatten()
    data_cy = paths_cy[:, :, :].flatten()

    # Compare means
    mean_py = np.mean(data_py)
    mean_cy = np.mean(data_cy)
    rel_diff_mean = np.abs(mean_py - mean_cy) / (np.abs(mean_py) + 1e-10)
    assert rel_diff_mean < rtol_mean

    # Compare standard deviations
    std_py = np.std(data_py)
    std_cy = np.std(data_cy)
    rel_diff_std = np.abs(std_py - std_cy) / std_py
    assert rel_diff_std < rtol_std

    # Compare quantiles
    quantiles = [0.05, 0.25, 0.5, 0.75, 0.95]
    for q in quantiles:
        q_py = np.quantile(data_py, q)
        q_cy = np.quantile(data_cy, q)
        abs_tol = max(1e-6, rtol_quantile * np.abs(q_py))
        assert np.abs(q_py - q_cy) < abs_tol
