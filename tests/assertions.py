"""Assertion helpers for simulation tests."""

from __future__ import annotations

import numpy as np


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


def assert_realized_shape(*, returns: np.ndarray, rvar: np.ndarray, expected_length: int) -> None:
    """Assert that realized returns and variance have correct shapes.

    Parameters
    ----------
    returns : np.ndarray
        Simulated realized returns array
    rvar : np.ndarray
        Simulated realized variance array
    expected_length : int
        Expected length (nperiods - aggh + 1)

    Raises
    ------
    AssertionError
        If shapes do not match (expected_length,)

    """
    assert returns.shape == (expected_length,)
    assert rvar.shape == (expected_length,)


def assert_realized_finite_variation(*, returns: np.ndarray, rvar: np.ndarray) -> None:
    """Assert that realized returns and variance have finite values and non-zero variation.

    Parameters
    ----------
    returns : np.ndarray
        Simulated realized returns array
    rvar : np.ndarray
        Simulated realized variance array

    Raises
    ------
    AssertionError
        If any NaN/infinite values found or variation is zero

    """
    assert np.all(np.isfinite(returns))
    assert np.all(np.isfinite(rvar))
    assert np.std(returns) > 0
    assert np.std(rvar) > 0


def assert_realized_equivalence(
    *,
    returns_py: np.ndarray,
    returns_cy: np.ndarray,
    rvar_py: np.ndarray,
    rvar_cy: np.ndarray,
    rtol_mean: float = 0.01,
    rtol_std: float = 0.05,
) -> None:
    """Assert that Python and Cython backends produce statistically equivalent realized data.

    Compares mean and standard deviation for both returns and realized variance
    with specified tolerances.

    Parameters
    ----------
    returns_py : np.ndarray
        Realized returns from Python backend
    returns_cy : np.ndarray
        Realized returns from Cython backend
    rvar_py : np.ndarray
        Realized variance from Python backend
    rvar_cy : np.ndarray
        Realized variance from Cython backend
    rtol_mean : float, optional
        Relative tolerance for mean comparison (default 0.01 = 1%)
    rtol_std : float, optional
        Relative tolerance for standard deviation comparison (default 0.05 = 5%)

    Raises
    ------
    AssertionError
        If statistical properties differ beyond tolerance

    """
    # Check statistical equivalence for returns
    mean_py = np.mean(returns_py)
    mean_cy = np.mean(returns_cy)
    rel_diff_mean = np.abs(mean_py - mean_cy) / (np.abs(mean_py) + 1e-10)
    assert rel_diff_mean < rtol_mean

    std_py = np.std(returns_py)
    std_cy = np.std(returns_cy)
    rel_diff_std = np.abs(std_py - std_cy) / std_py
    assert rel_diff_std < rtol_std

    # Check statistical equivalence for realized variance
    mean_rvar_py = np.mean(rvar_py)
    mean_rvar_cy = np.mean(rvar_cy)
    rel_diff_rvar_mean = np.abs(mean_rvar_py - mean_rvar_cy) / (np.abs(mean_rvar_py) + 1e-10)
    assert rel_diff_rvar_mean < rtol_mean

    std_rvar_py = np.std(rvar_py)
    std_rvar_cy = np.std(rvar_cy)
    rel_diff_rvar_std = np.abs(std_rvar_py - std_rvar_cy) / std_rvar_py
    assert rel_diff_rvar_std < rtol_std
