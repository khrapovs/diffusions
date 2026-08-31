"""Test suite for GBM model simulation."""

import numpy as np

from affidiff import GBM, GBMparam


class TestGBMSimulation:
    """Test GBM model simulation."""

    def test_simulate_gbm(self) -> None:
        """Test simulating GBM model with cython=False.

        Based on try_simulation function from examples/try_gbm.py.
        Uses cython=False to test Python-based simulation.
        """
        mean, sigma = 0.05, 0.2
        theta_true = GBMparam(mean=mean, sigma=sigma)
        gbm = GBM(theta_true)

        start, nperiods, nsub, ndiscr, nsim = 1, 500, 2, 10, 2
        nobs = nperiods * nsub
        paths = gbm.simulate(start=start, nsub=nsub, ndiscr=ndiscr, nobs=nobs, nsim=nsim, diff=0, cython=False)

        # After simulation and nice_errors antithetic sampling, nsim is doubled
        # Output shape: (nobs, 2*nsim, nvars) where nvars = 1 for GBM
        expected_nsim = 2 * nsim
        assert paths.shape == (nobs, expected_nsim, 1)

        # Extract the single series for inspection
        data = paths[:, 0, 0]

        # Check that data is a 1D array of length nobs
        assert data.shape == (nobs,)

        # Check that we have finite values
        assert np.all(np.isfinite(data))

        # With diff=0, simulations compute differences in prices,
        # so we can have both positive and negative values
        # Just verify we get some variation in the simulated data
        assert np.std(data) > 0

    def test_simulate_gbm_cython(self) -> None:
        """Test simulating GBM model with cython=True.

        Verifies that the Cython-accelerated simulation works correctly.
        """
        mean, sigma = 0.05, 0.2
        theta_true = GBMparam(mean=mean, sigma=sigma)
        gbm = GBM(theta_true)

        start, nperiods, nsub, ndiscr, nsim = 1, 500, 2, 10, 2
        nobs = nperiods * nsub
        paths = gbm.simulate(start=start, nsub=nsub, ndiscr=ndiscr, nobs=nobs, nsim=nsim, diff=0, cython=True)

        # After simulation and nice_errors antithetic sampling, nsim is doubled
        # Output shape: (nobs, 2*nsim, nvars) where nvars = 1 for GBM
        expected_nsim = 2 * nsim
        assert paths.shape == (nobs, expected_nsim, 1)

        # Extract the single series for inspection
        data = paths[:, 0, 0]

        # Check that data is a 1D array of length nobs
        assert data.shape == (nobs,)

        # Check that we have finite values
        assert np.all(np.isfinite(data))

        # Verify we get some variation in the simulated data
        assert np.std(data) > 0

    def test_simulate_gbm_python_vs_cython_statistical_equivalence(self) -> None:
        """Test that Python and Cython backends produce statistically equivalent results.

        Uses identical parameters and seeds to verify both implementations
        generate samples with matching statistical properties (mean, std, quantiles).
        Allows reasonable numerical tolerance due to floating-point differences.
        """
        mean, sigma = 0.05, 0.2
        theta_true = GBMparam(mean=mean, sigma=sigma)
        gbm_py = GBM(theta_true)
        gbm_cy = GBM(theta_true)

        # Use identical simulation parameters
        start, nperiods, nsub, ndiscr, nsim = 1, 500, 2, 10, 10
        nobs = nperiods * nsub

        # Simulate with Python backend
        np.random.seed(42)
        paths_py = gbm_py.simulate(start=start, nsub=nsub, ndiscr=ndiscr, nobs=nobs, nsim=nsim, diff=0, cython=False)

        # Simulate with Cython backend (same seed)
        np.random.seed(42)
        paths_cy = gbm_cy.simulate(start=start, nsub=nsub, ndiscr=ndiscr, nobs=nobs, nsim=nsim, diff=0, cython=True)

        # Both should have same shape
        assert paths_py.shape == paths_cy.shape
        expected_nsim = 2 * nsim  # antithetic sampling doubles nsim
        assert paths_py.shape == (nobs, expected_nsim, 1)

        # Flatten to get all samples as single 1D array for statistical comparison
        data_py = paths_py[:, :, 0].flatten()
        data_cy = paths_cy[:, :, 0].flatten()

        # Compare statistical properties with reasonable tolerance
        # Mean should be close (relative tolerance 1%)
        mean_py = np.mean(data_py)
        mean_cy = np.mean(data_cy)
        assert np.abs(mean_py - mean_cy) / (np.abs(mean_py) + 1e-10) < 0.01

        # Standard deviation should be close (relative tolerance 5%)
        std_py = np.std(data_py)
        std_cy = np.std(data_cy)
        assert np.abs(std_py - std_cy) / std_py < 0.05

        # Quantiles should be close (relative tolerance 5%)
        quantiles = [0.05, 0.25, 0.5, 0.75, 0.95]
        for q in quantiles:
            q_py = np.quantile(data_py, q)
            q_cy = np.quantile(data_cy, q)
            # Use absolute tolerance for quantiles near zero
            abs_tol = max(1e-6, 0.05 * np.abs(q_py))
            assert np.abs(q_py - q_cy) < abs_tol
