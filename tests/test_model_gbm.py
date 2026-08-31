"""Test suite for GBM model simulation."""

import numpy as np
import numpy.testing as npt

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
        paths = gbm.simulate(
            start=start,
            nsub=nsub,
            ndiscr=ndiscr,
            nobs=nobs,
            nsim=nsim,
            diff=0,
            cython=False,
        )

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
