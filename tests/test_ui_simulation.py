"""Tests for UI simulation wrapper module."""

from __future__ import annotations

import numpy as np
import pytest

from affidiff.ui.simulation import (
    extract_simulation_data,
    get_model_defaults,
    run_simulation,
)


class TestGetModelDefaults:
    """Tests for get_model_defaults function."""

    @pytest.mark.parametrize("model_name", ["GBM", "Vasicek", "CIR", "Heston", "CentTend"])
    def test_returns_dict(self, model_name: str) -> None:
        """Test that function returns a dictionary."""
        result = get_model_defaults(model_name)
        assert isinstance(result, dict)

    @pytest.mark.parametrize("model_name", ["GBM", "Vasicek", "CIR", "Heston", "CentTend"])
    def test_defaults_are_numeric_or_enum(self, model_name: str) -> None:
        """Test that all default values are numeric (excluding measure enum)."""
        defaults = get_model_defaults(model_name)
        for name, value in defaults.items():
            if name == "measure":
                continue
            assert isinstance(value, (int, float)), f"{name} is not numeric"

    def test_invalid_model_raises(self) -> None:
        """Test that invalid model name raises ValueError."""
        with pytest.raises(ValueError, match="Unknown model"):
            get_model_defaults("InvalidModel")

    def test_gbm_defaults(self) -> None:
        """Test GBM specific defaults."""
        defaults = get_model_defaults("GBM")
        assert "mean" in defaults
        assert "sigma" in defaults
        assert defaults["sigma"] > 0


class TestRunSimulation:
    """Tests for run_simulation function."""

    def test_gbm_simulation_succeeds(self) -> None:
        """Test GBM simulation runs without error."""
        params = {"mean": 0.0, "sigma": 0.2}
        result = run_simulation(model_name="GBM", params=params, nobs=100, seed=42)
        assert result.shape == (100, 2, 1)

    def test_simulation_respects_seed(self) -> None:
        """Test that same seed produces identical results."""
        params = {"mean": 0.0, "sigma": 0.2}
        result1 = run_simulation(model_name="GBM", params=params, nobs=100, seed=12345)
        result2 = run_simulation(model_name="GBM", params=params, nobs=100, seed=12345)
        np.testing.assert_array_equal(result1, result2)

    def test_different_seeds_produce_different_results(self) -> None:
        """Test that different seeds produce different results."""
        params = {"mean": 0.0, "sigma": 0.2}
        result1 = run_simulation(model_name="GBM", params=params, nobs=100, seed=111)
        result2 = run_simulation(model_name="GBM", params=params, nobs=100, seed=222)
        assert not np.allclose(result1, result2)

    def test_invalid_model_raises(self) -> None:
        """Test that invalid model name raises ValueError."""
        with pytest.raises(ValueError, match="Unknown model"):
            run_simulation(model_name="InvalidModel", params={}, nobs=100)

    def test_invalid_params_raise(self) -> None:
        """Test that invalid parameters raise ValueError."""
        with pytest.raises(ValueError, match="Invalid parameters"):
            run_simulation(model_name="GBM", params={"invalid_param": 1.0}, nobs=100)

    def test_new_innov_parameter(self) -> None:
        """Test that new_innov parameter is passed correctly."""
        params = {"mean": 0.0, "sigma": 0.2}
        result1 = run_simulation(model_name="GBM", params=params, nobs=100, seed=42, new_innov=True)
        assert result1.shape == (100, 2, 1)

    def test_new_innov_false_reuses_innovations(self) -> None:
        """Test that new_innov=False reuses stored innovations."""
        params = {"mean": 0.0, "sigma": 0.2}
        result1 = run_simulation(model_name="GBM", params=params, nobs=100, seed=42, new_innov=True)
        result2 = run_simulation(model_name="GBM", params=params, nobs=100, seed=42, new_innov=False)
        np.testing.assert_array_equal(result1, result2)

    def test_new_innov_true_different_params_produces_different_result(self) -> None:
        """Test that new_innov=True with different params produces different result."""
        params1 = {"mean": 0.0, "sigma": 0.2}
        params2 = {"mean": 0.05, "sigma": 0.2}
        result1 = run_simulation(model_name="GBM", params=params1, nobs=100, seed=42, new_innov=True)
        result2 = run_simulation(model_name="GBM", params=params2, nobs=100, seed=42, new_innov=True)
        assert not np.allclose(result1, result2)

    @pytest.mark.parametrize("model_name", ["GBM", "Vasicek", "CIR", "Heston", "CentTend"])
    def test_all_models_simulate(self, model_name: str) -> None:
        """Test that all models can simulate successfully."""
        defaults = get_model_defaults(model_name)
        result = run_simulation(model_name=model_name, params=defaults, nobs=50, seed=42)
        assert result.shape[0] == 50
        assert result.shape[1] == 2
        assert np.all(np.isfinite(result)), f"{model_name} produced non-finite values"


class TestExtractSimulationData:
    """Tests for extract_simulation_data function."""

    def test_extracts_first_path(self) -> None:
        """Test that function extracts the first antithetic path."""
        simulated = np.array([[[1.0]], [[2.0]]])
        result = extract_simulation_data(simulated)
        assert result.shape == (2,)
        np.testing.assert_array_equal(result, np.array([1.0, 2.0]))

    def test_handles_multivariate(self) -> None:
        """Test that function handles multivariate output."""
        simulated = np.random.randn(100, 2, 3)
        result = extract_simulation_data(simulated)
        assert result.shape == (100, 3)
        np.testing.assert_array_equal(result, simulated[:, 0, :])

    def test_squeeze_univariate(self) -> None:
        """Test that univariate paths are squeezed to 1D."""
        simulated = np.random.randn(100, 2, 1)
        result = extract_simulation_data(simulated)
        assert result.ndim == 1
        assert result.shape == (100,)
