"""Tests for UI simulation wrapper module."""

from __future__ import annotations

import numpy as np
import pytest

from affidiff.ui.simulation import (
    apply_display_transforms,
    extract_simulation_data,
    get_model_defaults,
    get_series_names,
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


def test_get_series_names_invalid_model_raises() -> None:
    with pytest.raises(ValueError, match="Unknown model"):
        get_series_names("InvalidModel")


class TestApplyDisplayTransforms:
    """Tests for apply_display_transforms function."""

    def test_log_price_exponentiated(self) -> None:
        """log(Price) column is replaced by its exponential (Price)."""
        log_prices = np.array([0.0, 0.5, 1.0])
        path, names = apply_display_transforms(path=log_prices, raw_series_names=("log(Price)",))
        np.testing.assert_allclose(path[:, 0], np.exp(log_prices))
        assert names[0] == "Price"

    def test_log_return_appended(self) -> None:
        """diff(log(Price)) is appended as Log Return with 0 prepended."""
        log_prices = np.array([0.0, 0.5, 1.5])
        path, names = apply_display_transforms(path=log_prices, raw_series_names=("log(Price)",))
        expected_returns = np.array([0.0, 0.5, 1.0])
        np.testing.assert_allclose(path[:, 1], expected_returns)
        assert names[1] == "Log Return"

    def test_output_shape_univariate(self) -> None:
        """Univariate log(Price) produces two display columns."""
        log_prices = np.arange(10, dtype=float)
        path, names = apply_display_transforms(path=log_prices, raw_series_names=("log(Price)",))
        assert path.shape == (10, 2)
        assert names == ("Price", "Log Return")

    def test_non_log_price_passthrough(self) -> None:
        """Series not named log(Price) are passed through unchanged."""
        raw = np.ones((5, 2))
        raw[:, 1] = 2.0
        path, names = apply_display_transforms(path=raw, raw_series_names=("Volatility", "CentralTend"))
        np.testing.assert_array_equal(path[:, 0], raw[:, 0])
        np.testing.assert_array_equal(path[:, 1], raw[:, 1])
        assert names == ("Volatility", "CentralTend")

    @pytest.mark.parametrize("model_name", ["GBM", "Heston", "CentTend"])
    def test_integration_with_real_simulation(self, model_name: str) -> None:
        """Display names and shape are consistent for all log(Price) models."""
        defaults = {k: v for k, v in get_model_defaults(model_name).items() if k != "measure"}
        simulated = run_simulation(model_name=model_name, params=defaults, nobs=50, seed=7)
        raw_path = extract_simulation_data(simulated)
        raw_names = get_series_names(model_name)
        display_path, display_names = apply_display_transforms(path=raw_path, raw_series_names=raw_names)
        n_log_price = sum(1 for n in raw_names if n == "log(Price)")
        expected_cols = len(raw_names) + n_log_price  # one extra Log Return per log(Price)
        assert display_path.shape == (50, expected_cols)
        assert "log(Price)" not in display_names
        assert "Price" in display_names
        assert "Log Return" in display_names
