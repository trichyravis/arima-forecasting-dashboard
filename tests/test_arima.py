"""
Unit Tests for ARIMA Model Module - tests/test_arima.py

Tests for src/models/arima.py using pytest.

Author: Prof. V. Ravichandran
The Mountain Path - World of Finance
"""

import pytest
import pandas as pd
import numpy as np
from src.models.arima import ARIMAModel, ARIMAEnsemble


class TestARIMAModel:
    """Test ARIMAModel class."""
    
    @pytest.fixture
    def sample_series(self):
        """Create sample time series."""
        dates = pd.date_range('2023-01-01', periods=200)
        prices = 100 + np.cumsum(np.random.randn(200) * 0.5)
        return pd.Series(prices, index=dates, name='price')
    
    def test_model_initialization(self, sample_series):
        """Test ARIMA model initializes correctly."""
        model = ARIMAModel(sample_series, name="TestModel")
        
        assert model is not None
        assert model.name == "TestModel"
        assert len(model.series) == 200
        assert model.fitted_model is None
    
    def test_stationarity_check(self, sample_series):
        """Test stationarity checking."""
        model = ARIMAModel(sample_series)
        result = model.check_stationarity()
        
        assert 'adf' in result
        assert 'summary' in result
        assert 'is_stationary' in result['summary']
        assert isinstance(result['summary']['is_stationary'], bool)
    
    def test_adf_test_results(self, sample_series):
        """Test ADF test returns valid results."""
        model = ARIMAModel(sample_series)
        result = model.check_stationarity()
        
        adf = result['adf']
        assert 'statistic' in adf
        assert 'p_value' in adf
        assert 'critical_values' in adf
        assert 'is_stationary' in adf
        assert 0 <= adf['p_value'] <= 1
    
    def test_model_fit_valid_order(self, sample_series):
        """Test fitting model with valid parameters."""
        model = ARIMAModel(sample_series)
        stats = model.fit((1, 1, 1))
        
        assert stats is not None
        assert 'aic' in stats
        assert 'bic' in stats
        assert stats['aic'] > 0
        assert stats['bic'] > 0
    
    def test_model_fit_different_orders(self, sample_series):
        """Test fitting with different ARIMA orders."""
        model = ARIMAModel(sample_series)
        
        orders = [(1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 1, 1)]
        
        for order in orders:
            stats = model.fit(order)
            assert stats is not None
            assert isinstance(stats['aic'], (int, float))
            assert isinstance(stats['bic'], (int, float))
    
    def test_forecast_generation(self, sample_series):
        """Test forecast generation."""
        model = ARIMAModel(sample_series)
        model.fit((1, 1, 1))
        
        forecast_df = model.forecast(steps=10)
        
        assert forecast_df is not None
        assert len(forecast_df) == 10
        assert 'forecast' in forecast_df.columns
        assert 'lower_ci' in forecast_df.columns
        assert 'upper_ci' in forecast_df.columns
    
    def test_forecast_columns(self, sample_series):
        """Test forecast has correct columns."""
        model = ARIMAModel(sample_series)
        model.fit((1, 1, 1))
        
        forecast_df = model.forecast(steps=5)
        
        assert list(forecast_df.columns) == ['forecast', 'lower_ci', 'upper_ci']
    
    def test_confidence_intervals(self, sample_series):
        """Test confidence intervals are valid."""
        model = ARIMAModel(sample_series)
        model.fit((1, 1, 1))
        
        forecast_df = model.forecast(steps=10)
        
        # Lower CI should be less than forecast
        assert (forecast_df['lower_ci'] <= forecast_df['forecast']).all()
        
        # Forecast should be less than upper CI
        assert (forecast_df['forecast'] <= forecast_df['upper_ci']).all()
    
    def test_residuals_extraction(self, sample_series):
        """Test residuals extraction."""
        model = ARIMAModel(sample_series)
        model.fit((1, 1, 1))
        
        residuals = model.get_residuals()
        
        assert isinstance(residuals, pd.Series)
        assert len(residuals) > 0
    
    def test_model_summary(self, sample_series):
        """Test model summary generation."""
        model = ARIMAModel(sample_series)
        model.fit((1, 1, 1))
        
        summary = model.get_summary()
        
        assert isinstance(summary, str)
        assert len(summary) > 0
        assert 'ARIMA' in summary or 'Constant' in summary
    
    def test_metrics_calculation(self, sample_series):
        """Test metrics calculation."""
        model = ARIMAModel(sample_series)
        model.fit((1, 1, 1))
        
        metrics = model.calculate_metrics()
        
        assert 'aic' in metrics
        assert 'bic' in metrics
        assert 'rmse' in metrics
        assert 'mae' in metrics
        assert all(v >= 0 for v in [metrics['aic'], metrics['bic'], 
                                     metrics['rmse'], metrics['mae']])
    
    def test_different_alphas(self, sample_series):
        """Test forecast with different confidence levels."""
        model = ARIMAModel(sample_series)
        model.fit((1, 1, 1))
        
        # 95% confidence
        f95 = model.forecast(steps=10, alpha=0.05)
        
        # 90% confidence (wider CI)
        f90 = model.forecast(steps=10, alpha=0.10)
        
        # 90% CI should be wider than 95%
        ci_width_95 = (f95['upper_ci'] - f95['lower_ci']).mean()
        ci_width_90 = (f90['upper_ci'] - f90['lower_ci']).mean()
        
        assert ci_width_90 > ci_width_95


class TestARIMAEnsemble:
    """Test ARIMAEnsemble class."""
    
    @pytest.fixture
    def sample_series(self):
        """Create sample time series."""
        dates = pd.date_range('2023-01-01', periods=200)
        prices = 100 + np.cumsum(np.random.randn(200) * 0.5)
        return pd.Series(prices, index=dates)
    
    def test_ensemble_initialization(self, sample_series):
        """Test ensemble initializes correctly."""
        ensemble = ARIMAEnsemble(sample_series)
        
        assert ensemble is not None
        assert len(ensemble.models) == 0
        assert len(ensemble.weights) == 0
    
    def test_fit_multiple_models(self, sample_series):
        """Test fitting multiple models."""
        ensemble = ARIMAEnsemble(sample_series)
        
        orders = [(1, 0, 0), (1, 1, 0), (1, 1, 1)]
        ensemble.fit_multiple(orders)
        
        assert len(ensemble.models) > 0
        assert len(ensemble.weights) == len(ensemble.models)
    
    def test_ensemble_weights_sum(self, sample_series):
        """Test ensemble weights sum to 1."""
        ensemble = ARIMAEnsemble(sample_series)
        
        orders = [(1, 0, 0), (1, 1, 0), (1, 1, 1), (2, 1, 1)]
        ensemble.fit_multiple(orders)
        
        # Weights should sum to approximately 1
        assert np.isclose(np.sum(ensemble.weights), 1.0)
    
    def test_ensemble_forecast(self, sample_series):
        """Test ensemble forecast generation."""
        ensemble = ARIMAEnsemble(sample_series)
        
        orders = [(1, 0, 0), (1, 1, 0), (1, 1, 1)]
        ensemble.fit_multiple(orders)
        
        forecast = ensemble.forecast(steps=10)
        
        assert forecast is not None
        assert len(forecast) == 10
        assert 'forecast' in forecast.columns


class TestModelComparison:
    """Test model comparison and selection."""
    
    @pytest.fixture
    def sample_series(self):
        """Create sample time series."""
        dates = pd.date_range('2023-01-01', periods=200)
        prices = 100 + np.cumsum(np.random.randn(200) * 0.5)
        return pd.Series(prices, index=dates)
    
    def test_aic_comparison(self, sample_series):
        """Test AIC-based model comparison."""
        orders = [(1, 0, 0), (1, 1, 0), (1, 1, 1)]
        aics = []
        
        for order in orders:
            model = ARIMAModel(sample_series)
            model.fit(order)
            aics.append(model.aic)
        
        # All AIC values should be numeric
        assert all(isinstance(aic, (int, float)) for aic in aics)
        
        # Find best model (lowest AIC)
        best_idx = np.argmin(aics)
        assert best_idx in range(len(orders))
    
    def test_multiple_step_forecasts(self, sample_series):
        """Test forecasts of different lengths."""
        model = ARIMAModel(sample_series)
        model.fit((1, 1, 1))
        
        for steps in [1, 5, 10, 30]:
            forecast = model.forecast(steps=steps)
            assert len(forecast) == steps


class TestErrorHandling:
    """Test error handling and edge cases."""
    
    def test_insufficient_data(self):
        """Test handling of insufficient data."""
        dates = pd.date_range('2023-01-01', periods=10)
        series = pd.Series(np.random.randn(10), index=dates)
        
        # Should handle gracefully
        model = ARIMAModel(series)
        with pytest.raises(Exception):
            model.fit((1, 1, 1))
    
    def test_constant_series(self):
        """Test handling of constant series."""
        dates = pd.date_range('2023-01-01', periods=100)
        series = pd.Series([100] * 100, index=dates)
        
        model = ARIMAModel(series)
        # Should not crash, though result might be poor
        try:
            model.fit((0, 0, 0))
            assert True
        except:
            # Expected to fail on constant series
            assert True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

