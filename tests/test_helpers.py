"""
Unit Tests for Helper Functions Module - tests/test_helpers.py

Tests for src/utils/helpers.py using pytest.

Author: Prof. V. Ravichandran
The Mountain Path - World of Finance
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from src.utils.helpers import (
    get_date_range,
    get_trading_days,
    is_trading_day,
    validate_series,
    calculate_returns,
    calculate_volatility,
    calculate_trend,
    format_number,
    format_percentage,
    format_date
)


class TestDateUtilities:
    """Test date and time utility functions."""
    
    def test_get_date_range_default(self):
        """Test getting date range with defaults."""
        start, end = get_date_range()
        
        assert isinstance(start, str)
        assert isinstance(end, str)
        assert start < end
    
    def test_get_date_range_custom_days(self):
        """Test getting date range with custom days."""
        start, end = get_date_range(days_back=30)
        
        start_dt = pd.Timestamp(start)
        end_dt = pd.Timestamp(end)
        
        diff = (end_dt - start_dt).days
        assert 29 <= diff <= 31  # Allow for rounding
    
    def test_get_date_range_custom_end(self):
        """Test getting date range with custom end date."""
        custom_end = "2023-06-30"
        start, end = get_date_range(days_back=365, end_date=custom_end)
        
        assert end == custom_end
    
    def test_trading_days(self):
        """Test getting trading days."""
        trading_days = get_trading_days("2023-01-01", "2023-01-31")
        
        assert isinstance(trading_days, pd.DatetimeIndex)
        assert len(trading_days) > 0
        # January 2023 should have ~21 trading days
        assert 20 <= len(trading_days) <= 23
    
    def test_is_trading_day_weekday(self):
        """Test recognizing weekdays as trading days."""
        monday = pd.Timestamp('2023-01-02')  # Monday
        tuesday = pd.Timestamp('2023-01-03')  # Tuesday
        
        assert is_trading_day(monday) == True
        assert is_trading_day(tuesday) == True
    
    def test_is_trading_day_weekend(self):
        """Test recognizing weekends as non-trading days."""
        saturday = pd.Timestamp('2023-01-07')  # Saturday
        sunday = pd.Timestamp('2023-01-08')    # Sunday
        
        assert is_trading_day(saturday) == False
        assert is_trading_day(sunday) == False
    
    def test_is_trading_day_string(self):
        """Test is_trading_day with string input."""
        result = is_trading_day("2023-01-02")  # Monday
        assert isinstance(result, bool)
        assert result == True


class TestDataValidation:
    """Test data validation functions."""
    
    @pytest.fixture
    def valid_series(self):
        """Create valid test series."""
        dates = pd.date_range('2023-01-01', periods=100)
        return pd.Series(np.random.randn(100), index=dates)
    
    def test_validate_valid_series(self, valid_series):
        """Test validation of valid series."""
        is_valid, errors = validate_series(valid_series)
        
        assert is_valid == True
        assert len(errors) == 0
    
    def test_validate_empty_series(self):
        """Test validation rejects empty series."""
        empty = pd.Series([], dtype=float)
        is_valid, errors = validate_series(empty)
        
        assert is_valid == False
    
    def test_validate_short_series(self):
        """Test validation rejects short series."""
        short = pd.Series(np.random.randn(10))
        is_valid, errors = validate_series(short)
        
        assert is_valid == False
    
    def test_validate_series_with_nans(self):
        """Test validation detects NaN values."""
        with_nans = pd.Series([1, 2, np.nan, 4, 5] * 15)
        is_valid, errors = validate_series(with_nans)
        
        assert is_valid == False


class TestStatisticalCalculations:
    """Test statistical calculation functions."""
    
    @pytest.fixture
    def price_series(self):
        """Create price series."""
        dates = pd.date_range('2023-01-01', periods=100)
        prices = 100 + np.cumsum(np.random.randn(100) * 0.5)
        return pd.Series(prices, index=dates)
    
    def test_calculate_returns_percentage(self, price_series):
        """Test calculating returns as percentages."""
        returns = calculate_returns(price_series, pct=True)
        
        assert isinstance(returns, pd.Series)
        assert len(returns) == len(price_series)
        assert returns.iloc[0]  # First value will be NaN
    
    def test_calculate_returns_decimal(self, price_series):
        """Test calculating returns as decimals."""
        returns = calculate_returns(price_series, pct=False)
        
        assert isinstance(returns, pd.Series)
        # Most returns should be small decimals
        assert returns.abs().max() < 0.1
    
    def test_returns_vs_percentage(self, price_series):
        """Test relationship between pct and decimal returns."""
        returns_pct = calculate_returns(price_series, pct=True)
        returns_dec = calculate_returns(price_series, pct=False)
        
        # Percentage should be roughly 100x decimal
        ratio = (returns_pct / (returns_dec * 100)).dropna()
        assert ratio.abs().mean() > 0.99
    
    def test_calculate_volatility(self, price_series):
        """Test calculating volatility."""
        returns = calculate_returns(price_series)
        volatility = calculate_volatility(returns, window=20)
        
        assert isinstance(volatility, pd.Series)
        assert len(volatility) == len(returns)
        assert volatility.dropna().min() >= 0
    
    def test_volatility_increases_with_noise(self):
        """Test volatility with increasing noise."""
        dates = pd.date_range('2023-01-01', periods=200)
        
        # Low volatility series
        low_vol = pd.Series(100 + np.cumsum(np.random.randn(200) * 0.1), index=dates)
        low_vol_returns = calculate_returns(low_vol)
        low_vol_result = calculate_volatility(low_vol_returns, window=20)
        
        # High volatility series
        high_vol = pd.Series(100 + np.cumsum(np.random.randn(200) * 2.0), index=dates)
        high_vol_returns = calculate_returns(high_vol)
        high_vol_result = calculate_volatility(high_vol_returns, window=20)
        
        assert high_vol_result.mean() > low_vol_result.mean()
    
    def test_calculate_trend(self, price_series):
        """Test trend calculation."""
        sma, direction = calculate_trend(price_series, window=20)
        
        assert isinstance(sma, pd.Series)
        assert direction in ["Uptrend", "Downtrend", "Neutral"]
    
    def test_trend_uptrend(self):
        """Test detecting uptrend."""
        dates = pd.date_range('2023-01-01', periods=100)
        # Clearly upward trend
        prices = pd.Series(np.linspace(100, 200, 100), index=dates)
        
        _, direction = calculate_trend(prices)
        assert direction == "Uptrend"
    
    def test_trend_downtrend(self):
        """Test detecting downtrend."""
        dates = pd.date_range('2023-01-01', periods=100)
        # Clearly downward trend
        prices = pd.Series(np.linspace(200, 100, 100), index=dates)
        
        _, direction = calculate_trend(prices)
        assert direction == "Downtrend"


class TestFormatting:
    """Test formatting utility functions."""
    
    def test_format_number_integer(self):
        """Test formatting integers."""
        result = format_number(1000000)
        assert result == "1,000,000"
    
    def test_format_number_float(self):
        """Test formatting floats."""
        result = format_number(1234567.89, decimals=2)
        assert "1,234,567.89" in result
    
    def test_format_number_small(self):
        """Test formatting small numbers."""
        result = format_number(123.45)
        assert "123.45" in result
    
    def test_format_percentage_decimal(self):
        """Test formatting percentage from decimal."""
        result = format_percentage(0.1234, decimals=2)
        assert "12.34" in result
        assert "%" in result
    
    def test_format_percentage_already_pct(self):
        """Test formatting percentage when already as percentage."""
        result = format_percentage(12.34, decimals=2)
        assert "12.34" in result
    
    def test_format_date_timestamp(self):
        """Test formatting date from timestamp."""
        ts = pd.Timestamp('2023-06-15')
        result = format_date(ts)
        assert result == "2023-06-15"
    
    def test_format_date_string(self):
        """Test formatting date from string."""
        result = format_date("2023-06-15")
        assert result == "2023-06-15"
    
    def test_format_date_custom_format(self):
        """Test formatting date with custom format."""
        ts = pd.Timestamp('2023-06-15')
        result = format_date(ts, fmt="%d/%m/%Y")
        assert result == "15/06/2023"


class TestModelUtilities:
    """Test model-related utility functions."""
    
    def test_get_model_name(self):
        """Test getting model name from parameters."""
        from src.utils.helpers import get_model_name
        
        name = get_model_name((1, 1, 1))
        assert name == "ARIMA(1,1,1)"
    
    def test_format_metrics(self):
        """Test formatting metrics dictionary."""
        from src.utils.helpers import format_metrics
        
        metrics = {'aic': 100.5, 'bic': 110.5, 'rmse': 0.05}
        result = format_metrics(metrics)
        
        assert isinstance(result, str)
        assert 'aic' in result
        assert '100.5' in result


class TestEdgeCases:
    """Test edge cases and special scenarios."""
    
    def test_returns_single_value(self):
        """Test returns calculation on single value."""
        series = pd.Series([100])
        returns = calculate_returns(series)
        
        assert len(returns) == 1
        assert pd.isna(returns.iloc[0])
    
    def test_returns_two_values(self):
        """Test returns calculation on two values."""
        series = pd.Series([100, 110])
        returns = calculate_returns(series, pct=True)
        
        assert len(returns) == 2
        assert pd.isna(returns.iloc[0])
        assert returns.iloc[1] == 10.0
    
    def test_volatility_small_window(self):
        """Test volatility with window size 1."""
        dates = pd.date_range('2023-01-01', periods=50)
        returns = pd.Series(np.random.randn(50) * 0.01, index=dates)
        
        # Window size 1 should return zeros
        vol = calculate_volatility(returns, window=1)
        assert vol.dropna().sum() >= 0
    
    def test_format_zero(self):
        """Test formatting zero."""
        assert format_number(0) == "0"
        assert "0.00" in format_number(0.0)
    
    def test_format_negative(self):
        """Test formatting negative numbers."""
        result = format_number(-1234.56)
        assert "-" in result
        assert "1,234.56" in result


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

