"""
Unit Tests for Data Loader Module - tests/test_loader.py

Tests for src/data/loader.py using pytest.

Author: Prof. V. Ravichandran
The Mountain Path - World of Finance
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from src.data.loader import DataLoader


class TestDataLoader:
    """Test DataLoader class."""
    
    @pytest.fixture
    def loader(self):
        """Initialize loader for testing."""
        return DataLoader(cache_dir="data/cache")
    
    @pytest.fixture
    def sample_series(self):
        """Create sample time series."""
        dates = pd.date_range('2023-01-01', periods=100)
        prices = 100 + np.cumsum(np.random.randn(100) * 0.5)
        return pd.Series(prices, index=dates)
    
    def test_loader_initialization(self, loader):
        """Test loader initializes correctly."""
        assert loader is not None
        assert loader.cache_dir == "data/cache"
    
    def test_validate_series_valid(self, sample_series):
        """Test validation of valid series."""
        loader = DataLoader()
        is_valid, errors = loader.validate_data(sample_series)
        
        assert is_valid == True
        assert len(errors) == 0
    
    def test_validate_series_empty(self):
        """Test validation rejects empty series."""
        loader = DataLoader()
        empty_series = pd.Series([], dtype=float)
        is_valid, errors = loader.validate_data(empty_series)
        
        assert is_valid == False
        assert len(errors) > 0
    
    def test_validate_series_short(self):
        """Test validation rejects short series."""
        loader = DataLoader()
        short_series = pd.Series(np.random.randn(10))
        is_valid, errors = loader.validate_data(short_series)
        
        assert is_valid == False
        assert "Less than 50" in str(errors)
    
    def test_validate_series_with_nans(self):
        """Test validation detects NaN values."""
        loader = DataLoader()
        series_with_nans = pd.Series([1, 2, np.nan, 4, 5] * 15)
        is_valid, errors = loader.validate_data(series_with_nans)
        
        assert is_valid == False
        assert any("missing" in e.lower() for e in errors)
    
    def test_validate_series_duplicates(self):
        """Test validation detects duplicate dates."""
        loader = DataLoader()
        dates = [pd.Timestamp('2023-01-01')] * 50
        series_dup = pd.Series(np.random.randn(50), index=dates)
        is_valid, errors = loader.validate_data(series_dup)
        
        assert is_valid == False
        assert any("duplicate" in e.lower() for e in errors)
    
    def test_prepare_series(self, sample_series):
        """Test series preparation."""
        loader = DataLoader()
        prepared = loader.prepare_series(sample_series)
        
        assert isinstance(prepared, pd.Series)
        assert len(prepared) > 0
        assert prepared.isnull().sum() == 0
    
    def test_prepare_series_removes_nans(self):
        """Test prepare removes NaN values."""
        loader = DataLoader()
        df = pd.DataFrame({
            'Close': [1, 2, np.nan, 4, 5]
        }, index=pd.date_range('2023-01-01', periods=5))
        
        series = loader.prepare_series(df, column='Close')
        assert series.isnull().sum() == 0
        assert len(series) == 4
    
    def test_csv_loading(self, tmp_path):
        """Test loading from CSV."""
        loader = DataLoader()
        
        # Create test CSV
        df = pd.DataFrame({
            'Date': pd.date_range('2023-01-01', periods=50),
            'Close': np.random.randn(50).cumsum() + 100
        })
        
        csv_file = tmp_path / "test.csv"
        df.to_csv(csv_file, index=False)
        
        # Load and test
        loaded_df = loader.load_from_csv(str(csv_file))
        
        assert isinstance(loaded_df, pd.DataFrame)
        assert len(loaded_df) == 50
        assert 'Close' in loaded_df.columns
    
    def test_excel_loading(self, tmp_path):
        """Test loading from Excel."""
        pytest.importorskip("openpyxl")
        
        loader = DataLoader()
        
        # Create test Excel
        df = pd.DataFrame({
            'Date': pd.date_range('2023-01-01', periods=50),
            'Close': np.random.randn(50).cumsum() + 100
        })
        
        excel_file = tmp_path / "test.xlsx"
        df.to_excel(excel_file, sheet_name='Data', index=False)
        
        # Load and test
        loaded_df = loader.load_from_excel(str(excel_file), sheet_name='Data')
        
        assert isinstance(loaded_df, pd.DataFrame)
        assert len(loaded_df) == 50


class TestDataValidation:
    """Test data validation functions."""
    
    def test_datetime_index_validation(self):
        """Test datetime index validation."""
        loader = DataLoader()
        
        # Valid datetime index
        dates = pd.date_range('2023-01-01', periods=100)
        series = pd.Series(np.random.randn(100), index=dates)
        is_valid, errors = loader.validate_data(series)
        
        # Check datetime index is recognized
        assert any("DatetimeIndex" not in e for e in errors)
    
    def test_timezone_handling(self):
        """Test handling of timezone-aware datetimes."""
        loader = DataLoader()
        
        # Create timezone-aware series
        dates = pd.date_range('2023-01-01', periods=100, tz='UTC')
        series = pd.Series(np.random.randn(100), index=dates)
        
        # Should handle timezone conversion
        is_valid, errors = loader.validate_data(series)
        
        # Might have errors but shouldn't crash
        assert isinstance(is_valid, bool)
        assert isinstance(errors, list)


class TestEdgeCases:
    """Test edge cases and error handling."""
    
    def test_single_value_series(self):
        """Test handling of single-value series."""
        loader = DataLoader()
        series = pd.Series([100], index=[pd.Timestamp('2023-01-01')])
        is_valid, errors = loader.validate_data(series)
        
        assert is_valid == False
    
    def test_constant_series(self):
        """Test handling of constant series."""
        loader = DataLoader()
        series = pd.Series([100] * 100, 
                          index=pd.date_range('2023-01-01', periods=100))
        is_valid, errors = loader.validate_data(series)
        
        # Constant series is valid but might warn
        assert isinstance(is_valid, bool)
    
    def test_very_large_values(self):
        """Test handling of very large values."""
        loader = DataLoader()
        series = pd.Series(np.random.randn(100) * 1e10 + 1e10,
                          index=pd.date_range('2023-01-01', periods=100))
        is_valid, errors = loader.validate_data(series)
        
        assert isinstance(is_valid, bool)
    
    def test_negative_values(self):
        """Test handling of negative values."""
        loader = DataLoader()
        series = pd.Series(-np.random.randn(100),
                          index=pd.date_range('2023-01-01', periods=100))
        is_valid, errors = loader.validate_data(series)
        
        assert isinstance(is_valid, bool)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

