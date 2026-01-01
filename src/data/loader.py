"""
Data Loader Module - src/data/loader.py

Handles data fetching and loading for ARIMA forecasting dashboard.
Supports multiple data sources (yfinance, CSV, Excel).

Author: Prof. V. Ravichandran
The Mountain Path - World of Finance
"""

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
from typing import Optional, Tuple, List
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataLoader:
    """
    Load financial time series data from various sources.
    
    Supports:
    - Yahoo Finance (yfinance)
    - CSV files
    - Excel files
    """
    
    def __init__(self, cache_dir: str = "data/cache"):
        """
        Initialize DataLoader.
        
        Args:
            cache_dir: Directory for cached data
        """
        self.cache_dir = cache_dir
        logger.info(f"DataLoader initialized with cache_dir: {cache_dir}")
    
    def fetch_from_yfinance(
        self,
        ticker: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        interval: str = "1d"
    ) -> pd.DataFrame:
        """
        Fetch data from Yahoo Finance.
        
        Args:
            ticker: Stock ticker symbol (e.g., 'NIFTY.NS')
            start_date: Start date (YYYY-MM-DD) - default 1 year ago
            end_date: End date (YYYY-MM-DD) - default today
            interval: '1d', '1wk', '1mo'
        
        Returns:
            DataFrame with OHLCV data
        
        Example:
            >>> loader = DataLoader()
            >>> df = loader.fetch_from_yfinance('NIFTY.NS')
            >>> print(df.head())
        """
        try:
            # Set default dates
            if end_date is None:
                end_date = datetime.now().strftime("%Y-%m-%d")
            if start_date is None:
                start_date = (datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d")
            
            logger.info(f"Fetching {ticker} from {start_date} to {end_date}")
            
            # Download data
            data = yf.download(
                ticker,
                start=start_date,
                end=end_date,
                interval=interval,
                progress=False
            )
            
            if data.empty:
                raise ValueError(f"No data found for ticker: {ticker}")
            
            # Convert index to datetime
            data.index = pd.to_datetime(data.index)
            
            # Ensure index is timezone-naive for ARIMA
            if data.index.tz is not None:
                data.index = data.index.tz_localize(None)
            
            logger.info(f"Successfully fetched {len(data)} records for {ticker}")
            
            return data
        
        except Exception as e:
            logger.error(f"Error fetching data: {str(e)}")
            raise
    
    def load_from_csv(
        self,
        filepath: str,
        date_column: str = "Date",
        price_column: str = "Close"
    ) -> pd.DataFrame:
        """
        Load data from CSV file.
        
        Args:
            filepath: Path to CSV file
            date_column: Name of date column
            price_column: Name of closing price column
        
        Returns:
            DataFrame with parsed dates
        """
        try:
            logger.info(f"Loading CSV from: {filepath}")
            
            df = pd.read_csv(filepath)
            df[date_column] = pd.to_datetime(df[date_column])
            df.set_index(date_column, inplace=True)
            df = df.sort_index()
            
            logger.info(f"Successfully loaded {len(df)} records from CSV")
            
            return df
        
        except Exception as e:
            logger.error(f"Error loading CSV: {str(e)}")
            raise
    
    def load_from_excel(
        self,
        filepath: str,
        sheet_name: str = 0,
        date_column: str = "Date",
        price_column: str = "Close"
    ) -> pd.DataFrame:
        """
        Load data from Excel file.
        
        Args:
            filepath: Path to Excel file
            sheet_name: Sheet name or index
            date_column: Name of date column
            price_column: Name of closing price column
        
        Returns:
            DataFrame with parsed dates
        """
        try:
            logger.info(f"Loading Excel from: {filepath}")
            
            df = pd.read_excel(filepath, sheet_name=sheet_name)
            df[date_column] = pd.to_datetime(df[date_column])
            df.set_index(date_column, inplace=True)
            df = df.sort_index()
            
            logger.info(f"Successfully loaded {len(df)} records from Excel")
            
            return df
        
        except Exception as e:
            logger.error(f"Error loading Excel: {str(e)}")
            raise
    
    def validate_data(self, df: pd.DataFrame) -> Tuple[bool, List[str]]:
        """
        Validate loaded data for ARIMA analysis.
        
        Args:
            df: DataFrame to validate
        
        Returns:
            Tuple of (is_valid, error_messages)
        """
        errors = []
        
        # Check if empty
        if df.empty:
            errors.append("DataFrame is empty")
        
        # Check for missing values
        if df.isnull().any().any():
            missing_pct = df.isnull().sum().sum() / df.size * 100
            errors.append(f"Contains {missing_pct:.2f}% missing values")
        
        # Check if datetime index
        if not isinstance(df.index, pd.DatetimeIndex):
            errors.append("Index is not DatetimeIndex")
        
        # Check for duplicates
        if df.index.duplicated().any():
            errors.append("Index contains duplicates")
        
        # Check minimum length for ARIMA
        if len(df) < 50:
            errors.append("Less than 50 observations (minimum for ARIMA)")
        
        is_valid = len(errors) == 0
        
        if is_valid:
            logger.info("Data validation passed")
        else:
            logger.warning(f"Data validation failed: {errors}")
        
        return is_valid, errors
    
    def prepare_series(
        self,
        df: pd.DataFrame,
        column: str = "Close"
    ) -> pd.Series:
        """
        Prepare time series for ARIMA analysis.
        
        Args:
            df: DataFrame with price data
            column: Column to extract (default 'Close')
        
        Returns:
            Prepared Series with proper index
        """
        try:
            series = df[column].copy()
            series = series.dropna()
            
            logger.info(f"Prepared series with {len(series)} observations")
            
            return series
        
        except Exception as e:
            logger.error(f"Error preparing series: {str(e)}")
            raise


# Example usage
if __name__ == "__main__":
    # Initialize loader
    loader = DataLoader()
    
    # Fetch NIFTY data
    df = loader.fetch_from_yfinance("^NSEI", start_date="2023-01-01")
    
    # Validate
    is_valid, errors = loader.validate_data(df)
    if is_valid:
        print("✅ Data validation passed")
    else:
        print(f"❌ Validation errors: {errors}")
    
    # Prepare series
    series = loader.prepare_series(df, column="Close")
    print(f"\nSeries info:")
    print(f"Length: {len(series)}")
    print(f"Date range: {series.index[0]} to {series.index[-1]}")
    print(f"\nFirst 5 values:\n{series.head()}")

