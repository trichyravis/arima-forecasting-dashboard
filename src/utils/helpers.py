"""
Helper Functions Module - src/utils/helpers.py

Utility functions for data processing, calculations, and common operations.
Used across the ARIMA forecasting dashboard.

Author: Prof. V. Ravichandran
The Mountain Path - World of Finance
28+ Years Corporate Finance & Banking
10+ Years Academic Excellence
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Tuple, Optional, List, Dict, Union
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============================================================================
# DATE & TIME UTILITIES
# ============================================================================

def get_date_range(
    days_back: int = 365,
    end_date: Optional[str] = None
) -> Tuple[str, str]:
    """
    Generate date range for data fetching.
    
    Args:
        days_back: Number of days back from end_date
        end_date: End date (YYYY-MM-DD), default today
    
    Returns:
        Tuple of (start_date, end_date) as strings
    """
    if end_date is None:
        end_date = datetime.now().strftime("%Y-%m-%d")
    else:
        end_date = datetime.strptime(end_date, "%Y-%m-%d").strftime("%Y-%m-%d")
    
    start_date = (datetime.strptime(end_date, "%Y-%m-%d") - timedelta(days=days_back)).strftime("%Y-%m-%d")
    
    logger.info(f"Date range: {start_date} to {end_date}")
    
    return start_date, end_date


def get_trading_days(start_date: str, end_date: str) -> pd.DatetimeIndex:
    """
    Get trading days between dates (weekdays only).
    
    Args:
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
    
    Returns:
        DatetimeIndex of trading days
    """
    date_range = pd.bdate_range(start=start_date, end=end_date)
    logger.info(f"Trading days: {len(date_range)}")
    return date_range


def is_trading_day(date: Union[str, pd.Timestamp]) -> bool:
    """
    Check if date is a trading day (weekday).
    
    Args:
        date: Date to check
    
    Returns:
        True if trading day, False otherwise
    """
    if isinstance(date, str):
        date = pd.Timestamp(date)
    
    return date.weekday() < 5  # 0-4 are weekdays


# ============================================================================
# DATA VALIDATION & CLEANING
# ============================================================================

def validate_series(series: pd.Series, min_length: int = 50) -> Tuple[bool, List[str]]:
    """
    Validate time series for ARIMA analysis.
    
    Args:
        series: Time series to validate
        min_length: Minimum required length
    
    Returns:
        Tuple of (is_valid, error_list)
    """
    errors = []
    
    if series.empty:
        errors.append("Series is empty")
    
    if len(series) < min_length:
        errors.append(f"Series length ({len(series)}) < {min_length}")
    
    if series.isnull().any():
        missing_pct = series.isnull().sum() / len(series) * 100
        errors.append(f"Contains {missing_pct:.2f}% missing values")
    
    if not isinstance(series.index, pd.DatetimeIndex):
        errors.append("Index is not DatetimeIndex")
    
    if series.index.duplicated().any():
        errors.append("Index contains duplicates")
    
    is_valid = len(errors) == 0
    
    if is_valid:
        logger.info(f"Series validation passed ({len(series)} observations)")
    else:
        logger.warning(f"Series validation failed: {errors}")
    
    return is_valid, errors


def fill_missing_values(
    series: pd.Series,
    method: str = "ffill"
) -> pd.Series:
    """
    Fill missing values in series.
    
    Args:
        series: Series with missing values
        method: 'ffill' (forward fill), 'bfill' (backward fill), 'interpolate'
    
    Returns:
        Series with filled values
    """
    if method == "ffill":
        filled = series.fillna(method="ffill").fillna(method="bfill")
    elif method == "bfill":
        filled = series.fillna(method="bfill").fillna(method="ffill")
    elif method == "interpolate":
        filled = series.interpolate(method="linear")
    else:
        filled = series
    
    logger.info(f"Filled {series.isnull().sum()} missing values using {method}")
    
    return filled


def remove_outliers(
    series: pd.Series,
    method: str = "zscore",
    threshold: float = 3.0
) -> pd.Series:
    """
    Remove outliers from series.
    
    Args:
        series: Time series
        method: 'zscore' or 'iqr'
        threshold: Threshold for zscore (default 3.0)
    
    Returns:
        Series with outliers removed
    """
    if method == "zscore":
        z_scores = np.abs((series - series.mean()) / series.std())
        mask = z_scores < threshold
    elif method == "iqr":
        Q1 = series.quantile(0.25)
        Q3 = series.quantile(0.75)
        IQR = Q3 - Q1
        mask = (series >= Q1 - 1.5*IQR) & (series <= Q3 + 1.5*IQR)
    else:
        mask = pd.Series([True] * len(series), index=series.index)
    
    outliers_removed = (~mask).sum()
    logger.info(f"Removed {outliers_removed} outliers using {method}")
    
    return series[mask]


# ============================================================================
# STATISTICAL CALCULATIONS
# ============================================================================

def calculate_returns(prices: pd.Series, pct: bool = True) -> pd.Series:
    """
    Calculate returns from price series.
    
    Args:
        prices: Price series
        pct: Return as percentage (True) or decimal (False)
    
    Returns:
        Returns series
    """
    if pct:
        returns = prices.pct_change() * 100
    else:
        returns = prices.pct_change()
    
    logger.info(f"Calculated returns - Mean: {returns.mean():.4f}%, Std: {returns.std():.4f}%")
    
    return returns


def calculate_log_returns(prices: pd.Series) -> pd.Series:
    """
    Calculate log returns (more suitable for analysis).
    
    Args:
        prices: Price series
    
    Returns:
        Log returns series
    """
    log_returns = np.log(prices / prices.shift(1))
    return log_returns.dropna()


def calculate_volatility(
    returns: pd.Series,
    window: int = 20
) -> pd.Series:
    """
    Calculate rolling volatility.
    
    Args:
        returns: Returns series
        window: Rolling window size
    
    Returns:
        Volatility series
    """
    volatility = returns.rolling(window=window).std()
    logger.info(f"Calculated volatility (window={window})")
    return volatility


def calculate_trend(
    series: pd.Series,
    window: int = 20
) -> Tuple[pd.Series, str]:
    """
    Determine trend direction.
    
    Args:
        series: Time series
        window: Window for comparison
    
    Returns:
        Tuple of (trend_series, direction)
    """
    sma = series.rolling(window=window).mean()
    
    if series.iloc[-1] > sma.iloc[-1]:
        direction = "Uptrend"
    elif series.iloc[-1] < sma.iloc[-1]:
        direction = "Downtrend"
    else:
        direction = "Neutral"
    
    logger.info(f"Trend: {direction}")
    
    return sma, direction


# ============================================================================
# FORECAST UTILITIES
# ============================================================================

def create_forecast_df(
    forecast_values: np.ndarray,
    lower_ci: np.ndarray,
    upper_ci: np.ndarray,
    last_date: pd.Timestamp,
    freq: str = 'D'
) -> pd.DataFrame:
    """
    Create properly formatted forecast DataFrame.
    
    Args:
        forecast_values: Forecast values
        lower_ci: Lower confidence interval
        upper_ci: Upper confidence interval
        last_date: Last date in series
        freq: Frequency ('D' for daily, etc.)
    
    Returns:
        Formatted forecast DataFrame
    """
    future_dates = pd.date_range(start=last_date, periods=len(forecast_values)+1, freq=freq)[1:]
    
    df = pd.DataFrame({
        'forecast': forecast_values,
        'lower_ci': lower_ci,
        'upper_ci': upper_ci
    }, index=future_dates)
    
    return df


def combine_actual_forecast(
    actual: pd.Series,
    forecast: pd.DataFrame,
    lookback_periods: int = 30
) -> pd.DataFrame:
    """
    Combine actual and forecasted values for plotting.
    
    Args:
        actual: Actual values series
        forecast: Forecast DataFrame
        lookback_periods: Number of recent actual values to include
    
    Returns:
        Combined DataFrame
    """
    # Get recent actual values
    recent_actual = actual.tail(lookback_periods)
    
    # Create combined DataFrame
    combined = pd.DataFrame({
        'actual': recent_actual,
        'type': 'Actual'
    })
    
    # Add forecast
    forecast_df = forecast.copy()
    forecast_df['type'] = 'Forecast'
    forecast_df['actual'] = forecast_df['forecast']
    
    combined = pd.concat([combined, forecast_df[['actual', 'type']]])
    
    logger.info(f"Combined {len(recent_actual)} actual + {len(forecast)} forecast")
    
    return combined


# ============================================================================
# MODEL UTILITIES
# ============================================================================

def get_model_name(order: Tuple[int, int, int]) -> str:
    """
    Get readable model name from parameters.
    
    Args:
        order: (p, d, q) tuple
    
    Returns:
        Model name string
    """
    p, d, q = order
    return f"ARIMA({p},{d},{q})"


def format_metrics(metrics: Dict) -> str:
    """
    Format metrics dictionary for display.
    
    Args:
        metrics: Dictionary of metrics
    
    Returns:
        Formatted string
    """
    output = "Model Metrics:\n"
    output += "-" * 40 + "\n"
    
    for key, value in metrics.items():
        if isinstance(value, float):
            output += f"{key:.<30} {value:>10.4f}\n"
        else:
            output += f"{key:.<30} {value:>10}\n"
    
    return output


def calculate_confidence_level(ci_width: float) -> float:
    """
    Calculate confidence level from CI width.
    
    Args:
        ci_width: Width of confidence interval
    
    Returns:
        Confidence level (e.g., 0.95 for 95%)
    """
    # Assumes symmetric CI
    return 1 - (1 - ci_width) * 2


# ============================================================================
# FORMATTING & DISPLAY
# ============================================================================

def format_number(value: Union[int, float], decimals: int = 2) -> str:
    """
    Format number for display.
    
    Args:
        value: Number to format
        decimals: Number of decimal places
    
    Returns:
        Formatted string
    """
    if isinstance(value, int):
        return f"{value:,}"
    else:
        return f"{value:,.{decimals}f}"


def format_percentage(value: float, decimals: int = 2) -> str:
    """
    Format value as percentage.
    
    Args:
        value: Value (0-1 or 0-100)
        decimals: Decimal places
    
    Returns:
        Formatted percentage string
    """
    if value <= 1:
        value = value * 100
    
    return f"{value:.{decimals}f}%"


def format_date(date: Union[str, pd.Timestamp], fmt: str = "%Y-%m-%d") -> str:
    """
    Format date consistently.
    
    Args:
        date: Date to format
        fmt: Format string
    
    Returns:
        Formatted date string
    """
    if isinstance(date, str):
        date = pd.Timestamp(date)
    
    return date.strftime(fmt)


# ============================================================================
# BATCH OPERATIONS
# ============================================================================

def batch_calculate_metrics(
    models_dict: Dict,
    metrics_func
) -> pd.DataFrame:
    """
    Calculate metrics for multiple models.
    
    Args:
        models_dict: Dictionary of {name: model}
        metrics_func: Function to calculate metrics
    
    Returns:
        DataFrame with metrics comparison
    """
    results = []
    
    for name, model in models_dict.items():
        metrics = metrics_func(model)
        metrics['model'] = name
        results.append(metrics)
    
    df = pd.DataFrame(results)
    
    logger.info(f"Calculated metrics for {len(models_dict)} models")
    
    return df


def save_results(
    data: Union[pd.DataFrame, pd.Series],
    filepath: str,
    format: str = "csv"
) -> bool:
    """
    Save analysis results to file.
    
    Args:
        data: Data to save
        filepath: Output file path
        format: 'csv', 'excel', or 'json'
    
    Returns:
        True if successful
    """
    try:
        if format == "csv":
            data.to_csv(filepath)
        elif format == "excel":
            data.to_excel(filepath)
        elif format == "json":
            data.to_json(filepath)
        
        logger.info(f"Saved results to {filepath}")
        return True
    
    except Exception as e:
        logger.error(f"Error saving results: {str(e)}")
        return False


# Example usage
if __name__ == "__main__":
    # Test helper functions
    prices = pd.Series(
        np.random.randn(100).cumsum() + 100,
        index=pd.date_range('2023-01-01', periods=100)
    )
    
    print("Testing helpers.py")
    print("=" * 50)
    
    # Date utilities
    start, end = get_date_range(days_back=30)
    print(f"\nDate range: {start} to {end}")
    
    # Data validation
    is_valid, errors = validate_series(prices)
    print(f"Series valid: {is_valid}")
    
    # Statistical calculations
    returns = calculate_returns(prices)
    print(f"\nReturns - Mean: {returns.mean():.4f}%, Std: {returns.std():.4f}%")
    
    # Trend analysis
    sma, direction = calculate_trend(prices, window=20)
    print(f"Trend direction: {direction}")
    
    # Formatting
    print(f"\nFormatted number: {format_number(1234567.89)}")
    print(f"Formatted percentage: {format_percentage(0.1234)}")
    print(f"Formatted date: {format_date(pd.Timestamp.now())}")

