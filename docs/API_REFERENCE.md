# API Reference - ARIMA Forecasting Dashboard

## Core Modules

### src.data.loader - Data Loading

#### DataLoader Class

```python
from src.data.loader import DataLoader

loader = DataLoader(cache_dir="data/cache")
```

**Methods:**

##### fetch_from_yfinance(ticker, start_date, end_date)
Fetch data from Yahoo Finance.

```python
df = loader.fetch_from_yfinance(
    ticker="^NSEI",
    start_date="2023-01-01",
    end_date="2024-01-01"
)
```

**Parameters:**
- `ticker` (str): Stock ticker symbol
- `start_date` (str): YYYY-MM-DD format
- `end_date` (str): YYYY-MM-DD format

**Returns:** pandas.DataFrame with OHLCV data

##### load_from_csv(filepath, date_column, parse_dates)
Load data from CSV file.

```python
df = loader.load_from_csv(
    filepath="data.csv",
    date_column="Date",
    parse_dates=True
)
```

##### load_from_excel(filepath, sheet_name)
Load data from Excel file.

```python
df = loader.load_from_excel(
    filepath="data.xlsx",
    sheet_name="Sheet1"
)
```

##### validate_data(series)
Validate time series for ARIMA analysis.

```python
is_valid, errors = loader.validate_data(series)
if is_valid:
    print("✅ Series is valid for ARIMA")
else:
    print(f"❌ Errors: {errors}")
```

**Returns:** (bool, list) - (is_valid, error_messages)

### src.models.arima - ARIMA Models

#### ARIMAModel Class

```python
from src.models.arima import ARIMAModel

model = ARIMAModel(series, name="NIFTY")
```

**Methods:**

##### check_stationarity()
Test series for stationarity using ADF and KPSS tests.

```python
result = model.check_stationarity()
print(result['summary']['is_stationary'])  # True/False
print(result['adf']['p_value'])             # P-value
```

**Returns:** dict with test results

##### auto_select_parameters(max_p, max_d, max_q)
Automatically select ARIMA parameters using stepwise algorithm.

```python
params = model.auto_select_parameters(max_p=5, max_d=2, max_q=5)
print(params)  # (1, 1, 1)
```

**Returns:** (p, d, q) tuple

##### fit(order)
Fit ARIMA model with specified order.

```python
stats = model.fit((1, 1, 1))
print(stats['aic'])    # AIC value
print(stats['bic'])    # BIC value
```

**Parameters:**
- `order` (tuple): (p, d, q) parameters

**Returns:** dict with model statistics

##### forecast(steps, alpha)
Generate forecast with confidence intervals.

```python
forecast_df = model.forecast(steps=30, alpha=0.05)
# Columns: forecast, lower_ci, upper_ci
```

**Parameters:**
- `steps` (int): Number of periods to forecast
- `alpha` (float): Significance level (0.05 for 95% CI)

**Returns:** pandas.DataFrame with forecast

##### get_residuals()
Extract model residuals.

```python
residuals = model.get_residuals()
```

**Returns:** pandas.Series of residuals

##### calculate_metrics()
Calculate model performance metrics.

```python
metrics = model.calculate_metrics()
# Returns: AIC, BIC, RMSE, MAE, etc.
```

### src.utils.helpers - Utility Functions

#### Date Functions

```python
from src.utils.helpers import (
    get_date_range,
    get_trading_days,
    is_trading_day
)

# Get date range
start, end = get_date_range(days_back=365)

# Get trading days
trading_days = get_trading_days("2023-01-01", "2024-01-01")

# Check if trading day
is_trading = is_trading_day("2023-01-02")  # True
```

#### Statistical Functions

```python
from src.utils.helpers import (
    calculate_returns,
    calculate_volatility,
    calculate_trend
)

# Calculate returns
returns = calculate_returns(prices, pct=True)

# Calculate rolling volatility
volatility = calculate_volatility(returns, window=20)

# Determine trend
sma, direction = calculate_trend(prices, window=20)
print(direction)  # "Uptrend", "Downtrend", or "Neutral"
```

#### Formatting Functions

```python
from src.utils.helpers import (
    format_number,
    format_percentage,
    format_date
)

# Format numbers
formatted = format_number(1234567.89)  # "1,234,567.89"

# Format percentages
pct_str = format_percentage(0.1234)  # "12.34%"

# Format dates
date_str = format_date(pd.Timestamp("2023-06-15"))  # "2023-06-15"
```

### src.visualization.charts - Charting

#### TimeSeriesCharts Class

```python
from src.visualization.charts import TimeSeriesCharts

# Plot time series
fig = TimeSeriesCharts.plot_series(
    series=price_series,
    title="NIFTY 50 Price",
    show_ma=True,
    ma_window=20
)
fig.show()

# Plot forecast
fig = TimeSeriesCharts.plot_forecast(
    actual=price_series,
    forecast=forecast_df,
    lookback=50
)
fig.show()
```

## Configuration

### src.config.py

Key configuration variables:

```python
# Colors
DARK_BLUE = "#003366"
LIGHT_BLUE = "#004d80"
GOLD_COLOR = "#FFD700"

# ARIMA parameters
MAX_P = 5
MAX_D = 2
MAX_Q = 5

# Data settings
MIN_OBSERVATIONS = 50
DEFAULT_CACHE_EXPIRE = 24  # hours
```

## Error Handling

### Common Exceptions

```python
try:
    model.fit((1, 1, 1))
except ValueError as e:
    print(f"❌ Value Error: {e}")
except RuntimeError as e:
    print(f"❌ Runtime Error: {e}")
```

## Example: Complete Workflow

```python
import pandas as pd
from src.data.loader import DataLoader
from src.models.arima import ARIMAModel
from src.visualization.charts import TimeSeriesCharts

# 1. Load data
loader = DataLoader()
df = loader.fetch_from_yfinance("^NSEI", "2023-01-01", "2024-01-01")

# 2. Prepare series
series = loader.prepare_series(df, column='Close')

# 3. Check stationarity
model = ARIMAModel(series)
stationarity = model.check_stationarity()

# 4. Auto-select parameters
if not stationarity['summary']['is_stationary']:
    params = model.auto_select_parameters()
else:
    params = (1, 0, 1)

# 5. Fit model
model.fit(params)

# 6. Generate forecast
forecast = model.forecast(steps=30)

# 7. Visualize
fig = TimeSeriesCharts.plot_forecast(series, forecast)
fig.show()
```

## Testing

### Running Unit Tests

```bash
# All tests
pytest tests/ -v

# Specific module
pytest tests/test_arima.py -v

# With coverage
pytest tests/ --cov=src
```

---

**For more details, see the module docstrings and examples.**
