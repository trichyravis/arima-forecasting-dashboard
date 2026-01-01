# Troubleshooting Guide

## Installation Issues

### ModuleNotFoundError: No module named 'pandas'

**Problem:** Missing Python packages

**Solutions:**
```bash
# Reinstall requirements
pip install --upgrade pip
pip install -r requirements.txt

# Or install specific packages
pip install pandas numpy statsmodels
```

### Python version not supported

**Problem:** Python < 3.8

**Solution:**
```bash
# Check version
python --version

# Update to Python 3.8+
# Visit https://www.python.org/downloads/
```

### Permission denied on macOS

**Problem:** Cannot install packages

**Solution:**
```bash
# Use virtual environment (recommended)
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Data Loading Issues

### FileNotFoundError: CSV file not found

**Problem:** File path is incorrect

**Solution:**
```python
# Check file path
import os
print(os.path.exists("data/prices.csv"))

# Use absolute path
from pathlib import Path
file_path = Path(__file__).parent / "data" / "prices.csv"
```

### ValueError: No columns to parse

**Problem:** CSV format is incorrect

**Solution:**
Check CSV format:
```
Date,Close,High,Low
2023-01-01,100.5,101.2,99.8
2023-01-02,101.2,102.0,100.5
```

Required columns:
- **Date** (YYYY-MM-DD format)
- **Close** (numeric prices)

### Empty DataFrame returned

**Problem:** Yahoo Finance data unavailable

**Solutions:**
```python
# Check internet connection
import requests
requests.get("https://finance.yahoo.com")

# Use different ticker
loader.fetch_from_yfinance("^NSEI")  # NIFTY
loader.fetch_from_yfinance("^BSESN")  # SENSEX

# Extend date range
df = loader.fetch_from_yfinance(
    ticker="^NSEI",
    start_date="2022-01-01",
    end_date="2024-01-01"
)
```

## Data Validation Issues

### ValueError: Series is empty

**Problem:** No data in time series

**Solution:**
```python
# Check data length
print(len(series))  # Should be >= 50

# Reload with more observations
df = loader.fetch_from_yfinance(ticker, start_date="2022-01-01")
```

### ValueError: Contains NaN values

**Problem:** Missing data points

**Solution:**
```python
# Fill missing values
series = series.fillna(method='ffill')

# Or remove NaN
series = series.dropna()

# Check percentage missing
print(f"{series.isnull().sum() / len(series) * 100:.2f}%")
```

### ValueError: Contains duplicate dates

**Problem:** Duplicate index entries

**Solution:**
```python
# Remove duplicates (keep first)
series = series[~series.index.duplicated(keep='first')]

# Or aggregate duplicates
series = series.groupby(series.index).mean()
```

## ARIMA Model Issues

### RuntimeError: Failed to converge

**Problem:** ARIMA model fitting failed

**Solutions:**
```python
# Try simpler model
model.fit((1, 0, 0))

# Check stationarity
result = model.check_stationarity()
print(result['summary'])

# Increase d parameter
model.fit((1, 2, 1))  # d=2 instead of d=1
```

### ValueError: Series is not stationary

**Problem:** Cannot fit model without differencing

**Solution:**
```python
# Increase differencing
stats = model.fit((1, 2, 1))  # d=2

# Or use seasonal differencing
model.fit((1, 1, 1))  # Seasonal ARIMA

# Check stationarity after differencing
diff_series = series.diff().dropna()
model.check_stationarity()
```

### Memory error with large datasets

**Problem:** Out of memory

**Solutions:**
```python
# Use subset of data
series_subset = series[-1000:]  # Last 1000 obs

# Reduce ARIMA parameters
# In src/config.py:
MAX_P = 2
MAX_D = 1
MAX_Q = 2
```

## Forecast Issues

### ValueError: Forecast shape mismatch

**Problem:** Forecast generation failed

**Solution:**
```python
# Ensure model is fitted first
model.fit((1, 1, 1))

# Then generate forecast
forecast = model.forecast(steps=30)

# Check forecast shape
print(forecast.shape)  # Should be (30, 3)
```

### Unrealistic forecast values

**Problem:** Forecast seems wrong

**Solutions:**
```python
# Check residual plots
residuals = model.get_residuals()
print(residuals.describe())

# Verify data quality
print(series.describe())

# Try different parameters
for p in range(3):
    for d in range(2):
        for q in range(3):
            model.fit((p, d, q))
            print(f"({p},{d},{q}): AIC={model.aic}")
```

## Visualization Issues

### ModuleNotFoundError: No module named 'plotly'

**Problem:** Plotly not installed

**Solution:**
```bash
pip install plotly
```

### Empty or blank plots

**Problem:** Data not visualized properly

**Solution:**
```python
# Check data is not empty
print(len(series))

# Verify Plotly installation
import plotly
print(plotly.__version__)

# Try simple plot
import plotly.express as px
fig = px.line(y=series.values)
fig.show()
```

## Streamlit Issues

### StreamlitAPIException: Streamlit not installed

**Problem:** Streamlit missing

**Solution:**
```bash
pip install streamlit
streamlit --version
```

### Port already in use

**Problem:** Port 8501 occupied

**Solutions:**
```bash
# Use different port
streamlit run app.py --server.port 8502

# Kill process using port
lsof -ti:8501 | xargs kill -9
```

### Slow dashboard performance

**Problem:** Streamlit cache not working

**Solutions:**
```python
# Clear cache
streamlit cache clear

# Reduce rerun frequency
@st.cache_data
def load_data():
    return loader.fetch_from_yfinance(...)

# Reduce ARIMA parameters
# In src/config.py
MAX_P = 2  # Was 5
```

### "File not found" error

**Problem:** Streamlit can't find files

**Solution:**
```python
# Use Path from pathlib
from pathlib import Path
file_path = Path(__file__).parent / "data" / "file.csv"

# Or absolute path
import os
os.path.abspath("data/file.csv")
```

## Testing Issues

### pytest: ModuleNotFoundError

**Problem:** Tests can't import modules

**Solution:**
```bash
# Run from project root
cd /path/to/project
pytest tests/ -v

# Add project to Python path
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
pytest tests/ -v
```

### Test failures on data validation

**Problem:** Tests expect specific data

**Solution:**
```bash
# Ensure pandas/numpy are installed
pip install pandas numpy

# Run tests with verbose output
pytest tests/ -v -s

# Run specific test
pytest tests/test_loader.py::TestDataLoader -v
```

## Performance Issues

### Slow data loading

**Problem:** Downloading from Yahoo Finance is slow

**Solutions:**
```python
# Enable caching
loader = DataLoader(cache_dir="data/cache")

# Reduce date range
loader.fetch_from_yfinance(ticker, "2023-01-01", "2024-01-01")

# Use local CSV instead
loader.load_from_csv("data/prices.csv")
```

### Slow ARIMA fitting

**Problem:** Parameter selection takes too long

**Solutions:**
```python
# Reduce parameter space
# In src/config.py
MAX_P = 2
MAX_D = 1
MAX_Q = 2

# Or specify parameters directly
model.fit((1, 1, 1))
```

### Dashboard sluggish

**Problem:** Streamlit recomputing on every interaction

**Solution:**
```python
# Add caching decorators
@st.cache_data
def load_data():
    return loader.fetch_from_yfinance(...)

@st.cache_resource
def fit_model(series):
    model = ARIMAModel(series)
    return model
```

## Getting Help

### Check logs for errors

```bash
# View application logs
tail -f logs/app.log

# Check Streamlit logs
streamlit run app.py 2>&1 | tee app_output.log
```

### Enable debug mode

```python
# In src/config.py
DEBUG_MODE = True
LOG_LEVEL = "DEBUG"
```

### Search documentation

- [README.md](README.md) - Overview
- [INSTALLATION.md](INSTALLATION.md) - Setup guide
- [USAGE.md](USAGE.md) - Usage examples
- [API_REFERENCE.md](API_REFERENCE.md) - API docs
- [CONFIGURATION.md](CONFIGURATION.md) - Configuration options

### Common Issues Checklist

- [ ] Python 3.8 or higher?
- [ ] All packages installed (`pip install -r requirements.txt`)?
- [ ] Data file exists and readable?
- [ ] Date range has at least 50 observations?
- [ ] Data doesn't have too many NaN values?
- [ ] ARIMA parameters reasonable (p,d,q < 10)?
- [ ] Running from project root directory?

---

**Still having issues?** Check the educational materials in `docs/educational/`

