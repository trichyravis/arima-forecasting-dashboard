# Configuration Guide

## Configuration Files

### 1. src/config.py - Application Configuration

Main configuration file for the ARIMA dashboard.

```python
# Color Scheme (Mountain Path Design)
DARK_BLUE = "#003366"
LIGHT_BLUE = "#004d80"
GOLD_COLOR = "#FFD700"

# ARIMA Model Parameters
MAX_P = 5
MAX_D = 2
MAX_Q = 5
MIN_OBSERVATIONS = 50

# Cache Settings
DEFAULT_CACHE_EXPIRE = 24  # hours
CACHE_DIR = "data/cache"
RESULTS_DIR = "data/results"

# Logging
LOG_LEVEL = "INFO"
LOG_DIR = "logs"
LOG_FILE = "app.log"
```

### 2. .streamlit/config.toml - Streamlit Configuration

Customize Streamlit app appearance and behavior.

```toml
[theme]
primaryColor = "#003366"
backgroundColor = "#FFFFFF"
secondaryBackgroundColor = "#F0F0F0"
textColor = "#003366"
font = "sans serif"

[client]
toolbarMode = "viewer"
showErrorDetails = false
showWarningOnDirectExecution = true

[logger]
level = "info"

[server]
maxUploadSize = 200
enableXsrfProtection = true
```

### 3. requirements.txt - Python Dependencies

```
pandas>=1.3.0
numpy>=1.21.0
scipy>=1.7.0
statsmodels>=0.13.0
pmdarima>=2.0.0
scikit-learn>=1.0.0
yfinance>=0.1.70
plotly>=5.0.0
streamlit>=1.0.0
```

## Environment Variables

### Setting Environment Variables

**macOS/Linux:**
```bash
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
export LOG_LEVEL="DEBUG"
```

**Windows:**
```cmd
set PYTHONPATH=%PYTHONPATH%;%cd%
set LOG_LEVEL=DEBUG
```

### Common Environment Variables

```bash
# Logging
LOG_LEVEL=DEBUG|INFO|WARNING|ERROR|CRITICAL

# Cache settings
CACHE_EXPIRE_HOURS=24

# Data paths
DATA_DIR=data
CACHE_DIR=data/cache
LOG_DIR=logs

# API Keys (if needed)
YAHOO_API_KEY=your_key_here
```

## Advanced Configuration

### ARIMA Parameter Tuning

Edit `src/config.py`:

```python
# For faster optimization (fewer parameters to test)
MAX_P = 3
MAX_D = 1
MAX_Q = 3

# For more thorough search (slower)
MAX_P = 10
MAX_D = 3
MAX_Q = 10
```

### Cache Configuration

```python
# Cache expiration (hours)
DEFAULT_CACHE_EXPIRE = 24

# Cache location
CACHE_DIR = "data/cache"

# Enable/disable caching
USE_CACHE = True
```

### Logging Configuration

```python
# Log level: DEBUG, INFO, WARNING, ERROR, CRITICAL
LOG_LEVEL = "INFO"

# Log directory
LOG_DIR = "logs"

# Log file name
LOG_FILE = "app.log"

# Maximum log file size (MB)
LOG_MAX_SIZE = 10

# Number of backup log files
LOG_BACKUP_COUNT = 5
```

### Streamlit Performance

Optimize for better performance in `config.toml`:

```toml
[client]
toolbarMode = "minimal"  # Reduce toolbar

[server]
maxUploadSize = 100      # Limit upload size (MB)

[logger]
level = "warning"        # Reduce logging verbosity
```

## Database Configuration

Currently uses file-based caching. For production, consider:

### SQLite Configuration

```python
# Enable SQLite cache
USE_SQLITE_CACHE = True
SQLITE_DB_PATH = "data/cache.db"
```

### PostgreSQL Configuration

```python
# PostgreSQL settings
POSTGRES_HOST = "localhost"
POSTGRES_PORT = 5432
POSTGRES_DB = "arima_cache"
POSTGRES_USER = "postgres"
POSTGRES_PASSWORD = "your_password"
```

## API Configuration

### Yahoo Finance API

```python
# Default tickers
DEFAULT_TICKERS = {
    "NIFTY": "^NSEI",
    "SENSEX": "^BSESN",
    "INDIAVIX": "^INDIAVIX"
}

# Data fetch timeout (seconds)
FETCH_TIMEOUT = 30
```

### Custom Data Sources

Add custom data sources in `src/config.py`:

```python
DATA_SOURCES = {
    "yahoo_finance": {
        "enabled": True,
        "timeout": 30
    },
    "csv_upload": {
        "enabled": True,
        "max_size_mb": 100
    },
    "excel_upload": {
        "enabled": True,
        "max_size_mb": 100
    }
}
```

## Customization Examples

### Change Color Scheme

```python
# In src/config.py
DARK_BLUE = "#1a3a52"      # Darker blue
LIGHT_BLUE = "#4da6ff"     # Lighter blue
GOLD_COLOR = "#ff9900"     # Orange-gold
```

### Adjust Model Parameters

```python
# For quick testing (fast)
MAX_P = 2
MAX_D = 1
MAX_Q = 2

# For production (thorough)
MAX_P = 8
MAX_D = 3
MAX_Q = 8
```

### Configure Forecast Defaults

```python
DEFAULT_FORECAST_STEPS = 30        # days
DEFAULT_CONFIDENCE_LEVEL = 0.95    # 95% CI
DEFAULT_MA_WINDOW = 20             # Moving average window
```

## Deployment Configuration

### Production Settings

```python
# Streamlit production config
DEBUG_MODE = False
SHOW_SIDEBAR = True
ENABLE_CACHE = True
CACHE_EXPIRE = 24  # hours

# Security
ENABLE_XSRF_PROTECTION = True
SECURE_HEADERS = True
```

### Docker Configuration

```dockerfile
# Dockerfile
FROM python:3.9-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
CMD ["streamlit", "run", "app.py"]
```

### Cloud Deployment (Streamlit Cloud)

`streamlit/secrets.toml`:
```toml
[database]
host = "your_host"
port = 5432
username = "user"
password = "pass"

[api]
timeout = 30
max_retries = 3
```

## Testing Configuration

### pytest Configuration

`pytest.ini`:
```ini
[pytest]
testpaths = tests
python_files = test_*.py
python_classes = Test*
python_functions = test_*
addopts = -v --tb=short
markers =
    slow: slow running tests
    integration: integration tests
```

## Troubleshooting Configuration Issues

### Issue: "Cannot find modules"
**Solution:** Add to `src/config.py`:
```python
import sys
sys.path.insert(0, '/path/to/project')
```

### Issue: "Cache directory not found"
**Solution:** Create directory:
```bash
mkdir -p data/cache
mkdir -p logs
```

### Issue: "Slow performance"
**Solution:** Reduce parameters in `config.py`:
```python
MAX_P = 2  # Was 5
MAX_D = 1  # Was 2
MAX_Q = 2  # Was 5
```

---

**For more configuration options, see the source code documentation.**
