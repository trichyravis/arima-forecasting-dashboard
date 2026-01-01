"""
═══════════════════════════════════════════════════════════════════════════════
ARIMA FORECASTING DASHBOARD - CONFIGURATION MODULE
The Mountain Path - World of Finance
Prof. V. Ravichandran | 28+ Years Finance | 10+ Years Academia
═══════════════════════════════════════════════════════════════════════════════
"""

import os
from pathlib import Path
from typing import Dict, List, Tuple

# ============================================================================
# PROJECT PATHS
# ============================================================================

PROJECT_ROOT = Path(__file__).parent.parent
DATA_CACHE_DIR = PROJECT_ROOT / "data" / "cache"
DATA_RESULTS_DIR = PROJECT_ROOT / "data" / "results"
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"
TESTS_DIR = PROJECT_ROOT / "tests"
LOGS_DIR = PROJECT_ROOT / "logs"

# Create directories if they don't exist
for dir_path in [DATA_CACHE_DIR, DATA_RESULTS_DIR, NOTEBOOKS_DIR, TESTS_DIR, LOGS_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# ============================================================================
# DESIGN & BRANDING (THE MOUNTAIN PATH)
# ============================================================================

# ─── PRIMARY COLORS ───
DARK_BLUE = "#003366"              # Primary brand color RGB(0, 51, 102)
LIGHT_BLUE = "#004d80"             # Lighter shade for gradients
LIGHT_BLUE_TEXT = "#E0F0FF"        # Light text for dark backgrounds
GOLD_COLOR = "#FFD700"             # Accent color RGB(255, 215, 0)
WHITE = "#FFFFFF"                  # Primary text
DARK_TEXT = "#000000"              # Dark text on light backgrounds
LIGHT_GRAY = "#F5F5F5"             # Light background

# ─── BRAND NAMES ───
BRAND_NAME = "The Mountain Path - World of Finance"
APP_NAME = "Real-Time ARIMA Forecasting Dashboard"
APP_ICON = "📊"

# ─── HERO HEADER ───
HERO_EMOJI = "📊"
HERO_TITLE = "THE MOUNTAIN PATH • ARIMA FORECASTING"
HERO_SUBTITLE = "Box-Jenkins Time Series Analysis"
HERO_DESCRIPTION = "Interactive Forecasting for Indian Equities"

# ─── SIDEBAR SECTIONS ───
SIDEBAR_SECTIONS = {
    "data_selection": "📊 DATA SELECTION",
    "model_config": "⚙️ MODEL CONFIGURATION",
    "forecast_settings": "🔮 FORECAST SETTINGS",
}

# ─── TAB NAMES ───
TAB_NAMES = {
    "timeseries": "📈 Time Series & Forecast",
    "diagnostics": "📊 Residual Diagnostics",
    "metrics": "📋 Model Metrics",
    "forecast": "🔮 Forecast Results",
    "help": "❓ Help & Guide",
}

# ─── ABOUT SECTION ───
ABOUT_DESCRIPTION = """
This application implements the complete Box-Jenkins ARIMA methodology 
for forecasting Indian equity indices and stocks.

**Key Features:**
- Real-time data fetching (yfinance, NSEpy)
- Manual & Auto ARIMA parameter selection
- Comprehensive diagnostic testing (ACF, PACF, Ljung-Box)
- Interactive Plotly visualizations
- Forecast with confidence intervals
"""

AUTHOR_INFO = {
    "name": "Prof. V. Ravichandran",
    "experience": "28+ Years Corporate Finance & Banking Experience",
    "academics": "10+ Years Academic Excellence",
    "linkedin": "https://www.linkedin.com/in/trichyravis"
}

# ============================================================================
# STYLING & UI CONFIGURATION
# ============================================================================

PAGE_LAYOUT = "wide"
PAGE_ICON = "📊"
PAGE_TITLE = "ARIMA Forecasting Dashboard"

# Sidebar width
SIDEBAR_WIDTH = 300

# Plot dimensions
PLOT_HEIGHT = 500
PLOT_WIDTH = 1000

# Decimal formatting
DECIMAL_PLACES = 4
PERCENTAGE_DECIMAL_PLACES = 2

# ============================================================================
# DATA CONFIGURATION
# ============================================================================

# ─── TICKER LISTS ───
INDICES = {
    "^NSEI": "NIFTY 50",
    "^NSEBANK": "BANKNIFTY",
    "^NIFTYNXT50": "NIFTY NEXT 50"
}

TOP_STOCKS = {
    "TCS.NS": "Tata Consultancy Services",
    "INFY.NS": "Infosys",
    "HDFC.NS": "HDFC Bank",
    "RELIANCE.NS": "Reliance Industries",
    "WIPRO.NS": "Wipro",
    "HCL.NS": "HCL Technologies",
    "BAJAJFINSV.NS": "Bajaj Financials",
    "MARUTI.NS": "Maruti Suzuki",
    "LTIM.NS": "LTIMindtree",
    "ASIANPAINT.NS": "Asian Paints"
}

CRYPTO_FX = {
    "BTC-USD": "Bitcoin",
    "EURINR=X": "EUR/INR",
    "GBPINR=X": "GBP/INR"
}

# Combine all tickers
ALL_TICKERS = {**INDICES, **TOP_STOCKS, **CRYPTO_FX}

# Default ticker
DEFAULT_TICKER = "^NSEI"  # NIFTY 50

# ─── DATA FETCHING ───
DATA_SOURCES = ["yfinance", "nsetools"]
CACHE_TTL_HOURS = 24

# ─── DATA VALIDATION ───
MIN_OBSERVATIONS = 250           # Minimum 1 year of daily data
DEFAULT_LOOKBACK_YEARS = 5
MIN_LOOKBACK_DAYS = 250
MAX_LOOKBACK_YEARS = 10
DATA_FREQUENCY = "D"             # Daily
TRADING_DAYS_PER_YEAR = 252

# ============================================================================
# ARIMA CONFIGURATION
# ============================================================================

# ─── DEFAULT PARAMETERS ───
DEFAULT_P = 1
DEFAULT_D = 1
DEFAULT_Q = 1

# ─── PARAMETER RANGES ───
P_RANGE = (0, 5)
D_RANGE = (0, 2)
Q_RANGE = (0, 5)

# ─── AUTO ARIMA SETTINGS ───
AUTO_ARIMA_MAX_P = 5
AUTO_ARIMA_MAX_D = 2
AUTO_ARIMA_MAX_Q = 5
AUTO_ARIMA_SEASONAL = False
AUTO_ARIMA_STEPWISE = True

# ============================================================================
# FORECAST CONFIGURATION
# ============================================================================

DEFAULT_FORECAST_HORIZON = 10      # days
MIN_FORECAST_HORIZON = 1
MAX_FORECAST_HORIZON = 60

# Confidence levels
DEFAULT_CONFIDENCE_LEVEL = 0.95    # 95%
CONFIDENCE_LEVELS = [0.80, 0.90, 0.95, 0.99]

# ============================================================================
# MODEL CONFIGURATION
# ============================================================================

# ─── TRAIN/TEST SPLIT ───
DEFAULT_TRAIN_PCT = 0.80
MIN_TRAIN_PCT = 0.60
MAX_TRAIN_PCT = 0.95

# ─── PRICE TRANSFORMATIONS ───
TRANSFORMATIONS = {
    "price": "Price Level",
    "log": "Log Prices",
    "log_returns": "Log Returns",
    "pct_returns": "Percentage Returns"
}
DEFAULT_TRANSFORMATION = "price"

# ============================================================================
# DIAGNOSTIC CONFIGURATION
# ============================================================================

ACF_LAGS = 40
PACF_LAGS = 40
LJUNG_BOX_LAGS = 10
LJUNG_BOX_ALPHA = 0.05             # Significance level for Ljung-Box test

# ============================================================================
# ENVIRONMENT
# ============================================================================

IS_DEVELOPMENT = os.getenv("ENVIRONMENT", "development").lower() == "development"
IS_PRODUCTION = not IS_DEVELOPMENT

LOG_LEVEL = "INFO"

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_ticker_display_name(ticker: str) -> str:
    """Get display name for ticker"""
    return ALL_TICKERS.get(ticker, ticker)

def get_all_ticker_options() -> Dict[str, str]:
    """Get all available ticker options"""
    return ALL_TICKERS

def get_ticker_by_group() -> Dict[str, Dict[str, str]]:
    """Get tickers grouped by category"""
    return {
        "Indices": INDICES,
        "Top Stocks": TOP_STOCKS,
        "Crypto & FX": CRYPTO_FX
    }

# ============================================================================
# MODULE INFO
# ============================================================================

if __name__ == "__main__":
    print(f"Project Root: {PROJECT_ROOT}")
    print(f"Data Cache: {DATA_CACHE_DIR}")
    print(f"Total Tickers: {len(ALL_TICKERS)}")
    print(f"Brand: {BRAND_NAME}")
    print(f"Primary Color: {DARK_BLUE}")
    print(f"Accent Color: {GOLD_COLOR}")
