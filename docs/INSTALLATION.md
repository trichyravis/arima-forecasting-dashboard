# Installation Guide - ARIMA Forecasting Dashboard

## Prerequisites

- Python 3.8 or higher
- pip (Python package manager)
- Git (for cloning the repository)
- 2GB RAM minimum
- macOS, Windows, or Linux

## Quick Start Installation

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/arima-forecasting-dashboard.git
cd arima-forecasting-dashboard
```

### 2. Create Virtual Environment

**macOS/Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

**Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Create Data Directories

```bash
mkdir -p data/cache
mkdir -p data/results
mkdir -p logs
```

### 5. Run the Application

```bash
streamlit run app.py
```

The dashboard opens at `http://localhost:8501`

## Detailed Installation

### Step 1: System Requirements Check

```bash
python --version        # Should be 3.8+
pip --version          # Should be 20.0+
```

### Step 2: Virtual Environment Setup

**Why use virtual environment?**
- Isolates project dependencies
- Prevents version conflicts
- Allows multiple projects

### Step 3: Install Python Packages

Core packages installed:
- **pandas** - Data manipulation
- **numpy** - Numerical computing
- **statsmodels** - Statistical models (ARIMA)
- **scikit-learn** - Machine learning utilities
- **plotly** - Interactive visualizations
- **streamlit** - Web dashboard framework
- **yfinance** - Financial data fetching

### Step 4: Configure Application

1. Create `.streamlit/config.toml`:
```toml
[theme]
primaryColor = "#003366"
backgroundColor = "#FFFFFF"
secondaryBackgroundColor = "#F0F0F0"
textColor = "#003366"
```

2. Set environment variables (optional):
```bash
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
```

## Verification

Run verification script:

```bash
python -c "
import pandas as pd
import numpy as np
import streamlit
import plotly
from src.data.loader import DataLoader
from src.models.arima import ARIMAModel
print('✅ All imports successful!')
"
```

## Troubleshooting Installation

**Problem: ModuleNotFoundError**
```bash
# Solution: Reinstall requirements
pip install --upgrade pip
pip install -r requirements.txt --force-reinstall
```

**Problem: Streamlit not found**
```bash
# Solution: Install specifically
pip install streamlit
```

**Problem: Permission denied on macOS**
```bash
# Solution: Use sudo (careful!)
sudo pip install -r requirements.txt
```

## Configuration

After installation, configure the application:

1. Edit `src/config.py` with your preferences
2. Create `.streamlit/config.toml` for Streamlit settings
3. Set up cache directory for faster data loading

## Running Tests

```bash
# Install testing packages
pip install pytest pytest-cov

# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html
```

## Next Steps

1. Read [USAGE.md](USAGE.md) to learn how to use the dashboard
2. Check [CONFIGURATION.md](CONFIGURATION.md) for advanced setup
3. Review [design/DESIGN_SYSTEM.md](design/DESIGN_SYSTEM.md) for customization

---

**Installation Complete!** 🎉

Run `streamlit run app.py` to start the dashboard.
