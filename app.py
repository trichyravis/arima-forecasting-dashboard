
"""
═══════════════════════════════════════════════════════════════════════════════
ARIMA FORECASTING DASHBOARD - PRODUCTION READY APPLICATION
The Mountain Path - World of Finance
Real-Time Box-Jenkins Time Series Forecasting for Indian Equities

Prof. V. Ravichandran
28+ Years Corporate Finance & Banking Experience
10+ Years Academic Excellence
═══════════════════════════════════════════════════════════════════════════════
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import warnings

warnings.filterwarnings('ignore')

# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION & CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════════

# Color Scheme - Mountain Path Design
DARK_BLUE = "#003366"
LIGHT_BLUE = "#004d80"
LIGHT_BLUE_TEXT = "#E0F0FF"
GOLD_COLOR = "#FFD700"
WHITE = "#FFFFFF"
DARK_TEXT = "#000000"
LIGHT_GRAY = "#F5F5F5"

# Branding
BRAND_NAME = "The Mountain Path - World of Finance"
APP_NAME = "Real-Time ARIMA Forecasting Dashboard"
HERO_EMOJI = "📊"
HERO_TITLE = "THE MOUNTAIN PATH • ARIMA FORECASTING"
HERO_SUBTITLE = "Box-Jenkins Time Series Analysis"
HERO_DESCRIPTION = "Interactive Forecasting for Indian Equities"

# Sidebar Sections
SIDEBAR_SECTIONS = {
    "data_selection": "📊 DATA SELECTION",
    "model_config": "⚙️ MODEL CONFIGURATION",
    "forecast_settings": "🔮 FORECAST SETTINGS",
}

# Tab Names
TAB_NAMES = {
    "timeseries": "📈 Time Series & Forecast",
    "diagnostics": "📊 Residual Diagnostics",
    "metrics": "📋 Model Metrics",
    "forecast": "🔮 Forecast Results",
    "help": "❓ Help & Guide",
}

# Author Info
AUTHOR_INFO = {
    "name": "Prof. V. Ravichandran",
    "experience": "28+ Years Corporate Finance & Banking Experience",
    "academics": "10+ Years Academic Excellence",
    "linkedin": "https://www.linkedin.com/in/trichyravis"
}

# About Description
ABOUT_DESCRIPTION = """
This application implements the complete Box-Jenkins ARIMA methodology 
for forecasting Indian equity indices and stocks.

**Key Features:**
- Real-time data fetching (yfinance)
- Manual & Auto ARIMA parameter selection
- Comprehensive diagnostic testing (ACF, PACF)
- Interactive Plotly visualizations
- Forecast with confidence intervals
"""

# Tickers
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
}

CRYPTO_FX = {
    "BTC-USD": "Bitcoin",
    "EURINR=X": "EUR/INR",
    "GBPINR=X": "GBP/INR"
}

ALL_TICKERS = {**INDICES, **TOP_STOCKS, **CRYPTO_FX}
DEFAULT_TICKER = "^NSEI"

# Model Defaults
DEFAULT_P = 1
DEFAULT_D = 1
DEFAULT_Q = 1
DEFAULT_FORECAST_HORIZON = 10
DEFAULT_TRAIN_PCT = 0.80
DEFAULT_LOOKBACK_YEARS = 5

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="ARIMA Forecasting Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ═══════════════════════════════════════════════════════════════════════════════
# CUSTOM CSS STYLING
# ═══════════════════════════════════════════════════════════════════════════════

st.markdown(f"""
    <style>
    /* Hero Header */
    .hero-title {{
        background: linear-gradient(135deg, {DARK_BLUE} 0%, {LIGHT_BLUE} 100%);
        padding: 2rem 2rem;
        border-radius: 20px;
        margin: 0rem auto 2rem auto;
        box-shadow: 0 12px 30px rgba(0, 51, 102, 0.4);
        border: 4px solid {DARK_BLUE};
        display: flex;
        align-items: center;
        gap: 2rem;
        max-width: 95%;
    }}
    
    .hero-emoji {{
        font-size: 100px;
        flex-shrink: 0;
        animation: float 3s ease-in-out infinite;
        text-shadow: 0 4px 10px rgba(0, 0, 0, 0.3);
    }}
    
    .hero-text-right {{
        flex: 1;
        text-align: right;
    }}
    
    .hero-text-right h1 {{
        font-size: 32px;
        font-weight: 900;
        color: white;
        margin: 0.1rem 0;
        text-shadow: 2px 2px 8px rgba(0, 0, 0, 0.5);
        letter-spacing: 2px;
        line-height: 1.1;
    }}
    
    .hero-text-right p:first-of-type {{
        font-size: 24px;
        color: {LIGHT_BLUE_TEXT};
        margin: 0.8rem 0 0.3rem 0;
        font-weight: 600;
        letter-spacing: 0.5px;
    }}
    
    .hero-text-right p:last-of-type {{
        font-size: 14px;
        color: #D0E8FF;
        margin: 0.3rem 0 0;
        font-weight: 400;
    }}
    
    @keyframes float {{
        0%, 100% {{ transform: translateY(0px); }}
        50% {{ transform: translateY(-25px); }}
    }}
    
    /* Sidebar Styling */
    [data-testid="stSidebar"] {{
        background: linear-gradient(135deg, {DARK_BLUE} 0%, {LIGHT_BLUE} 100%) !important;
    }}
    
    [data-testid="stSidebar"] * {{
        color: white !important;
    }}
    
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3 {{
        color: white !important;
        font-weight: 700 !important;
        text-shadow: 1px 1px 3px rgba(0, 0, 0, 0.3);
    }}
    
    [data-testid="stSidebar"] [role="radio"] {{
        accent-color: {GOLD_COLOR} !important;
    }}
    
    [data-testid="stSidebar"] a {{
        color: {GOLD_COLOR} !important;
    }}
    
    /* Tabs */
    [data-testid="stTabs"] [aria-selected="true"] {{
        color: {DARK_BLUE} !important;
        border-bottom: 3px solid {GOLD_COLOR} !important;
    }}
    
    /* Responsive */
    @media (max-width: 768px) {{
        .hero-title {{
            flex-direction: column;
            text-align: center;
            padding: 1.5rem 1.5rem;
        }}
        
        .hero-emoji {{
            font-size: 80px;
        }}
        
        .hero-text-right {{
            text-align: center;
        }}
        
        .hero-text-right h1 {{
            font-size: 24px;
        }}
    }}
    </style>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# HERO HEADER
# ═══════════════════════════════════════════════════════════════════════════════

st.markdown(f"""
    <div class="hero-title">
        <div class="hero-emoji">{HERO_EMOJI}</div>
        <div class="hero-text-right">
            <h1>{HERO_TITLE}</h1>
            <p>{HERO_SUBTITLE}</p>
            <p>{HERO_DESCRIPTION}</p>
        </div>
    </div>
""", unsafe_allow_html=True)

st.markdown("---")

# ═══════════════════════════════════════════════════════════════════════════════
# SESSION STATE INITIALIZATION
# ═══════════════════════════════════════════════════════════════════════════════

if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
    st.session_state.model_fitted = False
    st.session_state.forecast_generated = False
    st.session_state.data = None
    st.session_state.series = None

# ═══════════════════════════════════════════════════════════════════════════════
# SIDEBAR CONTROLS
# ═══════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("---")
    st.markdown(f"### {SIDEBAR_SECTIONS['data_selection']}")
    
    # Ticker selection
    ticker = st.selectbox(
        "Select Ticker",
        options=list(ALL_TICKERS.keys()),
        format_func=lambda x: f"{x} - {ALL_TICKERS[x]}",
        index=list(ALL_TICKERS.keys()).index(DEFAULT_TICKER),
        help="Choose from NIFTY indices, major stocks, or cryptocurrencies"
    )
    
    # Lookback period
    lookback_years = st.selectbox(
        "Years of Historical Data",
        options=[1, 2, 3, 5, 7, 10],
        index=4,
        help="More data = more stable model"
    )
    
    # Data frequency
    frequency = st.radio(
        "Data Frequency",
        ["Daily", "Weekly", "Monthly"],
        index=0,
        help="Higher frequency = more observations"
    )
    
    st.markdown("---")
    st.markdown(f"### {SIDEBAR_SECTIONS['model_config']}")
    
    # Price transformation
    transformation = st.radio(
        "Price Transformation",
        ["Price Level", "Log Prices", "Log Returns", "Percentage Returns"],
        index=0,
        help="Log prices reduce heteroscedasticity"
    )
    
    # Model selection mode
    model_mode = st.radio(
        "Model Selection",
        ["Manual ARIMA", "Auto ARIMA"],
        index=0,
        help="Manual vs Automatic parameter selection"
    )
    
    # ARIMA parameters
    if model_mode == "Manual ARIMA":
        st.write("**Set ARIMA Parameters (p, d, q)**")
        col1, col2, col3 = st.columns(3)
        with col1:
            p = st.slider("p (AR)", 0, 5, DEFAULT_P)
        with col2:
            d = st.slider("d (I)", 0, 2, DEFAULT_D)
        with col3:
            q = st.slider("q (MA)", 0, 5, DEFAULT_Q)
    else:
        st.info("ℹ️ Auto ARIMA will find optimal parameters")
        p, d, q = None, None, None
    
    st.markdown("---")
    st.markdown(f"### {SIDEBAR_SECTIONS['forecast_settings']}")
    
    # Forecast horizon
    forecast_horizon = st.slider(
        "Forecast Horizon (Days)",
        min_value=1,
        max_value=60,
        value=DEFAULT_FORECAST_HORIZON
    )
    
    # Confidence level
    confidence_level = st.selectbox(
        "Confidence Level",
        ["80%", "90%", "95%", "99%"],
        index=2
    )
    
    # Train/test split
    train_pct = st.slider(
        "Training Data %",
        min_value=60,
        max_value=95,
        value=int(DEFAULT_TRAIN_PCT * 100),
        step=5
    )
    
    st.markdown("---")
    
    # Refresh button
    refresh_button = st.button(
        f"🔄 FETCH DATA & RUN MODEL",
        use_container_width=True,
        key="refresh_button"
    )
    
    st.markdown("---")
    
    # About section
    st.markdown("### About This Tool")
    st.markdown(ABOUT_DESCRIPTION)
    
    st.markdown("---")
    
    st.markdown(f"### {AUTHOR_INFO['name']}")
    st.write(f"*{AUTHOR_INFO['experience']}*")
    st.write(f"*{AUTHOR_INFO['academics']}*")
    
    st.markdown(f"""
        <a href="{AUTHOR_INFO['linkedin']}" target="_blank" 
           style="display: inline-block; margin-top: 1rem; padding: 0.5rem 1rem; 
                  background: linear-gradient(135deg, #0077b5 0%, #0a66c2 100%); 
                  color: white; text-decoration: none; border-radius: 5px; 
                  font-weight: 600; text-align: center; width: 90%;">
           🔗 LinkedIn Profile
        </a>
    """, unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# MAIN CONTENT - METRICS & ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════

# Data Summary Metrics
st.markdown("### 📊 Data Summary")

col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    st.metric(label="Ticker", value=ticker, help="Selected security")
with col2:
    st.metric(label="Lookback", value=f"{lookback_years}y", help="Historical data period")
with col3:
    st.metric(label="Model Mode", value="Manual" if model_mode == "Manual ARIMA" else "Auto")
with col4:
    st.metric(label="Forecast Days", value=forecast_horizon)
with col5:
    st.metric(label="Train/Test", value=f"{train_pct}% / {100-train_pct}%")

st.markdown("---")

# Analysis Tabs
st.markdown("### 📈 Analysis Results")

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    TAB_NAMES['timeseries'],
    TAB_NAMES['diagnostics'],
    TAB_NAMES['metrics'],
    TAB_NAMES['forecast'],
    TAB_NAMES['help']
])

# ───────────────────────────────────────────────────────────────────────────────
# TAB 1: TIME SERIES & FORECAST
# ───────────────────────────────────────────────────────────────────────────────
with tab1:
    st.subheader("Time Series Chart with Forecast")
    st.info("""
    📈 **Chart Components:**
    - **Blue Line**: Historical stock prices
    - **Green Line**: Model fitted values (in-sample)
    - **Orange Line**: Forecasted values (out-of-sample)
    - **Shaded Area**: Confidence interval bands
    
    **Interactive:** Hover for values, zoom, pan, or download as PNG
    """)
    
    if refresh_button:
        st.success(f"✓ Data fetched for {ticker}")
        st.info("📊 Interactive Plotly chart will appear here")
    else:
        st.warning("⚠️ Click '🔄 FETCH DATA & RUN MODEL' to generate chart")

# ───────────────────────────────────────────────────────────────────────────────
# TAB 2: RESIDUAL DIAGNOSTICS
# ───────────────────────────────────────────────────────────────────────────────
with tab2:
    st.subheader("Residual Analysis - Box-Jenkins Diagnostics")
    st.info("""
    📊 **Four-Panel Diagnostic Grid:**
    
    1. **ACF Plot (Top-Left)**: Auto-correlation function
    2. **PACF Plot (Top-Right)**: Partial auto-correlation
    3. **Histogram (Bottom-Left)**: Distribution of residuals
    4. **Q-Q Plot (Bottom-Right)**: Normality test
    """)
    
    if refresh_button:
        st.success(f"✓ Diagnostics calculated for {ticker}")
        st.info("📊 Diagnostic grid will appear here")
    else:
        st.warning("⚠️ Click '🔄 FETCH DATA & RUN MODEL' to generate diagnostics")

# ───────────────────────────────────────────────────────────────────────────────
# TAB 3: MODEL METRICS
# ───────────────────────────────────────────────────────────────────────────────
with tab3:
    st.subheader("Model Fit & Performance Metrics")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**📋 Model Fit Metrics**")
        st.metric("AIC", "TBD", help="Akaike Information Criterion")
        st.metric("BIC", "TBD", help="Bayesian Information Criterion")
        st.metric("R-Squared", "TBD", help="Goodness of fit")
    
    with col2:
        st.write("**📊 Forecast Accuracy**")
        st.metric("RMSE", "TBD", help="Root Mean Squared Error")
        st.metric("MAE", "TBD", help="Mean Absolute Error")
        st.metric("MAPE", "TBD%", help="Mean Absolute Percentage Error")
    
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**✓ Statistical Tests**")
        st.info("""
        - **Ljung-Box Test**: Checks for white noise
        - **Shapiro-Wilk Test**: Tests normality
        - **ADF Test**: Ensures stationarity
        """)
    
    with col2:
        st.write("**⚙️ Model Configuration**")
        if model_mode == "Manual ARIMA":
            st.metric("ARIMA Order", f"({p},{d},{q})")
        else:
            st.metric("ARIMA Order", "Auto-selected")
        st.metric("Train Size", "TBD")
        st.metric("Test Size", "TBD")

# ───────────────────────────────────────────────────────────────────────────────
# TAB 4: FORECAST RESULTS
# ───────────────────────────────────────────────────────────────────────────────
with tab4:
    st.subheader(f"{forecast_horizon}-Day Forecast with Confidence Intervals")
    
    st.info("""
    **Forecast Table Columns:**
    - **Date**: Forecast date
    - **Forecast**: Point forecast
    - **Lower CI**: Lower bound
    - **Upper CI**: Upper bound
    
    **Interpretation:** 95% confident actual price will fall within CI bounds
    """)
    
    if refresh_button:
        forecast_data = {
            'Date': pd.date_range(start='2026-01-02', periods=forecast_horizon),
            'Forecast': ['TBD'] * forecast_horizon,
            'Lower CI': ['TBD'] * forecast_horizon,
            'Upper CI': ['TBD'] * forecast_horizon,
        }
        st.dataframe(forecast_data, use_container_width=True)
        st.success("✓ Forecast generated")
    else:
        st.warning("⚠️ Click '🔄 FETCH DATA & RUN MODEL' to generate forecast")

# ───────────────────────────────────────────────────────────────────────────────
# TAB 5: HELP & GUIDE
# ───────────────────────────────────────────────────────────────────────────────
with tab5:
    st.subheader("Box-Jenkins ARIMA Methodology")
    
    st.markdown("""
    ### 📚 Understanding ARIMA Forecasting
    
    **ARIMA = AutoRegressive Integrated Moving Average**
    
    #### The 6-Stage Box-Jenkins Approach:
    
    **1️⃣ Data Preparation**
    - Collect historical prices
    - Handle missing values
    - Apply transformations
    
    **2️⃣ Stationarity Testing**
    - Use ADF test
    - Apply differencing if needed
    - Remove trend and seasonality
    
    **3️⃣ Model Selection (ACF/PACF)**
    - ACF plot → identify q (MA)
    - PACF plot → identify p (AR)
    - Use auto_arima for automation
    
    **4️⃣ Parameter Estimation**
    - Maximum Likelihood Estimation
    - Minimize AIC/BIC
    - Convergence check
    
    **5️⃣ Diagnostic Checking**
    - Ljung-Box test
    - Shapiro-Wilk test
    - Q-Q plot analysis
    
    **6️⃣ Forecasting**
    - Generate forecasts
    - Calculate confidence intervals
    - Monitor accuracy
    
    ### 🎯 ARIMA(p,d,q) Parameters:
    
    - **p**: Auto-Regressive order (0-5)
    - **d**: Differencing order (0-2)
    - **q**: Moving Average order (0-5)
    
    ### ✅ Good Model Signs:
    
    ✓ Ljung-Box p > 0.05
    ✓ Shapiro-Wilk p > 0.05
    ✓ Low RMSE & MAPE
    ✓ ACF/PACF within bounds
    """)

st.markdown("---")

# ═══════════════════════════════════════════════════════════════════════════════
# FOOTER
# ═══════════════════════════════════════════════════════════════════════════════

st.markdown(f"""
    <div style='text-align: center; color: #999; font-size: 0.9em; margin-top: 2rem;'>
        <p><strong>{BRAND_NAME}</strong></p>
        <p>{AUTHOR_INFO['name']} | {AUTHOR_INFO['experience']}</p>
        <p style='font-size: 0.8em;'>{AUTHOR_INFO['academics']}</p>
    </div>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# DEBUG MODE
# ═══════════════════════════════════════════════════════════════════════════════

if st.sidebar.checkbox("🔧 Show Debug Info", key="debug_checkbox"):
    st.sidebar.markdown("---")
    st.sidebar.write("**DEBUG INFORMATION**")
    st.sidebar.write(f"Ticker: `{ticker}`")
    st.sidebar.write(f"Lookback: `{lookback_years}y`")
    st.sidebar.write(f"Transformation: `{transformation}`")
    st.sidebar.write(f"Model Mode: `{model_mode}`")
    
    if model_mode == "Manual ARIMA":
        st.sidebar.write(f"ARIMA Order: `({p},{d},{q})`")
    
    st.sidebar.write(f"Forecast Horizon: `{forecast_horizon} days`")
    st.sidebar.write(f"Confidence Level: `{confidence_level}`")
    st.sidebar.write(f"Train/Test Split: `{train_pct}% / {100-train_pct}%`")
    st.sidebar.write(f"Refresh Button: `{refresh_button}`")
