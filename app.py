
"""
═══════════════════════════════════════════════════════════════════════════════
ARIMA FORECASTING DASHBOARD - FULLY FUNCTIONAL APPLICATION
The Mountain Path - World of Finance
Real-Time Box-Jenkins Time Series Forecasting for Indian Equities

Prof. V. Ravichandran
28+ Years Corporate Finance & Banking Experience
10+ Years Academic Excellence

STATUS: ✅ FULLY IMPLEMENTED WITH INTERACTIVE CHARTS & DATA
═══════════════════════════════════════════════════════════════════════════════
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from typing import Tuple, Dict, Optional
import warnings

warnings.filterwarnings('ignore')

# Try imports, with fallbacks
try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except ImportError:
    YFINANCE_AVAILABLE = False
    st.warning("⚠️ yfinance not installed. Install with: pip install yfinance")

try:
    from statsmodels.tsa.arima.model import ARIMA
    from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
    from statsmodels.tsa.stattools import adfuller
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False
    st.warning("⚠️ statsmodels not installed. Install with: pip install statsmodels")

try:
    import pmdarima as pm
    PMDARIMA_AVAILABLE = True
except ImportError:
    PMDARIMA_AVAILABLE = False

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
DEFAULT_LOOKBACK_YEARS = 3

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
    st.session_state.arima_model = None
    st.session_state.forecast_df = None
    st.session_state.fitted_values = None
    st.session_state.residuals = None
    st.session_state.metrics = None

# ═══════════════════════════════════════════════════════════════════════════════
# UTILITY FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

@st.cache_data
def fetch_data(ticker: str, start_date: str, end_date: str, interval: str = "1d") -> Optional[pd.DataFrame]:
    """Fetch data from Yahoo Finance"""
    try:
        if not YFINANCE_AVAILABLE:
            return None
        
        data = yf.download(ticker, start=start_date, end=end_date, interval=interval, progress=False)
        
        if data.empty:
            return None
        
        data.index = pd.to_datetime(data.index)
        if data.index.tz is not None:
            data.index = data.index.tz_localize(None)
        
        return data
    except Exception as e:
        st.error(f"Error fetching data: {str(e)}")
        return None

def prepare_series(df: pd.DataFrame, column: str = "Close") -> Optional[pd.Series]:
    """Prepare time series for ARIMA"""
    try:
        series = df[column].copy()
        series = series.dropna()
        return series
    except Exception as e:
        st.error(f"Error preparing series: {str(e)}")
        return None

def fit_arima_model(series: pd.Series, order: Tuple[int, int, int]) -> Dict:
    """Fit ARIMA model"""
    try:
        if not STATSMODELS_AVAILABLE:
            return None
        
        model = ARIMA(series, order=order)
        fitted_model = model.fit()
        
        return {
            'model': fitted_model,
            'aic': fitted_model.aic,
            'bic': fitted_model.bic,
            'rsquared': fitted_model.rsquared
        }
    except Exception as e:
        st.error(f"Error fitting ARIMA model: {str(e)}")
        return None

def generate_forecast(fitted_model, steps: int, alpha: float = 0.05) -> Optional[pd.DataFrame]:
    """Generate ARIMA forecast"""
    try:
        forecast_result = fitted_model.get_forecast(steps=steps)
        forecast_df = forecast_result.conf_int(alpha=alpha)
        forecast_df['forecast'] = forecast_result.predicted_mean
        forecast_df.columns = ['lower_ci', 'upper_ci', 'forecast']
        forecast_df = forecast_df[['forecast', 'lower_ci', 'upper_ci']]
        
        return forecast_df
    except Exception as e:
        st.error(f"Error generating forecast: {str(e)}")
        return None

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
        index=2,
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

# ═══════════════════════════════════════════════════════════════════════════════
# DATA LOADING & MODEL EXECUTION
# ═══════════════════════════════════════════════════════════════════════════════

if refresh_button:
    with st.spinner(f"⏳ Fetching data for {ticker}..."):
        # Calculate date range
        end_date = datetime.now().strftime("%Y-%m-%d")
        start_date = (datetime.now() - timedelta(days=lookback_years*365)).strftime("%Y-%m-%d")
        
        # Map frequency
        interval_map = {"Daily": "1d", "Weekly": "1wk", "Monthly": "1mo"}
        interval = interval_map.get(frequency, "1d")
        
        # Fetch data
        data = fetch_data(ticker, start_date, end_date, interval)
        
        if data is not None and not data.empty:
            st.session_state.data = data
            st.session_state.series = prepare_series(data, "Close")
            st.session_state.data_loaded = True
            st.success(f"✅ Data fetched for {ticker}: {len(st.session_state.series)} observations")
        else:
            st.error("❌ Could not fetch data")
            st.session_state.data_loaded = False

# ═══════════════════════════════════════════════════════════════════════════════
# MAIN CONTENT - METRICS & ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════

if st.session_state.data_loaded:
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
        
        # Fit model if not already done
        if refresh_button and not st.session_state.model_fitted:
            with st.spinner("⏳ Fitting ARIMA model..."):
                if model_mode == "Manual ARIMA":
                    arima_result = fit_arima_model(st.session_state.series, (p, d, q))
                    if arima_result:
                        st.session_state.arima_model = arima_result['model']
                        st.session_state.model_fitted = True
                        st.success(f"✅ ARIMA({p},{d},{q}) fitted successfully")
                else:
                    if PMDARIMA_AVAILABLE:
                        with st.spinner("Finding optimal parameters..."):
                            try:
                                auto_model = pm.auto_arima(st.session_state.series, seasonal=False, stepwise=True)
                                st.session_state.arima_model = auto_model
                                st.session_state.model_fitted = True
                                st.success(f"✅ Auto ARIMA selected: {auto_model.order}")
                            except:
                                st.warning("Auto ARIMA failed, using manual (1,1,1)")
                                arima_result = fit_arima_model(st.session_state.series, (1, 1, 1))
                                if arima_result:
                                    st.session_state.arima_model = arima_result['model']
                                    st.session_state.model_fitted = True
        
        # Generate forecast
        if st.session_state.model_fitted and not st.session_state.forecast_generated and refresh_button:
            with st.spinner("⏳ Generating forecast..."):
                alpha_map = {"80%": 0.20, "90%": 0.10, "95%": 0.05, "99%": 0.01}
                alpha = alpha_map.get(confidence_level, 0.05)
                
                forecast = generate_forecast(st.session_state.arima_model, forecast_horizon, alpha)
                if forecast is not None:
                    st.session_state.forecast_df = forecast
                    st.session_state.forecast_generated = True
                    st.session_state.fitted_values = st.session_state.arima_model.fittedvalues
                    st.session_state.residuals = st.session_state.arima_model.resid
                    st.success("✅ Forecast generated")
        
        # Create interactive chart
        if st.session_state.forecast_generated:
            fig = go.Figure()
            
            # Historical data
            fig.add_trace(go.Scatter(
                x=st.session_state.series.index,
                y=st.session_state.series.values,
                mode='lines',
                name='Historical Price',
                line=dict(color=DARK_BLUE, width=2),
                hovertemplate='<b>Date</b>: %{x|%Y-%m-%d}<br><b>Price</b>: ₹%{y:.2f}<extra></extra>'
            ))
            
            # Fitted values
            fig.add_trace(go.Scatter(
                x=st.session_state.fitted_values.index,
                y=st.session_state.fitted_values.values,
                mode='lines',
                name='Fitted Values',
                line=dict(color='green', width=2, dash='dash'),
                hovertemplate='<b>Date</b>: %{x|%Y-%m-%d}<br><b>Fitted</b>: ₹%{y:.2f}<extra></extra>'
            ))
            
            # Forecast
            forecast_df = st.session_state.forecast_df
            fig.add_trace(go.Scatter(
                x=forecast_df.index,
                y=forecast_df['forecast'].values,
                mode='lines',
                name='Forecast',
                line=dict(color=GOLD_COLOR, width=2),
                hovertemplate='<b>Date</b>: %{x|%Y-%m-%d}<br><b>Forecast</b>: ₹%{y:.2f}<extra></extra>'
            ))
            
            # Confidence interval
            fig.add_trace(go.Scatter(
                x=forecast_df.index,
                y=forecast_df['upper_ci'].values,
                fill=None,
                mode='lines',
                line_color='rgba(0,0,0,0)',
                showlegend=False,
                hoverinfo='skip'
            ))
            
            fig.add_trace(go.Scatter(
                x=forecast_df.index,
                y=forecast_df['lower_ci'].values,
                fill='tonexty',
                mode='lines',
                line_color='rgba(0,0,0,0)',
                name='95% Confidence Interval',
                fillcolor=f'rgba(255,215,0,0.2)',
                hovertemplate='<b>CI Range</b><extra></extra>'
            ))
            
            # Layout
            fig.update_layout(
                title=f'<b>{ALL_TICKERS[ticker]} - ARIMA Forecast</b>',
                xaxis_title='Date',
                yaxis_title='Price (₹)',
                template='plotly_white',
                hovermode='x unified',
                height=600,
                font=dict(family="Arial", size=12)
            )
            
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("📊 Click button to generate interactive chart")
    
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
        
        if st.session_state.model_fitted and st.session_state.residuals is not None:
            try:
                fig, axes = plt.subplots(2, 2, figsize=(14, 10))
                
                # ACF
                plot_acf(st.session_state.residuals, lags=40, ax=axes[0, 0])
                axes[0, 0].set_title('ACF Plot')
                
                # PACF
                plot_pacf(st.session_state.residuals, lags=40, ax=axes[0, 1])
                axes[0, 1].set_title('PACF Plot')
                
                # Histogram
                axes[1, 0].hist(st.session_state.residuals, bins=30, edgecolor='black')
                axes[1, 0].set_title('Histogram of Residuals')
                axes[1, 0].set_xlabel('Residuals')
                
                # Q-Q Plot
                from scipy import stats
                stats.probplot(st.session_state.residuals, dist="norm", plot=axes[1, 1])
                axes[1, 1].set_title('Q-Q Plot')
                
                plt.tight_layout()
                st.pyplot(fig)
                st.success("✅ Diagnostics calculated")
            except Exception as e:
                st.error(f"Error generating diagnostics: {str(e)}")
        else:
            st.warning("⚠️ Click '🔄 FETCH DATA & RUN MODEL' to generate diagnostics")
    
    # ───────────────────────────────────────────────────────────────────────────────
    # TAB 3: MODEL METRICS
    # ───────────────────────────────────────────────────────────────────────────────
    with tab3:
        st.subheader("Model Fit & Performance Metrics")
        
        if st.session_state.model_fitted:
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**📋 Model Fit Metrics**")
                st.metric("AIC", f"{st.session_state.arima_model.aic:.2f}", help="Akaike Information Criterion")
                st.metric("BIC", f"{st.session_state.arima_model.bic:.2f}", help="Bayesian Information Criterion")
                st.metric("R-Squared", f"{st.session_state.arima_model.rsquared:.4f}", help="Goodness of fit")
            
            with col2:
                st.write("**📊 Forecast Accuracy**")
                residuals = st.session_state.residuals
                rmse = np.sqrt(np.mean(residuals**2))
                mae = np.mean(np.abs(residuals))
                mape = np.mean(np.abs(residuals / st.session_state.series.mean())) * 100
                
                st.metric("RMSE", f"₹{rmse:.2f}", help="Root Mean Squared Error")
                st.metric("MAE", f"₹{mae:.2f}", help="Mean Absolute Error")
                st.metric("MAPE", f"{mape:.2f}%", help="Mean Absolute Percentage Error")
        else:
            st.warning("⚠️ Click '🔄 FETCH DATA & RUN MODEL' to calculate metrics")
    
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
        """)
        
        if st.session_state.forecast_generated:
            forecast_display = st.session_state.forecast_df.copy()
            forecast_display.index = forecast_display.index.strftime('%Y-%m-%d')
            forecast_display.columns = ['Forecast (₹)', 'Lower CI (₹)', 'Upper CI (₹)']
            forecast_display = forecast_display.round(2)
            
            st.dataframe(forecast_display, use_container_width=True)
            
            # Download button
            csv = forecast_display.to_csv()
            st.download_button(
                label="📥 Download Forecast as CSV",
                data=csv,
                file_name=f"{ticker}_forecast_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv"
            )
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
        - Collect historical daily prices
        - Remove outliers and handle missing values
        - Apply log transformation to stabilize variance
        
        **2️⃣ Stationarity Testing**
        - Use ADF (Augmented Dickey-Fuller) test
        - Non-stationary series → apply differencing (d)
        - Goal: Remove trend and seasonality
        
        **3️⃣ Model Selection (ACF/PACF)**
        - ACF plot → identify q (MA order)
        - PACF plot → identify p (AR order)
        - Use auto_arima for automatic selection
        
        **4️⃣ Parameter Estimation**
        - Maximum Likelihood Estimation (MLE)
        - Minimize AIC/BIC criteria
        - Convergence = optimal parameters found
        
        **5️⃣ Diagnostic Checking**
        - Ljung-Box test: Are residuals white noise?
        - Shapiro-Wilk: Are residuals normally distributed?
        - Q-Q plot: Visual normality check
        
        **6️⃣ Forecasting & Monitoring**
        - Generate point forecasts + confidence intervals
        - Track actual vs. predicted
        - Retrain if forecast errors exceed thresholds
        
        ### 🎯 ARIMA(p,d,q) Parameters:
        
        - **p (AR order)**: # previous values used for prediction (0-5)
        - **d (Differencing)**: # times to difference for stationarity (0-2)
        - **q (MA order)**: # previous errors used for prediction (0-5)
        
        **Examples:**
        - ARIMA(1,1,1): Basic trend + mean reversion
        - ARIMA(2,1,2): More complex patterns
        - ARIMA(0,1,0): Random walk (naive forecast)
        
        ### ✅ Good Model Signs:
        
        ✓ Ljung-Box p-value > 0.05 (white noise residuals)
        ✓ Shapiro-Wilk p-value > 0.05 (normal distribution)
        ✓ Low RMSE & MAPE on test set
        ✓ ACF/PACF within confidence bands
        ✓ No significant spikes in residuals
        
        ### ⚠️ When to Reconsider:
        
        ⚠️ Ljung-Box p < 0.05 (structure in residuals)
        ⚠️ High MAPE (>5%) on test set
        ⚠️ Try SARIMA for seasonal patterns
        ⚠️ Consider ARIMAX with exogenous variables
        """)

else:
    st.warning("⏳ Click '🔄 FETCH DATA & RUN MODEL' button in sidebar to start")

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
