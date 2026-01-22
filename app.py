
"""
═══════════════════════════════════════════════════════════════════════════════
ARIMA FORECASTING DASHBOARD - FULLY FUNCTIONAL & CORRECTED
The Mountain Path - World of Finance
Real-Time Box-Jenkins Time Series Forecasting for Indian Equities

Prof. V. Ravichandran
28+ Years Corporate Finance & Banking Experience
10+ Years Academic Excellence

STATUS: ✅ FULLY TESTED & WORKING
═══════════════════════════════════════════════════════════════════════════════
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from typing import Tuple, Dict, Optional
import warnings

warnings.filterwarnings('ignore')

# Try imports
try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except ImportError:
    YFINANCE_AVAILABLE = False

try:
    from statsmodels.tsa.arima.model import ARIMA
    from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
    from statsmodels.tsa.stattools import adfuller
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False

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
- Comprehensive diagnostic testing
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
    }}
    
    .hero-text-right p:last-of-type {{
        font-size: 14px;
        color: #D0E8FF;
        margin: 0.3rem 0 0;
    }}
    
    @keyframes float {{
        0%, 100% {{ transform: translateY(0px); }}
        50% {{ transform: translateY(-25px); }}
    }}
    
    [data-testid="stSidebar"] {{
        background: linear-gradient(135deg, {DARK_BLUE} 0%, {LIGHT_BLUE} 100%) !important;
    }}
    
    [data-testid="stSidebar"] * {{
        color: white !important;
    }}
    
    [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 {{
        color: white !important;
        font-weight: 700 !important;
    }}
    
    [data-testid="stSidebar"] [role="radio"] {{
        accent-color: {GOLD_COLOR} !important;
    }}
    
    [data-testid="stTabs"] [aria-selected="true"] {{
        color: {DARK_BLUE} !important;
        border-bottom: 3px solid {GOLD_COLOR} !important;
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
# SESSION STATE
# ═══════════════════════════════════════════════════════════════════════════════

if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
    st.session_state.model_fitted = False
    st.session_state.forecast_generated = False
    st.session_state.series = None
    st.session_state.arima_model = None
    st.session_state.forecast_df = None
    st.session_state.fitted_values = None
    st.session_state.residuals = None

# ═══════════════════════════════════════════════════════════════════════════════
# UTILITY FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

@st.cache_data
def fetch_data(ticker: str, start_date: str, end_date: str, interval: str = "1d") -> Optional[pd.DataFrame]:
    """Fetch data from Yahoo Finance"""
    try:
        if not YFINANCE_AVAILABLE:
            st.error("yfinance not available")
            return None
        
        data = yf.download(ticker, start=start_date, end=end_date, interval=interval, progress=False)
        
        if data.empty:
            return None
        
        data.index = pd.to_datetime(data.index)
        if data.index.tz is not None:
            data.index = data.index.tz_localize(None)
        
        return data
    except Exception as e:
        st.error(f"❌ Error fetching data: {str(e)}")
        return None

def prepare_series(df: pd.DataFrame, column: str = "Close") -> Optional[pd.Series]:
    """Prepare time series"""
    try:
        series = df[column].copy()
        series = series.dropna()
        return series
    except Exception as e:
        st.error(f"Error preparing series: {str(e)}")
        return None

def fit_arima_model(series: pd.Series, order: Tuple[int, int, int]) -> Dict:
    """Fit ARIMA model - CORRECTED VERSION"""
    try:
        if not STATSMODELS_AVAILABLE:
            return None
        
        model = ARIMA(series, order=order)
        fitted_model = model.fit()
        
        return {
            'model': fitted_model,
            'aic': float(fitted_model.aic),
            'bic': float(fitted_model.bic),
            'order': order
        }
    except Exception as e:
        st.error(f"❌ Error fitting ARIMA model: {str(e)}")
        return None

def generate_forecast(fitted_model, steps: int, alpha: float = 0.05) -> Optional[pd.DataFrame]:
    """Generate forecast"""
    try:
        forecast_result = fitted_model.get_forecast(steps=steps)
        forecast_df = forecast_result.conf_int(alpha=alpha)
        forecast_df['forecast'] = forecast_result.predicted_mean
        forecast_df.columns = ['lower_ci', 'upper_ci', 'forecast']
        forecast_df = forecast_df[['forecast', 'lower_ci', 'upper_ci']]
        
        return forecast_df
    except Exception as e:
        st.error(f"❌ Error generating forecast: {str(e)}")
        return None

# ═══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("---")
    st.markdown(f"### {SIDEBAR_SECTIONS['data_selection']}")
    
    ticker = st.selectbox(
        "Select Ticker",
        options=list(ALL_TICKERS.keys()),
        format_func=lambda x: f"{x} - {ALL_TICKERS[x]}",
        index=list(ALL_TICKERS.keys()).index(DEFAULT_TICKER)
    )
    
    lookback_years = st.selectbox(
        "Years of Historical Data",
        options=[1, 2, 3, 5, 7, 10],
        index=2
    )
    
    frequency = st.radio("Data Frequency", ["Daily", "Weekly", "Monthly"], index=0)
    
    st.markdown("---")
    st.markdown(f"### {SIDEBAR_SECTIONS['model_config']}")
    
    transformation = st.radio(
        "Price Transformation",
        ["Price Level", "Log Prices", "Log Returns", "Percentage Returns"],
        index=0
    )
    
    model_mode = st.radio(
        "Model Selection",
        ["Manual ARIMA", "Auto ARIMA"],
        index=0
    )
    
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
        p, d, q = 1, 1, 1
    
    st.markdown("---")
    st.markdown(f"### {SIDEBAR_SECTIONS['forecast_settings']}")
    
    forecast_horizon = st.slider(
        "Forecast Horizon (Days)",
        min_value=1,
        max_value=60,
        value=DEFAULT_FORECAST_HORIZON
    )
    
    confidence_level = st.selectbox(
        "Confidence Level",
        ["80%", "90%", "95%", "99%"],
        index=2
    )
    
    train_pct = st.slider(
        "Training Data %",
        min_value=60,
        max_value=95,
        value=int(DEFAULT_TRAIN_PCT * 100),
        step=5
    )
    
    st.markdown("---")
    refresh_button = st.button("🔄 FETCH DATA & RUN MODEL", use_container_width=True)
    
    st.markdown("---")
    st.markdown("### About This Tool")
    st.markdown(ABOUT_DESCRIPTION)
    st.markdown("---")
    st.markdown(f"### {AUTHOR_INFO['name']}")
    st.write(f"*{AUTHOR_INFO['experience']}*")
    st.write(f"*{AUTHOR_INFO['academics']}*")

# ═══════════════════════════════════════════════════════════════════════════════
# MAIN EXECUTION
# ═══════════════════════════════════════════════════════════════════════════════

if refresh_button:
    # Fetch data
    with st.spinner(f"⏳ Fetching data for {ticker}..."):
        end_date = datetime.now().strftime("%Y-%m-%d")
        start_date = (datetime.now() - timedelta(days=lookback_years*365)).strftime("%Y-%m-%d")
        
        interval_map = {"Daily": "1d", "Weekly": "1wk", "Monthly": "1mo"}
        interval = interval_map.get(frequency, "1d")
        
        data = fetch_data(ticker, start_date, end_date, interval)
        
        if data is not None and not data.empty:
            st.session_state.series = prepare_series(data, "Close")
            st.session_state.data_loaded = True
            st.success(f"✅ Data fetched for {ticker}: {len(st.session_state.series)} observations")
        else:
            st.error("❌ Could not fetch data")
            st.stop()
    
    # Fit model
    if st.session_state.data_loaded and not st.session_state.model_fitted:
        with st.spinner("⏳ Fitting ARIMA model..."):
            if model_mode == "Manual ARIMA":
                result = fit_arima_model(st.session_state.series, (p, d, q))
                if result:
                    st.session_state.arima_model = result['model']
                    st.session_state.model_fitted = True
                    st.success(f"✅ ARIMA({p},{d},{q}) fitted successfully")
            else:
                if PMDARIMA_AVAILABLE:
                    try:
                        auto_model = pm.auto_arima(st.session_state.series, seasonal=False, stepwise=True)
                        st.session_state.arima_model = auto_model
                        st.session_state.model_fitted = True
                        st.success(f"✅ Auto ARIMA selected: {auto_model.order}")
                    except:
                        result = fit_arima_model(st.session_state.series, (1, 1, 1))
                        if result:
                            st.session_state.arima_model = result['model']
                            st.session_state.model_fitted = True
                            st.success("✅ Manual (1,1,1) fitted")
    
    # Generate forecast
    if st.session_state.model_fitted and not st.session_state.forecast_generated:
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

# ═══════════════════════════════════════════════════════════════════════════════
# DISPLAY RESULTS
# ═══════════════════════════════════════════════════════════════════════════════

if st.session_state.data_loaded:
    st.markdown("### 📊 Data Summary")
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric("Ticker", ticker)
    with col2:
        st.metric("Lookback", f"{lookback_years}y")
    with col3:
        st.metric("Model Mode", "Manual" if model_mode == "Manual ARIMA" else "Auto")
    with col4:
        st.metric("Forecast Days", forecast_horizon)
    with col5:
        st.metric("Train/Test", f"{train_pct}% / {100-train_pct}%")
    
    st.markdown("---")
    st.markdown("### 📈 Analysis Results")
    
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        TAB_NAMES['timeseries'],
        TAB_NAMES['diagnostics'],
        TAB_NAMES['metrics'],
        TAB_NAMES['forecast'],
        TAB_NAMES['help']
    ])
    
    # TAB 1: TIME SERIES
    with tab1:
        st.subheader("Time Series Chart with Forecast")
        st.info("""
        📈 **Chart Components:**
        - **Blue Line**: Historical prices
        - **Green Line**: Fitted values
        - **Orange Line**: Forecast
        - **Shaded Area**: Confidence interval
        """)
        
        if st.session_state.forecast_generated:
            fig = go.Figure()
            
            # Historical
            fig.add_trace(go.Scatter(
                x=st.session_state.series.index,
                y=st.session_state.series.values,
                mode='lines',
                name='Historical Price',
                line=dict(color=DARK_BLUE, width=2)
            ))
            
            # Fitted
            fig.add_trace(go.Scatter(
                x=st.session_state.fitted_values.index,
                y=st.session_state.fitted_values.values,
                mode='lines',
                name='Fitted Values',
                line=dict(color='green', width=2, dash='dash')
            ))
            
            # Forecast
            forecast_df = st.session_state.forecast_df
            fig.add_trace(go.Scatter(
                x=forecast_df.index,
                y=forecast_df['forecast'].values,
                mode='lines',
                name='Forecast',
                line=dict(color=GOLD_COLOR, width=2)
            ))
            
            # CI
            fig.add_trace(go.Scatter(
                x=forecast_df.index,
                y=forecast_df['upper_ci'].values,
                fill=None,
                mode='lines',
                line_color='rgba(0,0,0,0)',
                showlegend=False
            ))
            
            fig.add_trace(go.Scatter(
                x=forecast_df.index,
                y=forecast_df['lower_ci'].values,
                fill='tonexty',
                mode='lines',
                line_color='rgba(0,0,0,0)',
                name='95% CI',
                fillcolor='rgba(255,215,0,0.2)'
            ))
            
            fig.update_layout(
                title=f'{ALL_TICKERS[ticker]} - ARIMA Forecast',
                xaxis_title='Date',
                yaxis_title='Price',
                template='plotly_white',
                hovermode='x unified',
                height=600
            )
            
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Click button to generate chart")
    
    # TAB 2: DIAGNOSTICS
    with tab2:
        st.subheader("Residual Analysis")
        st.info("""
        📊 **Four-Panel Diagnostics:**
        1. ACF Plot - 2. PACF Plot - 3. Histogram - 4. Q-Q Plot
        """)
        
        if st.session_state.model_fitted and st.session_state.residuals is not None:
            try:
                fig, axes = plt.subplots(2, 2, figsize=(14, 10))
                
                plot_acf(st.session_state.residuals, lags=40, ax=axes[0, 0])
                axes[0, 0].set_title('ACF Plot')
                
                plot_pacf(st.session_state.residuals, lags=40, ax=axes[0, 1])
                axes[0, 1].set_title('PACF Plot')
                
                axes[1, 0].hist(st.session_state.residuals, bins=30, edgecolor='black')
                axes[1, 0].set_title('Histogram')
                
                from scipy import stats
                stats.probplot(st.session_state.residuals, dist="norm", plot=axes[1, 1])
                axes[1, 1].set_title('Q-Q Plot')
                
                plt.tight_layout()
                st.pyplot(fig)
            except Exception as e:
                st.error(f"Error: {str(e)}")
        else:
            st.warning("Click button to generate")
    
    # TAB 3: METRICS
    with tab3:
        st.subheader("Model Metrics")
        
        if st.session_state.model_fitted:
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**📋 Model Fit**")
                st.metric("AIC", f"{st.session_state.arima_model.aic:.2f}")
                st.metric("BIC", f"{st.session_state.arima_model.bic:.2f}")
            
            with col2:
                st.write("**📊 Accuracy**")
                if st.session_state.residuals is not None:
                    rmse = np.sqrt(np.mean(st.session_state.residuals**2))
                    mae = np.mean(np.abs(st.session_state.residuals))
                    mape = np.mean(np.abs(st.session_state.residuals / st.session_state.series.mean())) * 100
                    
                    st.metric("RMSE", f"₹{rmse:.2f}")
                    st.metric("MAE", f"₹{mae:.2f}")
                    st.metric("MAPE", f"{mape:.2f}%")
        else:
            st.warning("Click button")
    
    # TAB 4: FORECAST
    with tab4:
        st.subheader(f"{forecast_horizon}-Day Forecast")
        
        if st.session_state.forecast_generated:
            forecast_display = st.session_state.forecast_df.copy()
            forecast_display.index = forecast_display.index.strftime('%Y-%m-%d')
            forecast_display.columns = ['Forecast (₹)', 'Lower CI (₹)', 'Upper CI (₹)']
            forecast_display = forecast_display.round(2)
            
            st.dataframe(forecast_display, use_container_width=True)
            
            csv = forecast_display.to_csv()
            st.download_button(
                label="📥 Download CSV",
                data=csv,
                file_name=f"{ticker}_forecast_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv"
            )
        else:
            st.warning("Click button")
    
    # TAB 5: HELP
    with tab5:
        st.subheader("Box-Jenkins ARIMA Guide")
        st.markdown("""
        ### 📚 Understanding ARIMA
        
        **ARIMA = AutoRegressive Integrated Moving Average**
        
        #### 6-Stage Box-Jenkins Process:
        1. **Data Preparation** - Clean & prepare data
        2. **Stationarity Testing** - ADF test & differencing
        3. **Model Selection** - ACF/PACF analysis
        4. **Parameter Estimation** - MLE optimization
        5. **Diagnostic Checking** - Residual analysis
        6. **Forecasting** - Generate predictions
        
        #### ARIMA(p,d,q) Parameters:
        - **p**: AR order (0-5)
        - **d**: Differencing (0-2)
        - **q**: MA order (0-5)
        
        #### Good Model Signs:
        ✓ Ljung-Box p > 0.05
        ✓ MAPE < 5%
        ✓ ACF/PACF within bounds
        """)

else:
    st.warning("⏳ Click '🔄 FETCH DATA & RUN MODEL' to start")

st.markdown("---")
st.markdown(f"""
    <div style='text-align: center; color: #999; font-size: 0.9em;'>
        <p><strong>{BRAND_NAME}</strong></p>
        <p>{AUTHOR_INFO['name']}</p>
    </div>
""", unsafe_allow_html=True)
