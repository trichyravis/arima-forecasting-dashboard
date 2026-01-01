
"""
═══════════════════════════════════════════════════════════════════════════════
ARIMA FORECASTING DASHBOARD - MAIN APPLICATION
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
from datetime import datetime, timedelta
import yfinance as yf
import warnings
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pmdarima as pm

# Suppress warnings for cleaner UI
warnings.filterwarnings("ignore")

# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION (INLINE FOR PORTABILITY - REPLACE WITH src.config IF PREFERRED)
# ═══════════════════════════════════════════════════════════════════════════════

# Colors
DARK_BLUE = "#003366"
LIGHT_BLUE = "#0066CC"
LIGHT_BLUE_TEXT = "#66CCFF"
GOLD_COLOR = "#FFD700"
WHITE = "#FFFFFF"
DARK_TEXT = "#333333"
LIGHT_GRAY = "#F0F0F0"

# Branding
BRAND_NAME = "The Mountain Path - World of Finance"
APP_NAME = "ARIMA Forecasting Dashboard"
HERO_EMOJI = "🏔️"
HERO_TITLE = "ARIMA FORECASTING DASHBOARD"
HERO_SUBTITLE = "Real-Time Box-Jenkins Time Series Forecasting for Indian Equities"
HERO_DESCRIPTION = "Prof. V. Ravichandran | 28+ Years Corporate Finance & Banking Experience"

SIDEBAR_SECTIONS = {
    'data_selection': "📊 Data Selection",
    'model_config': "⚙️ Model Configuration",
    'forecast_settings': "🔮 Forecast Settings"
}
TAB_NAMES = {
    'timeseries': "📈 Time Series & Forecast",
    'diagnostics': "🔍 Residual Diagnostics",
    'metrics': "📊 Model Metrics",
    'forecast': "📋 Forecast Table",
    'help': "📚 Help & Guide"
}
ABOUT_DESCRIPTION = """
This tool implements the **Box-Jenkins ARIMA methodology** for forecasting Indian equity prices. 
It supports manual and automatic ARIMA model selection, diagnostic testing, and confidence intervals.
"""
AUTHOR_INFO = {
    'name': "Prof. V. Ravichandran",
    'experience': "28+ Years Corporate Finance & Banking Experience",
    'academics': "10+ Years Academic Excellence",
    'linkedin': "https://www.linkedin.com/in/vravichandran"
}

# UI Config
PAGE_LAYOUT = "wide"
PAGE_ICON = "🏔️"
PAGE_TITLE = "ARIMA Forecasting Dashboard - The Mountain Path"

# Data Config
ALL_TICKERS = {
    "^NSEI": "NIFTY 50",
    "^BSESN": "SENSEX",
    "RELIANCE.NS": "Reliance Industries",
    "TCS.NS": "Tata Consultancy Services",
    "HDFCBANK.NS": "HDFC Bank",
    "INFY.NS": "Infosys",
    "ICICIBANK.NS": "ICICI Bank",
    "SBIN.NS": "State Bank of India",
    "BHARTIARTL.NS": "Bharti Airtel",
    "LT.NS": "Larsen & Toubro"
}
DEFAULT_TICKER = "^NSEI"
DEFAULT_LOOKBACK_YEARS = 5

# ARIMA Config
DEFAULT_P = 1
DEFAULT_D = 1
DEFAULT_Q = 1
DEFAULT_FORECAST_HORIZON = 10

# Other
DEFAULT_TRAIN_PCT = 0.8
DEFAULT_TRANSFORMATION = "Log Returns"

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title=PAGE_TITLE,
    page_icon=PAGE_ICON,
    layout=PAGE_LAYOUT,
    initial_sidebar_state="expanded"
)

# ═══════════════════════════════════════════════════════════════════════════════
# CUSTOM CSS STYLING - MOUNTAIN PATH DESIGN SYSTEM
# ═══════════════════════════════════════════════════════════════════════════════

st.markdown(f"""
    <style>
    /* ... (Your existing CSS remains unchanged) ... */
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
    
    [data-testid="stSidebar"] a {{
        color: {GOLD_COLOR} !important;
    }}
    
    [data-testid="stTabs"] [aria-selected="true"] {{
        color: {DARK_BLUE} !important;
        border-bottom: 3px solid {GOLD_COLOR} !important;
    }}
    
    hr {{
        border-color: rgba(0, 51, 102, 0.2) !important;
    }}
    
    @media (max-width: 768px) {{
        .hero-title {{
            flex-direction: column;
            text-align: center;
            padding: 1.5rem 1.5rem;
        }}
        .hero-emoji {{ font-size: 80px; }}
        .hero-text-right {{ text-align: center; }}
        .hero-text-right h1 {{ font-size: 24px; }}
    }}
    </style>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def fetch_data(ticker, years):
    """Fetch OHLC data from Yahoo Finance"""
    end_date = datetime.today()
    start_date = end_date - timedelta(days=years * 365)
    try:
        data = yf.download(ticker, start=start_date, end=end_date)
        if data.empty:
            return None
        return data['Close'].dropna()
    except Exception as e:
        st.error(f"❌ Failed to fetch data for {ticker}: {e}")
        return None

def transform_series(series, method):
    """Apply transformation to price series"""
    if method == "Price Level":
        return series
    elif method == "Log Prices":
        return np.log(series)
    elif method == "Log Returns":
        return np.log(series).diff().dropna()
    elif method == "Percentage Returns":
        return series.pct_change().dropna()
    return series

def invert_transform(forecast, last_obs, method, original_series):
    """Invert transformation for forecast interpretation"""
    if method == "Price Level":
        return forecast
    elif method == "Log Prices":
        return np.exp(forecast)
    elif method in ["Log Returns", "Percentage Returns"]:
        # Cumulative reconstruction from last observed price
        cumsum = np.cumsum(forecast)
        if method == "Log Returns":
            return last_obs * np.exp(cumsum)
        else:
            return last_obs * (1 + cumsum)
    return forecast

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
# SIDEBAR - DATA SELECTION & MODEL CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("---")
    st.markdown(f"### {SIDEBAR_SECTIONS['data_selection']}")
    
    ticker = st.selectbox(
        "Select Ticker",
        options=list(ALL_TICKERS.keys()),
        format_func=lambda x: f"{x} - {ALL_TICKERS[x]}",
        index=list(ALL_TICKERS.keys()).index(DEFAULT_TICKER),
        help="Choose from NIFTY indices, major stocks, or cryptocurrencies"
    )
    
    lookback_years = st.selectbox(
        "Years of Historical Data",
        options=[1, 2, 3, 5, 7, 10],
        index=3,  # Default 5 years (index 3)
        help="More data = more stable model, but older patterns"
    )
    
    frequency = st.radio(
        "Data Frequency",
        ["Daily", "Weekly", "Monthly"],
        index=0,
        help="Higher frequency = more observations, more noise"
    )
    
    st.markdown("---")
    st.markdown(f"### {SIDEBAR_SECTIONS['model_config']}")
    
    transformation = st.radio(
        "Price Transformation",
        ["Price Level", "Log Prices", "Log Returns", "Percentage Returns"],
        index=2,  # Default: Log Returns
        help="Log returns reduce heteroscedasticity"
    )
    
    model_mode = st.radio(
        "Model Selection",
        ["Manual ARIMA", "Auto ARIMA"],
        index=1,
        help="Manual: specify (p,d,q) | Auto: uses AIC to find best"
    )
    
    if model_mode == "Manual ARIMA":
        st.write("**Set ARIMA Parameters (p, d, q)**")
        col1, col2, col3 = st.columns(3)
        with col1:
            p = st.slider("p (AR Order)", 0, 5, DEFAULT_P, help="Auto-Regressive terms")
        with col2:
            d = st.slider("d (Differencing)", 0, 2, DEFAULT_D, help="Differencing order")
        with col3:
            q = st.slider("q (MA Order)", 0, 5, DEFAULT_Q, help="Moving Average terms")
    else:
        p, d, q = None, None, None
    
    st.markdown("---")
    st.markdown(f"### {SIDEBAR_SECTIONS['forecast_settings']}")
    
    forecast_horizon = st.slider(
        "Forecast Horizon (Days)",
        min_value=1,
        max_value=60,
        value=DEFAULT_FORECAST_HORIZON,
        help="How many days ahead to forecast"
    )
    
    confidence_level = st.selectbox(
        "Confidence Level",
        ["80%", "90%", "95%", "99%"],
        index=2,
        help="Confidence interval for forecast bands"
    )
    
    train_pct = st.slider(
        "Training Data %",
        min_value=60,
        max_value=95,
        value=int(DEFAULT_TRAIN_PCT * 100),
        step=5,
        help="% of data for model training (rest for testing)"
    )
    
    st.markdown("---")
    refresh_button = st.button(
        f"🔄 FETCH DATA & RUN MODEL",
        use_container_width=True,
        key="refresh_button",
        help="Click to fetch data and run ARIMA model"
    )
    
    st.markdown("---")
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
# MAIN CONTENT
# ═══════════════════════════════════════════════════════════════════════════════

# Metrics Row
st.markdown("### 📊 Data Summary")
col1, col2, col3, col4, col5 = st.columns(5)
with col1: st.metric("Ticker", ticker)
with col2: st.metric("Lookback", f"{lookback_years}y")
with col3: st.metric("Model Mode", "Manual" if model_mode == "Manual ARIMA" else "Auto")
with col4: st.metric("Forecast Days", forecast_horizon)
with col5: st.metric("Train/Test", f"{train_pct}% / {100-train_pct}%")
st.markdown("---")

# Tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs(list(TAB_NAMES.values()))

# Placeholder for results
results = {}

if refresh_button:
    with st.spinner("🔄 Fetching data and fitting ARIMA model..."):
        # 1. Fetch data
        price_data = fetch_data(ticker, lookback_years)
        if price_data is None or len(price_data) < 30:
            st.error("❌ Not enough data. Try a different ticker or longer lookback period.")
        else:
            # 2. Resample if needed
            if frequency == "Weekly":
                price_data = price_data.resample('W').last()
            elif frequency == "Monthly":
                price_data = price_data.resample('M').last()
            
            # 3. Transform
            transformed = transform_series(price_data, transformation)
            
            # 4. Split
            n = len(transformed)
            train_size = int(train_pct / 100 * n)
            train, test = transformed[:train_size], transformed[train_size:]
            
            # 5. Fit model
            try:
                if model_mode == "Auto ARIMA":
                    auto_model = pm.auto_arima(
                        train,
                        seasonal=False,
                        stepwise=True,
                        suppress_warnings=True,
                        error_action="ignore",
                        max_p=5, max_q=5, max_d=2
                    )
                    order = auto_model.order
                    fitted = auto_model.fit(train)
                    forecast, conf_int = fitted.predict(
                        n_periods=forecast_horizon,
                        return_conf_int=True,
                        alpha=1 - float(confidence_level.strip('%'))/100
                    )
                    aic = fitted.aic()
                    bic = fitted.bic()
                else:
                    model = ARIMA(train, order=(p, d, q))
                    fitted = model.fit()
                    forecast = fitted.forecast(steps=forecast_horizon)
                    conf_int = fitted.get_forecast(steps=forecast_horizon).conf_int(
                        alpha=1 - float(confidence_level.strip('%'))/100
                    )
                    order = (p, d, q)
                    aic = fitted.aic
                    bic = fitted.bic
                
                # Invert forecast if needed
                last_price = price_data.iloc[-1]
                forecast_inverted = invert_transform(forecast, last_price, transformation, price_data)
                lower_ci = invert_transform(conf_int[:, 0], last_price, transformation, price_data)
                upper_ci = invert_transform(conf_int[:, 1], last_price, transformation, price_data)
                
                # Store results
                results = {
                    'price_data': price_data,
                    'train': train,
                    'test': test,
                    'fitted_values': fitted.fittedvalues,
                    'forecast': forecast_inverted,
                    'lower_ci': lower_ci,
                    'upper_ci': upper_ci,
                    'order': order,
                    'aic': aic,
                    'bic': bic,
                    'residuals': fitted.resid
                }
                
            except Exception as e:
                st.error(f"❌ Model fitting failed: {e}")

# TAB 1: Time Series Chart
with tab1:
    st.subheader("Time Series Chart with Forecast")
    if results:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=results['price_data'].index,
            y=results['price_data'],
            mode='lines',
            name='Historical',
            line=dict(color='blue')
        ))
        # Fitted (in-sample)
        fitted_inverted = invert_transform(
            results['fitted_values'], 
            results['price_data'][results['fitted_values'].index[0] - timedelta(days=1)], 
            transformation, 
            results['price_data']
        )
        fig.add_trace(go.Scatter(
            x=fitted_inverted.index,
            y=fitted_inverted,
            mode='lines',
            name='Fitted',
            line=dict(color='green')
        ))
        # Forecast
        future_dates = pd.date_range(start=results['price_data'].index[-1] + timedelta(days=1), periods=forecast_horizon)
        fig.add_trace(go.Scatter(
            x=future_dates,
            y=results['forecast'],
            mode='lines',
            name='Forecast',
            line=dict(color='orange')
        ))
        fig.add_trace(go.Scatter(
            x=future_dates,
            y=results['upper_ci'],
            mode='lines',
            line=dict(width=0),
            showlegend=False
        ))
        fig.add_trace(go.Scatter(
            x=future_dates,
            y=results['lower_ci'],
            mode='lines',
            fill='tonexty',
            fillcolor='rgba(255,165,0,0.2)',
            line=dict(width=0),
            showlegend=False
        ))
        fig.update_layout(title="Time Series with ARIMA Forecast", xaxis_title="Date", yaxis_title="Price")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("⚠️ Click '🔄 FETCH DATA & RUN MODEL' to generate chart")

# TAB 2: Diagnostics
with tab2:
    st.subheader("Residual Diagnostics")
    if results:
        residuals = results['residuals']
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        plot_acf(residuals, ax=axes[0,0], lags=20)
        plot_pacf(residuals, ax=axes[0,1], lags=20)
        axes[1,0].hist(residuals, bins=20, edgecolor='k')
        axes[1,0].set_title('Histogram of Residuals')
        from scipy import stats
        stats.probplot(residuals, dist="norm", plot=axes[1,1])
        axes[1,1].set_title('Q-Q Plot')
        st.pyplot(fig)
    else:
        st.warning("⚠️ Click '🔄 FETCH DATA & RUN MODEL' to generate diagnostics")

# TAB 3: Metrics
with tab3:
    st.subheader("Model Metrics")
    if results:
        col1, col2 = st.columns(2)
        with col1:
            st.metric("AIC", f"{results['aic']:.2f}")
            st.metric("BIC", f"{results['bic']:.2f}")
        with col2:
            # Compute RMSE if test set exists
            if len(results['test']) > 0:
                test_pred = invert_transform(
                    results['fitted_values'].iloc[-len(results['test']):],
                    results['price_data'][results['fitted_values'].index[-len(results['test'])] - timedelta(days=1)],
                    transformation,
                    results['price_data']
                )
                rmse = np.sqrt(np.mean((test_pred.values - results['price_data'].loc[test_pred.index].values)**2))
                st.metric("RMSE", f"{rmse:.2f}")
            else:
                st.metric("RMSE", "N/A (no test set)")
        st.write(f"**ARIMA Order**: {results['order']}")
    else:
        st.warning("⚠️ Click '🔄 FETCH DATA & RUN MODEL' to generate metrics")

# TAB 4: Forecast Table
with tab4:
    st.subheader(f"{forecast_horizon}-Day Forecast")
    if results:
        forecast_df = pd.DataFrame({
            'Date': pd.date_range(start=results['price_data'].index[-1] + timedelta(days=1), periods=forecast_horizon),
            'Forecast': results['forecast'],
            'Lower CI': results['lower_ci'],
            'Upper CI': results['upper_ci']
        })
        st.dataframe(forecast_df.round(2), use_container_width=True)
    else:
        st.warning("⚠️ Click '🔄 FETCH DATA & RUN MODEL' to generate forecast")

# TAB 5: Help (unchanged)
with tab5:
    st.subheader("Box-Jenkins ARIMA Methodology")
    st.markdown("""
    ### 📚 Understanding ARIMA Forecasting
    ... (your existing help text) ...
    """)  # Keep your original help content

# Footer
st.markdown("---")
st.markdown(f"""
    <div style='text-align: center; color: #999; font-size: 0.9em; margin-top: 2rem;'>
        <p><strong>{BRAND_NAME}</strong></p>
        <p>{AUTHOR_INFO['name']} | {AUTHOR_INFO['experience']}</p>
        <p style='font-size: 0.8em;'>{AUTHOR_INFO['academics']}</p>
    </div>
""", unsafe_allow_html=True)

# Debug
if st.sidebar.checkbox("🔧 Show Debug Info", key="debug_checkbox"):
    st.sidebar.markdown("---")
    st.sidebar.write("**DEBUG INFORMATION**")
    st.sidebar.write(f"Ticker: `{ticker}`")
    st.sidebar.write(f"Lookback: `{lookback_years}y`")
    st.sidebar.write(f"Model Mode: `{model_mode}`")
    if model_mode == "Manual ARIMA":
        st.sidebar.write(f"ARIMA Order: `({p},{d},{q})`")
    st.sidebar.write(f"Refresh Clicked: `{refresh_button}`")
    if results:
        st.sidebar.write("✅ Results available")
