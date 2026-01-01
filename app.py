
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
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

DARK_BLUE = "#003366"
LIGHT_BLUE = "#0066CC"
LIGHT_BLUE_TEXT = "#66CCFF"
GOLD_COLOR = "#FFD700"

BRAND_NAME = "The Mountain Path - World of Finance"
HERO_EMOJI = "🏔️"
HERO_TITLE = "ARIMA FORECASTING DASHBOARD"
HERO_SUBTITLE = "Real-Time Box-Jenkins Time Series Forecasting for Indian Equities"
HERO_DESCRIPTION = "Prof. V. Ravichandran | 28+ Years Corporate Finance & Banking Experience"

SIDEBAR_SECTIONS = {'data_selection': "📊 Data Selection", 'model_config': "⚙️ Model Configuration", 'forecast_settings': "🔮 Forecast Settings"}
TAB_NAMES = {'timeseries': "📈 Time Series & Forecast", 'diagnostics': "🔍 Residual Diagnostics", 'metrics': "📊 Model Metrics", 'forecast': "📋 Forecast Table", 'help': "📚 Help & Guide"}
AUTHOR_INFO = {'name': "Prof. V. Ravichandran", 'experience': "28+ Years Corporate Finance & Banking Experience", 'academics': "10+ Years Academic Excellence", 'linkedin': "https://www.linkedin.com/in/vravichandran"}

ALL_TICKERS = {"^NSEI": "NIFTY 50", "^BSESN": "SENSEX", "RELIANCE.NS": "Reliance Industries", "TCS.NS": "Tata Consultancy Services", "HDFCBANK.NS": "HDFC Bank", "INFY.NS": "Infosys", "ICICIBANK.NS": "ICICI Bank", "SBIN.NS": "State Bank of India", "BHARTIARTL.NS": "Bharti Airtel", "LT.NS": "Larsen & Toubro"}
DEFAULT_TICKER = "^NSEI"

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE CONFIGURATION & CSS (FIXED TEXT VISIBILITY)
# ═══════════════════════════════════════════════════════════════════════════════

st.set_page_config(page_title="ARIMA Dashboard - The Mountain Path", page_icon=HERO_EMOJI, layout="wide")

st.markdown(f"""
    <style>
    /* Hero Section */
    .hero-title {{ background: linear-gradient(135deg, {DARK_BLUE} 0%, {LIGHT_BLUE} 100%); padding: 2rem; border-radius: 20px; margin-bottom: 2rem; box-shadow: 0 12px 30px rgba(0, 51, 102, 0.4); border: 4px solid {DARK_BLUE}; display: flex; align-items: center; gap: 2rem; }}
    .hero-emoji {{ font-size: 100px; animation: float 3s ease-in-out infinite; }}
    .hero-text-right {{ flex: 1; text-align: right; color: white; }}
    .hero-text-right h1 {{ font-size: 32px; font-weight: 900; margin: 0; letter-spacing: 2px; }}
    @keyframes float {{ 0%, 100% {{ transform: translateY(0px); }} 50% {{ transform: translateY(-25px); }} }}
    
    /* Sidebar Styling */
    [data-testid="stSidebar"] {{ 
        background: linear-gradient(135deg, {DARK_BLUE} 0%, {LIGHT_BLUE} 100%) !important; 
    }}
    
    /* Force Sidebar labels/headers to white, but keep input text dark */
    [data-testid="stSidebar"] .stMarkdown p, 
    [data-testid="stSidebar"] .stHeader, 
    [data-testid="stSidebar"] label {{ 
        color: white !important; 
        font-weight: 600;
    }}
    
    /* Ensure text inside Selectboxes and Inputs is visible (Dark text on light background) */
    div[data-baseweb="select"] > div, 
    div[data-baseweb="base-input"] input {{
        color: {DARK_BLUE} !important;
    }}
    
    /* Style the Refresh Button */
    .stButton>button {{
        background-color: {GOLD_COLOR} !important;
        color: {DARK_BLUE} !important;
        font-weight: bold !important;
        border-radius: 10px !important;
    }}
    </style>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def fetch_data(ticker, years):
    end_date = datetime.today()
    start_date = end_date - timedelta(days=years * 365)
    try:
        data = yf.download(ticker, start=start_date, end=end_date)
        if data.empty: return None
        if isinstance(data.columns, pd.MultiIndex):
            return data['Close'][ticker].dropna()
        return data['Close'].dropna()
    except Exception as e:
        st.error(f"❌ Fetch Error: {e}")
        return None

def transform_series(series, method):
    if method == "Price Level": return series
    elif method == "Log Prices": return np.log(series)
    elif method == "Log Returns": return np.log(series).diff().dropna()
    elif method == "Percentage Returns": return series.pct_change().dropna()
    return series

# ═══════════════════════════════════════════════════════════════════════════════
# UI - HERO & SIDEBAR
# ═══════════════════════════════════════════════════════════════════════════════

st.markdown(f"""<div class="hero-title"><div class="hero-emoji">{HERO_EMOJI}</div><div class="hero-text-right"><h1>{HERO_TITLE}</h1><p>{HERO_SUBTITLE}</p><p>{HERO_DESCRIPTION}</p></div></div>""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown(f"### {SIDEBAR_SECTIONS['data_selection']}")
    ticker = st.selectbox("Select Ticker", options=list(ALL_TICKERS.keys()), format_func=lambda x: f"{x} - {ALL_TICKERS[x]}", index=0)
    lookback_years = st.selectbox("Years of Historical Data", options=[1, 2, 3, 5, 7, 10], index=2)
    frequency = st.radio("Data Frequency", ["Daily", "Weekly", "Monthly"])
    
    st.markdown(f"### {SIDEBAR_SECTIONS['model_config']}")
    transformation = st.radio("Price Transformation", ["Price Level", "Log Prices", "Log Returns", "Percentage Returns"], index=2)
    model_mode = st.radio("Model Selection", ["Manual ARIMA", "Auto ARIMA"], index=1)
    
    p, d, q = 1, 1, 1
    if model_mode == "Manual ARIMA":
        c1, c2, c3 = st.columns(3)
        p = c1.slider("p", 0, 5, 1)
        d = c2.slider("d", 0, 2, 1)
        q = c3.slider("q", 0, 5, 1)

    st.markdown(f"### {SIDEBAR_SECTIONS['forecast_settings']}")
    forecast_horizon = st.slider("Forecast Horizon", 1, 60, 10)
    confidence_level = st.selectbox("Confidence Level", ["80%", "90%", "95%", "99%"], index=2)
    train_pct = st.slider("Training Data %", 60, 95, 80)
    refresh_button = st.button("🔄 FETCH DATA & RUN MODEL", use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════════
# MAIN LOGIC & OUTPUT
# ═══════════════════════════════════════════════════════════════════════════════

tab1, tab2, tab3, tab4, tab5 = st.tabs(list(TAB_NAMES.values()))
results = {}

if refresh_button:
    with st.spinner("Analyzing Time Series..."):
        raw_prices = fetch_data(ticker, lookback_years)
        if raw_prices is not None:
            if frequency == "Weekly": raw_prices = raw_prices.resample('W').last()
            elif frequency == "Monthly": raw_prices = raw_prices.resample('M').last()
            
            transformed = transform_series(raw_prices, transformation)
            train_size = int(len(transformed) * (train_pct / 100))
            train = transformed[:train_size]
            alpha = 1 - int(confidence_level.strip('%'))/100
            
            try:
                if model_mode == "Auto ARIMA":
                    model_fit = pm.auto_arima(train, seasonal=False, stepwise=True)
                    order = model_fit.order
                    fc, conf = model_fit.predict(n_periods=forecast_horizon, return_conf_int=True, alpha=alpha)
                    lower_ci, upper_ci = conf[:, 0], conf[:, 1]
                    aic, bic, resid = model_fit.aic(), model_fit.bic(), model_fit.resid()
                else:
                    model = ARIMA(train, order=(p, d, q)).fit()
                    order = (p, d, q)
                    fc_res = model.get_forecast(steps=forecast_horizon)
                    fc = fc_res.predicted_mean.values
                    ci_df = fc_res.conf_int(alpha=alpha)
                    lower_ci, upper_ci = ci_df.iloc[:, 0].values, ci_df.iloc[:, 1].values
                    aic, bic, resid = model.aic, model.bic, model.resid

                # INVERSION
                last_price = raw_prices.iloc[train_size - 1]
                if transformation == "Price Level":
                    inv_fc, inv_low, inv_high = fc, lower_ci, upper_ci
                elif transformation == "Log Prices":
                    inv_fc, inv_low, inv_high = np.exp(fc), np.exp(lower_ci), np.exp(upper_ci)
                else:
                    c_fc, c_low, c_high = np.cumsum(fc), np.cumsum(lower_ci), np.cumsum(upper_ci)
                    if transformation == "Log Returns":
                        inv_fc = last_price * np.exp(c_fc)
                        inv_low, inv_high = last_price * np.exp(c_low), last_price * np.exp(c_high)
                    else:
                        inv_fc = last_price * (1 + c_fc)
                        inv_low, inv_high = last_price * (1 + c_low), last_price * (1 + c_high)

                results = {'raw': raw_prices, 'order': order, 'fc': inv_fc, 'low': inv_low, 'high': inv_high, 'aic': aic, 'bic': bic, 'resid': resid}
            except Exception as e:
                st.error(f"Model Error: {e}")

# Display Logic (Tabs)
if results:
    with tab1:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=results['raw'].index, y=results['raw'], name="Actual Price", line=dict(color=DARK_BLUE)))
        freq_code = {"Daily": "B", "Weekly": "W", "Monthly": "M"}[frequency]
        f_dates = pd.date_range(results['raw'].index[-1], periods=len(results['fc'])+1, freq=freq_code)[1:]
        fig.add_trace(go.Scatter(x=f_dates, y=results['fc'], name="Forecast", line=dict(color="orange", width=3)))
        fig.add_trace(go.Scatter(x=f_dates, y=results['high'], line=dict(width=0), showlegend=False))
        fig.add_trace(go.Scatter(x=f_dates, y=results['low'], fill='tonexty', fillcolor='rgba(255,165,0,0.2)', name="Conf. Interval"))
        st.plotly_chart(fig, use_container_width=True)
    with tab2:
        fig, ax = plt.subplots(1, 2, figsize=(10, 4))
        plot_acf(results['resid'], ax=ax[0]); plot_pacf(results['resid'], ax=ax[1])
        st.pyplot(fig)
    with tab3:
        st.metric("Optimal ARIMA Order", str(results['order']))
        st.metric("AIC", round(results['aic'], 2)); st.metric("BIC", round(results['bic'], 2))
    with tab4:
        st.dataframe(pd.DataFrame({"Date": f_dates, "Forecast": results['fc'], "Lower": results['low'], "Upper": results['high']}))
with tab5:
    st.markdown("### Box-Jenkins Methodology")
    st.write("This dashboard follows the identification, estimation, and diagnostic stages of time series forecasting.")
    ```
