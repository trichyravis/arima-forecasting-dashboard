
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import yfinance as yf
import warnings
from statsmodels.tsa.arima.model import ARIMA
import plotly.graph_objects as go
import pmdarima as pm
from io import BytesIO

# Suppress warnings
warnings.filterwarnings("ignore")

# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════
DARK_BLUE = "#003366"
LIGHT_BLUE = "#0066CC"
GOLD_COLOR = "#FFD700"

ALL_TICKERS = {
    "^NSEI": "NIFTY 50", 
    "^BSESN": "SENSEX", 
    "RELIANCE.NS": "Reliance Industries", 
    "TCS.NS": "TCS", 
    "HDFCBANK.NS": "HDFC Bank",
    "INFY.NS": "Infosys"
}

st.set_page_config(page_title="ARIMA Dashboard - The Mountain Path", page_icon="🏔️", layout="wide")

# ═══════════════════════════════════════════════════════════════════════════════
# CSS FIXES (Sidebar Visibility & White Text)
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown(f"""
    <style>
    [data-testid="stSidebar"] {{ background: linear-gradient(135deg, {DARK_BLUE} 0%, {LIGHT_BLUE} 100%) !important; }}
    [data-testid="stSidebar"] h3, [data-testid="stSidebar"] label, [data-testid="stSidebar"] p, [data-testid="stSidebar"] div[role="radiogroup"] {{ 
        color: white !important; font-weight: 600 !important; 
    }}
    div[data-baseweb="select"] > div {{ color: {DARK_BLUE} !important; }}
    input {{ color: {DARK_BLUE} !important; }}
    .stButton>button {{
        background-color: {GOLD_COLOR} !important;
        color: {DARK_BLUE} !important;
        font-weight: bold !important;
        width: 100%;
    }}
    </style>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# SIDEBAR CONTROLS
# ═══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("### 📊 Data Selection")
    ticker = st.selectbox("Select Ticker", options=list(ALL_TICKERS.keys()), format_func=lambda x: f"{x} - {ALL_TICKERS[x]}")
    lookback = st.selectbox("Years of Historical Data", [1, 2, 3, 5, 10], index=2)
    freq = st.radio("Data Frequency", ["Daily", "Weekly", "Monthly"])
    
    st.markdown("### ⚙️ Model Configuration")
    transformation = st.radio("Price Transformation", ["Price Level", "Log Prices", "Log Returns", "Percentage Returns"], index=2)
    model_mode = st.radio("Model Selection", ["Manual ARIMA", "Auto ARIMA"], index=1)
    
    # Restored Manual Sliders
    p, d, q = 1, 1, 1
    if model_mode == "Manual ARIMA":
        col1, col2, col3 = st.columns(3)
        p = col1.slider("p", 0, 5, 1)
        d = col2.slider("d", 0, 2, 1)
        q = col3.slider("q", 0, 5, 1)
    
    st.markdown("### 🔮 Forecast Settings")
    forecast_horizon = st.slider("Forecast Horizon (Periods)", 1, 60, 10)
    conf_level = st.selectbox("Confidence Level", ["80%", "90%", "95%", "99%"], index=2)
    refresh_button = st.button("🔄 FETCH DATA & RUN MODEL")

# ═══════════════════════════════════════════════════════════════════════════════
# HERO & MAIN PAGE SUMMARY (RESTORED)
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown(f"""<div style='background:linear-gradient(135deg, {DARK_BLUE}, {LIGHT_BLUE}); padding:2rem; border-radius:15px; color:white; text-align:center;'>
    <h1>ARIMA FORECASTING DASHBOARD</h1>
    <p>Prof. V. Ravichandran | 28+ Years Experience</p>
</div>""", unsafe_allow_html=True)

st.markdown("### 📊 Current Selections")
m1, m2, m3, m4 = st.columns(4)
m1.metric("Security", ticker)
m2.metric("Years", f"{lookback}y")
m3.metric("Mode", model_mode)
m4.metric("Confidence", conf_level)

# ═══════════════════════════════════════════════════════════════════════════════
# MODELING LOGIC
# ═══════════════════════════════════════════════════════════════════════════════
tab1, tab2, tab3 = st.tabs(["📈 Forecast Chart", "📊 Model Metrics", "📋 Export Data"])
results = None

if refresh_button:
    with st.spinner("Calculating..."):
        data = yf.download(ticker, start=datetime.now()-timedelta(days=lookback*365))
        if not data.empty:
            raw_prices = data['Close'][ticker] if isinstance(data.columns, pd.MultiIndex) else data['Close']
            raw_prices = raw_prices.dropna()
            
            # Resampling
            res_map = {"Daily": "B", "Weekly": "W", "Monthly": "M"}
            raw_prices = raw_prices.resample(res_map[freq]).last().ffill()
            
            # Transformation
            if transformation == "Log Returns": train_series = np.log(raw_prices).diff().dropna()
            elif transformation == "Log Prices": train_series = np.log(raw_prices)
            elif transformation == "Percentage Returns": train_series = raw_prices.pct_change().dropna()
            else: train_series = raw_prices

            try:
                if model_mode == "Auto ARIMA":
                    model = pm.auto_arima(train_series, seasonal=False)
                    fc_vals = model.predict(n_periods=forecast_horizon)
                    order, aic = model.order, model.aic()
                else:
                    fit = ARIMA(train_series, order=(p, d, q)).fit()
                    fc_vals = fit.forecast(steps=forecast_horizon)
                    order, aic = (p, d, q), fit.aic

                # Inversion
                last_p = raw_prices.iloc[-1]
                if transformation == "Log Returns": forecast_prices = last_p * np.exp(np.cumsum(fc_vals))
                elif transformation == "Log Prices": forecast_prices = np.exp(fc_vals)
                elif transformation == "Percentage Returns": forecast_prices = last_p * (1 + np.cumsum(fc_vals))
                else: forecast_prices = fc_vals
                
                f_dates = pd.date_range(raw_prices.index[-1], periods=forecast_horizon + 1, freq=res_map[freq])[1:]
                fc_df = pd.DataFrame({"Forecasted Price": np.array(forecast_prices).flatten()}, index=f_dates)
                
                results = {"raw": raw_prices, "fc_df": fc_df, "order": order, "aic": aic}
            except Exception as e:
                st.error(f"Error: {e}")

# ═══════════════════════════════════════════════════════════════════════════════
# OUTPUT DISPLAY
# ═══════════════════════════════════════════════════════════════════════════════
if results:
    with tab1:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=results["raw"].index, y=results["raw"], name="Historical"))
        fig.add_trace(go.Scatter(x=results["fc_df"].index, y=results["fc_df"]["Forecasted Price"], name="Forecast", line=dict(color='orange')))
        st.plotly_chart(fig, use_container_width=True)
    with tab2:
        st.write(f"**Final ARIMA Order:** {results['order']}")
        st.write(f"**AIC Score:** {results['aic']:.2f}")
    with tab3:
        st.dataframe(results["fc_df"].style.format("{:.2f}"), use_container_width=True)
        buffer = BytesIO()
        with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
            results["fc_df"].to_excel(writer, sheet_name='Forecast')
        st.download_button(label="📥 Download Excel", data=buffer.getvalue(), file_name=f"{ticker}_forecast.xlsx")
