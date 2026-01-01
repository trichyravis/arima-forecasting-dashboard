
# ═══════════════════════════════════════════════════════════════════════════════
# ARIMA FORECASTING DASHBOARD - MAIN APPLICATION
# The Mountain Path - World of Finance
# ═══════════════════════════════════════════════════════════════════════════════

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from statsmodels.tsa.arima.model import ARIMA

from src.config import *

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE CONFIG
# ═══════════════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title=PAGE_TITLE,
    page_icon=PAGE_ICON,
    layout=PAGE_LAYOUT,
    initial_sidebar_state="expanded"
)

# ═══════════════════════════════════════════════════════════════════════════════
# SESSION STATE INITIALIZATION (CRITICAL)
# ═══════════════════════════════════════════════════════════════════════════════

if "run_model" not in st.session_state:
    st.session_state.run_model = False

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
# SIDEBAR
# ═══════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown(f"### {SIDEBAR_SECTIONS['data_selection']}")

    ticker = st.selectbox(
        "Select Ticker",
        list(ALL_TICKERS.keys()),
        format_func=lambda x: f"{x} - {ALL_TICKERS[x]}",
        index=list(ALL_TICKERS.keys()).index(DEFAULT_TICKER)
    )

    lookback_years = st.selectbox("Years of Historical Data", [1,2,3,5,7,10], index=3)
    frequency = st.radio("Data Frequency", ["Daily","Weekly","Monthly"], index=0)

    st.markdown(f"### {SIDEBAR_SECTIONS['model_config']}")

    transformation = st.radio(
        "Price Transformation",
        ["Price Level","Log Prices","Log Returns","Percentage Returns"],
        index=0
    )

    model_mode = st.radio("Model Selection", ["Manual ARIMA","Auto ARIMA"], index=0)

    if model_mode == "Manual ARIMA":
        p = st.slider("p (AR)", 0, 5, DEFAULT_P)
        d = st.slider("d (Diff)", 0, 2, DEFAULT_D)
        q = st.slider("q (MA)", 0, 5, DEFAULT_Q)
    else:
        p = d = q = None

    forecast_horizon = st.slider("Forecast Horizon (Days)", 1, 60, DEFAULT_FORECAST_HORIZON)
    train_pct = st.slider("Training Data %", 60, 95, int(DEFAULT_TRAIN_PCT * 100), step=5)

    if st.button("🔄 FETCH DATA & RUN MODEL", use_container_width=True):
        st.session_state.run_model = True

# ═══════════════════════════════════════════════════════════════════════
# CORE FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════

@st.cache_data
def fetch_data(ticker, years, frequency):
    df = yf.download(ticker, period=f"{years}y", progress=False)
    price = df["Close"].dropna()

    if frequency == "Weekly":
        price = price.resample("W").last()
    elif frequency == "Monthly":
        price = price.resample("M").last()

    return price

def transform_series(ts, method):
    if method == "Log Prices":
        return np.log(ts)
    if method == "Log Returns":
        return np.log(ts).diff().dropna()
    if method == "Percentage Returns":
        return ts.pct_change().dropna()
    return ts

def plot_forecast(ts, fitted, forecast, ci):
    fig = go.Figure()

    fig.add_trace(go.Scatter(x=ts.index, y=ts, name="Historical", line=dict(color="blue")))
    fig.add_trace(go.Scatter(x=fitted.index, y=fitted, name="Fitted", line=dict(color="green")))
    fig.add_trace(go.Scatter(x=forecast.index, y=forecast, name="Forecast", line=dict(color="orange")))

    fig.add_trace(go.Scatter(
        x=list(ci.index) + list(ci.index[::-1]),
        y=list(ci.iloc[:, 0]) + list(ci.iloc[:, 1][::-1]),
        fill="toself",
        fillcolor="rgba(255,165,0,0.25)",
        line=dict(color="rgba(255,255,255,0)"),
        name="95% CI"
    ))

    fig.update_layout(
        height=520,
        template="plotly_white",
        legend=dict(orientation="h")
    )

    return fig

# ═══════════════════════════════════════════════════════════════════════
# RUN MODEL
# ═══════════════════════════════════════════════════════════════════════

if st.session_state.run_model:
    with st.spinner("Fetching data and fitting ARIMA model..."):
        raw = fetch_data(ticker, lookback_years, frequency)
        ts = transform_series(raw, transformation)

        split = int(len(ts) * train_pct / 100)
        train = ts.iloc[:split]

        model = ARIMA(train, order=(p,d,q)) if p is not None else ARIMA(train, order=(1,1,1))
        result = model.fit()

        fitted = result.fittedvalues
        forecast_res = result.get_forecast(steps=forecast_horizon)
        forecast = forecast_res.predicted_mean
        ci = forecast_res.conf_int()

    st.success(f"✓ Data fetched and model fitted for {ticker}")

# ═══════════════════════════════════════════════════════════════════════
# ANALYSIS TABS
# ═══════════════════════════════════════════════════════════════════════

st.markdown("### 📈 Analysis Results")

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    TAB_NAMES['timeseries'],
    TAB_NAMES['diagnostics'],
    TAB_NAMES['metrics'],
    TAB_NAMES['forecast'],
    TAB_NAMES['help']
])

with tab1:
    if st.session_state.run_model:
        fig = plot_forecast(train, fitted, forecast, ci)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("⚠️ Click ‘FETCH DATA & RUN MODEL’")

with tab3:
    if st.session_state.run_model:
        st.metric("AIC", round(result.aic, 2))
        st.metric("BIC", round(result.bic, 2))
        st.metric("Log Likelihood", round(result.llf, 2))

with tab4:
    if st.session_state.run_model:
        out = pd.DataFrame({
            "Forecast": forecast,
            "Lower CI": ci.iloc[:, 0],
            "Upper CI": ci.iloc[:, 1]
        })
        st.dataframe(out, use_container_width=True)
        st.download_button("⬇ Download Forecast CSV", out.to_csv().encode(), "forecast.csv")

# ═══════════════════════════════════════════════════════════════════════
# FOOTER
# ═══════════════════════════════════════════════════════════════════════

st.markdown(f"""
<div style="text-align:center;color:#999;margin-top:2rem">
<b>{BRAND_NAME}</b><br>
{AUTHOR_INFO['name']} | {AUTHOR_INFO['experience']}<br>
{AUTHOR_INFO['academics']}
</div>
""", unsafe_allow_html=True)
