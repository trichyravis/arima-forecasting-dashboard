
"""
═══════════════════════════════════════════════════════════════════════════════
ARIMA FORECASTING DASHBOARD - MAIN APPLICATION
The Mountain Path - World of Finance
═══════════════════════════════════════════════════════════════════════════════
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import yfinance as yf
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
from datetime import timedelta

# ═══════════════════════════════════════════════════════════════════════════════
# IMPORTS FROM CONFIG
# ═══════════════════════════════════════════════════════════════════════════════

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
# DATA FETCHING
# ═══════════════════════════════════════════════════════════════════════════════

@st.cache_data(show_spinner=False)
def fetch_price_data(ticker, years, frequency):
    df = yf.download(ticker, period=f"{years}y", progress=False)

    if frequency == "Weekly":
        df = df.resample("W").last()
    elif frequency == "Monthly":
        df = df.resample("M").last()

    return df["Close"].dropna()

# ═══════════════════════════════════════════════════════════════════════════════
# TRANSFORMATION
# ═══════════════════════════════════════════════════════════════════════════════

def apply_transformation(series, transformation):
    if transformation == "Log Prices":
        return np.log(series)
    if transformation == "Log Returns":
        return np.log(series).diff().dropna()
    if transformation == "Percentage Returns":
        return series.pct_change().dropna()
    return series

# ═══════════════════════════════════════════════════════════════════════════════
# FORECAST PLOT
# ═══════════════════════════════════════════════════════════════════════════════

def plot_forecast(ts, fitted, forecast, conf_int):
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=ts.index, y=ts,
        name="Historical",
        line=dict(color="blue")
    ))

    fig.add_trace(go.Scatter(
        x=fitted.index, y=fitted,
        name="Fitted",
        line=dict(color="green")
    ))

    fig.add_trace(go.Scatter(
        x=forecast.index, y=forecast,
        name="Forecast",
        line=dict(color="orange")
    ))

    fig.add_trace(go.Scatter(
        x=list(conf_int.index) + list(conf_int.index[::-1]),
        y=list(conf_int.iloc[:, 0]) + list(conf_int.iloc[:, 1][::-1]),
        fill="toself",
        fillcolor="rgba(255,165,0,0.2)",
        line=dict(color="rgba(255,255,255,0)"),
        name="95% CI"
    ))

    fig.update_layout(
        height=520,
        template="plotly_white",
        legend=dict(orientation="h"),
        margin=dict(l=30, r=30, t=40, b=30)
    )

    return fig

# ═══════════════════════════════════════════════════════════════════════════════
# RUN MODEL ON REFRESH
# ═══════════════════════════════════════════════════════════════════════════════

if refresh_button:

    with st.spinner("📥 Fetching data & running ARIMA model..."):

        raw_ts = fetch_price_data(ticker, lookback_years, frequency)
        ts = apply_transformation(raw_ts, transformation)

        split = int(len(ts) * train_pct / 100)
        train, test = ts.iloc[:split], ts.iloc[split:]

        order = (p, d, q) if model_mode == "Manual ARIMA" else (1, 1, 1)

        model = ARIMA(train, order=order)
        model_fit = model.fit()

        fitted_vals = model_fit.fittedvalues
        forecast_res = model_fit.get_forecast(steps=forecast_horizon)

        forecast_vals = forecast_res.predicted_mean
        conf_int = forecast_res.conf_int()

        st.session_state["model_output"] = {
            "ts": train,
            "fitted": fitted_vals,
            "forecast": forecast_vals,
            "conf_int": conf_int,
            "model_fit": model_fit,
            "test": test
        }

# ═══════════════════════════════════════════════════════════════════════════════
# TABS
# ═══════════════════════════════════════════════════════════════════════════════

with tab1:
    st.subheader("Time Series Chart with Forecast")

    if "model_output" in st.session_state:
        fig = plot_forecast(
            st.session_state["ts"],
            st.session_state["fitted"],
            st.session_state["forecast"],
            st.session_state["conf_int"]
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("⚠️ Click 'FETCH DATA & RUN MODEL'")

with tab2:
    st.subheader("Residual Diagnostics")
    if "model_output" in st.session_state:
        resid = st.session_state["model_output"]["model_fit"].resid
        st.line_chart(resid)
    else:
        st.warning("⚠️ Run model first")

with tab3:
    st.subheader("Model Metrics")
    if "model_output" in st.session_state:
        mf = st.session_state["model_output"]["model_fit"]
        st.metric("AIC", round(mf.aic, 2))
        st.metric("BIC", round(mf.bic, 2))
        st.metric("Log Likelihood", round(mf.llf, 2))
    else:
        st.warning("⚠️ Run model first")

with tab4:
    st.subheader("Forecast Table")
    if "model_output" in st.session_state:
        forecast_df = pd.DataFrame({
            "Forecast": st.session_state["model_output"]["forecast"],
            "Lower CI": st.session_state["model_output"]["conf_int"].iloc[:, 0],
            "Upper CI": st.session_state["model_output"]["conf_int"].iloc[:, 1]
        })
        st.dataframe(forecast_df, use_container_width=True)

        st.download_button(
            "⬇️ Download Forecast CSV",
            forecast_df.to_csv().encode(),
            "forecast.csv",
            "text/csv"
        )
    else:
        st.warning("⚠️ Run model first")

with tab5:
    st.markdown("### ARIMA Help & Interpretation")
    st.markdown("See methodology explanation above (unchanged).")
