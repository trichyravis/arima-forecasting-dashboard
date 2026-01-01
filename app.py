
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

# ═══════════════════════════════════════════════════════════════════════════════
# IMPORTS FROM CONFIG
# ═══════════════════════════════════════════════════════════════════════════════

from src.config import (
    DARK_BLUE, LIGHT_BLUE, LIGHT_BLUE_TEXT, GOLD_COLOR, WHITE, DARK_TEXT, LIGHT_GRAY,
    BRAND_NAME, APP_NAME, HERO_EMOJI, HERO_TITLE, HERO_SUBTITLE, HERO_DESCRIPTION,
    SIDEBAR_SECTIONS, TAB_NAMES, ABOUT_DESCRIPTION, AUTHOR_INFO,
    PAGE_LAYOUT, PAGE_ICON, PAGE_TITLE,
    ALL_TICKERS, DEFAULT_TICKER, DEFAULT_LOOKBACK_YEARS,
    DEFAULT_P, DEFAULT_D, DEFAULT_Q, DEFAULT_FORECAST_HORIZON,
    DEFAULT_TRAIN_PCT, DEFAULT_TRANSFORMATION
)

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
# SESSION STATE INITIALIZATION (CRITICAL FIX)
# ═══════════════════════════════════════════════════════════════════════

if "refresh_clicked" not in st.session_state:
    st.session_state.refresh_clicked = False

# ═══════════════════════════════════════════════════════════════════════════════
# CUSTOM CSS STYLING - MOUNTAIN PATH DESIGN SYSTEM
# ═══════════════════════════════════════════════════════════════════════════════

st.markdown(f"""
<style>
/* ALL YOUR ORIGINAL CSS — UNCHANGED */
.hero-title {{
    background: linear-gradient(135deg, {DARK_BLUE} 0%, {LIGHT_BLUE} 100%);
    padding: 2rem;
    border-radius: 20px;
    box-shadow: 0 12px 30px rgba(0,51,102,0.4);
    border: 4px solid {DARK_BLUE};
    display: flex;
    gap: 2rem;
}}
.hero-emoji {{ font-size: 100px; animation: float 3s ease-in-out infinite; }}
@keyframes float {{ 0%,100%{{transform:translateY(0)}} 50%{{transform:translateY(-25px)}} }}
</style>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# HERO HEADER
# ═══════════════════════════════════════════════════════════════════════════════

st.markdown(f"""
<div class="hero-title">
    <div class="hero-emoji">{HERO_EMOJI}</div>
    <div>
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

    lookback_years = st.selectbox("Years of Historical Data", [1,2,3,5,7,10], index=2)
    frequency = st.radio("Data Frequency", ["Daily","Weekly","Monthly"], index=0)

    st.markdown(f"### {SIDEBAR_SECTIONS['model_config']}")

    transformation = st.radio(
        "Price Transformation",
        ["Price Level","Log Prices","Log Returns","Percentage Returns"],
        index=0
    )

    model_mode = st.radio("Model Selection", ["Manual ARIMA","Auto ARIMA"], index=0)

    if model_mode == "Manual ARIMA":
        col1,col2,col3 = st.columns(3)
        with col1: p = st.slider("p",0,5,DEFAULT_P)
        with col2: d = st.slider("d",0,2,DEFAULT_D)
        with col3: q = st.slider("q",0,5,DEFAULT_Q)
    else:
        p=d=q=None

    forecast_horizon = st.slider("Forecast Horizon (Days)",1,60,DEFAULT_FORECAST_HORIZON)
    train_pct = st.slider("Training Data %",60,95,int(DEFAULT_TRAIN_PCT*100),step=5)

    refresh_button = st.button("🔄 FETCH DATA & RUN MODEL", use_container_width=True)

    if refresh_button:
        st.session_state.refresh_clicked = True

# ═══════════════════════════════════════════════════════════════════════════════
# MAIN CONTENT
# ═══════════════════════════════════════════════════════════════════════════════

st.markdown("### 📈 Analysis Results")

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    TAB_NAMES['timeseries'],
    TAB_NAMES['diagnostics'],
    TAB_NAMES['metrics'],
    TAB_NAMES['forecast'],
    TAB_NAMES['help']
])

# ───────────────── TAB 1 ─────────────────
with tab1:
    if st.session_state.refresh_clicked:
        st.success(f"✓ Data fetched for {ticker}")
        st.info("📊 Placeholder: Interactive Plotly chart will appear here in Week 2")
    else:
        st.warning("⚠️ Click 'FETCH DATA & RUN MODEL' to generate chart")

# ───────────────── TAB 2 ─────────────────
with tab2:
    if st.session_state.refresh_clicked:
        st.success("✓ Diagnostics calculated")
        st.info("📊 Placeholder: Diagnostic plots will appear here in Week 2")
    else:
        st.warning("⚠️ Click 'FETCH DATA & RUN MODEL'")

# ───────────────── TAB 3 ─────────────────
with tab3:
    st.metric("AIC","TBD")
    st.metric("BIC","TBD")
    st.metric("Log-Likelihood","TBD")

# ───────────────── TAB 4 ─────────────────
with tab4:
    if st.session_state.refresh_clicked:
        st.dataframe(pd.DataFrame({
            "Date": pd.date_range(datetime.today(), periods=10),
            "Forecast": ["TBD"]*10
        }))
    else:
        st.warning("⚠️ Click 'FETCH DATA & RUN MODEL'")

# ───────────────── TAB 5 ─────────────────
with tab5:
    st.markdown("### Box-Jenkins ARIMA Methodology\n(unchanged)")

# ═══════════════════════════════════════════════════════════════════════
# DEBUG
# ═══════════════════════════════════════════════════════════════════════

if st.sidebar.checkbox("🔧 Show Debug Info"):
    st.sidebar.write(f"Refresh Clicked: {st.session_state.refresh_clicked}")
