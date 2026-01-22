
"""
═══════════════════════════════════════════════════════════════════════════════
ARIMA FORECASTING DASHBOARD - MAIN APPLICATION (ENHANCED WITH IMPLEMENTATION)
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
import plotly.express as px
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# ═══════════════════════════════════════════════════════════════════════════════
# IMPORTS FROM CONFIG & MODULES
# ═══════════════════════════════════════════════════════════════════════════════

from src.config import (
    # Colors
    DARK_BLUE, LIGHT_BLUE, LIGHT_BLUE_TEXT, GOLD_COLOR, WHITE, DARK_TEXT, LIGHT_GRAY,
    # Branding
    BRAND_NAME, APP_NAME, HERO_EMOJI, HERO_TITLE, HERO_SUBTITLE, HERO_DESCRIPTION,
    SIDEBAR_SECTIONS, TAB_NAMES, ABOUT_DESCRIPTION, AUTHOR_INFO,
    # UI Config
    PAGE_LAYOUT, PAGE_ICON, PAGE_TITLE,
    # Data Config
    ALL_TICKERS, DEFAULT_TICKER, DEFAULT_LOOKBACK_YEARS,
    # ARIMA Config
    DEFAULT_P, DEFAULT_D, DEFAULT_Q, DEFAULT_FORECAST_HORIZON,
    # Other
    DEFAULT_TRAIN_PCT, DEFAULT_TRANSFORMATION
)

from src.data.loader import DataLoader
from src.models.arima import ARIMAModel

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
    /* ═══════════════════════════════════════════════════════════════════════ */
    /* HERO HEADER - BRAND IDENTITY */
    /* ═══════════════════════════════════════════════════════════════════════ */
    
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
    
    /* ═══════════════════════════════════════════════════════════════════════ */
    /* SIDEBAR STYLING */
    /* ═══════════════════════════════════════════════════════════════════════ */
    
    [data-testid="stSidebar"] {{
        background: linear-gradient(135deg, {DARK_BLUE} 0%, {LIGHT_BLUE} 100%) !important;
    }}
    
    [data-testid="stSidebar"] [data-testid="stVerticalBlock"] > div:has(> label) {{
        background-color: transparent;
    }}
    
    /* Sidebar text - white */
    [data-testid="stSidebar"] * {{
        color: white !important;
    }}
    
    /* Sidebar headers */
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3,
    [data-testid="stSidebar"] .stMarkdown h1,
    [data-testid="stSidebar"] .stMarkdown h2,
    [data-testid="stSidebar"] .stMarkdown h3 {{
        color: white !important;
        font-weight: 700 !important;
        text-shadow: 1px 1px 3px rgba(0, 0, 0, 0.3);
    }}
    
    /* Sidebar paragraphs and text */
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] span,
    [data-testid="stSidebar"] div {{
        color: white !important;
    }}
    
    /* Sidebar radio buttons - gold accent */
    [data-testid="stSidebar"] [role="radio"] {{
        accent-color: {GOLD_COLOR} !important;
    }}
    
    [data-testid="stSidebar"] .stRadio > label {{
        color: white !important;
        font-weight: 500;
    }}
    
    /* Sidebar dividers */
    [data-testid="stSidebar"] hr {{
        border-color: rgba(255, 255, 255, 0.3) !important;
    }}
    
    /* Sidebar links - gold color */
    [data-testid="stSidebar"] a {{
        color: {GOLD_COLOR} !important;
    }}
    
    [data-testid="stSidebar"] a:hover {{
        color: #FFF9E6 !important;
    }}
    
    /* ═══════════════════════════════════════════════════════════════════════ */
    /* MAIN CONTENT STYLING */
    /* ═══════════════════════════════════════════════════════════════════════ */
    
    .main {{
        padding: 0rem 1rem;
    }}
    
    /* Tabs styling */
    [data-testid="stTabs"] [aria-selected="true"] {{
        color: {DARK_BLUE} !important;
        border-bottom: 3px solid {GOLD_COLOR} !important;
    }}
    
    /* Dividers */
    hr {{
        border-color: rgba(0, 51, 102, 0.2) !important;
    }}
    
    /* ═══════════════════════════════════════════════════════════════════════ */
    /* RESPONSIVE DESIGN */
    /* ═══════════════════════════════════════════════════════════════════════ */
    
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
    st.session_state.model_metrics = None

# ═══════════════════════════════════════════════════════════════════════════════
# SIDEBAR - DATA SELECTION & MODEL CONFIGURATION
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
        index=2,  # Default 3 years
        help="More data = more stable model, but older patterns"
    )
    
    # Data frequency
    frequency = st.radio(
        "Data Frequency",
        ["Daily", "Weekly", "Monthly"],
        index=0,
        help="Higher frequency = more observations, more noise"
    )
    
    st.markdown("---")
    st.markdown(f"### {SIDEBAR_SECTIONS['model_config']}")
    
    # Price transformation
    transformation = st.radio(
        "Price Transformation",
        list(["Price Level", "Log Prices", "Log Returns", "Percentage Returns"]),
        index=0,
        help="Log returns reduce heteroscedasticity"
    )
    
    # Model selection mode
    model_mode = st.radio(
        "Model Selection",
        ["Manual ARIMA", "Auto ARIMA"],
        index=0,
        help="Manual: specify (p,d,q) | Auto: uses AIC to find best"
    )
    
    # ARIMA parameters
    if model_mode == "Manual ARIMA":
        st.write("**Set ARIMA Parameters (p, d, q)**")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            p = st.slider("p (AR Order)", 0, 5, DEFAULT_P, help="Auto-Regressive terms")
        with col2:
            d = st.slider("d (Differencing)", 0, 2, DEFAULT_D, help="Differencing order")
        with col3:
            q = st.slider("q (MA Order)", 0, 5, DEFAULT_Q, help="Moving Average terms")
    
    else:  # Auto ARIMA
        st.info("ℹ️ Auto ARIMA will automatically find optimal (p,d,q) using AIC criterion")
        p, d, q = None, None, None
    
    st.markdown("---")
    st.markdown(f"### {SIDEBAR_SECTIONS['forecast_settings']}")
    
    # Forecast horizon
    forecast_horizon = st.slider(
        "Forecast Horizon (Days)",
        min_value=1,
        max_value=60,
        value=DEFAULT_FORECAST_HORIZON,
        help="How many days ahead to forecast"
    )
    
    # Confidence level
    confidence_level = st.selectbox(
        "Confidence Level",
        ["80%", "90%", "95%", "99%"],
        index=2,  # Default 95%
        help="Confidence interval for forecast bands"
    )
    
    # Train/test split
    train_pct = st.slider(
        "Training Data %",
        min_value=60,
        max_value=95,
        value=int(DEFAULT_TRAIN_PCT * 100),
        step=5,
        help="% of data for model training (rest for testing)"
    )
    
    st.markdown("---")
    
    # Refresh button
    refresh_button = st.button(
        f"🔄 FETCH DATA & RUN MODEL",
        use_container_width=True,
        key="refresh_button",
        help="Click to fetch data and run ARIMA model"
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
# DATA FETCHING & MODEL EXECUTION
# ═══════════════════════════════════════════════════════════════════════════════

if refresh_button:
    with st.spinner(f"⏳ Fetching data for {ticker}..."):
        try:
            # Initialize data loader
            loader = DataLoader()
            
            # Calculate date range
            end_date = datetime.now().strftime("%Y-%m-%d")
            start_date = (datetime.now() - timedelta(days=lookback_years*365)).strftime("%Y-%m-%d")
            
            # Map frequency
            interval_map = {"Daily": "1d", "Weekly": "1wk", "Monthly": "1mo"}
            interval = interval_map.get(frequency, "1d")
            
            # Fetch data
            st.session_state.data = loader.fetch_from_yfinance(
                ticker=ticker,
                start_date=start_date,
                end_date=end_date,
                interval=interval
            )
            
            # Validate data
            is_valid, errors = loader.validate_data(st.session_state.data)
            
            if not is_valid:
                st.error(f"❌ Data validation failed: {errors}")
            else:
                st.session_state.series = loader.prepare_series(st.session_state.data, column="Close")
                st.session_state.data_loaded = True
                st.success(f"✅ Data fetched: {len(st.session_state.series)} observations")
        
        except Exception as e:
            st.error(f"❌ Error fetching data: {str(e)}")
            st.session_state.data_loaded = False

# ═══════════════════════════════════════════════════════════════════════════════
# MODEL FITTING
# ═══════════════════════════════════════════════════════════════════════════════

if st.session_state.data_loaded and refresh_button:
    with st.spinner("⏳ Fitting ARIMA model..."):
        try:
            # Create ARIMA model
            st.session_state.arima_model = ARIMAModel(st.session_state.series, name=ticker)
            
            # Fit model
            if model_mode == "Auto ARIMA":
                order, aic, bic = st.session_state.arima_model.auto_select_parameters(
                    max_p=5, max_d=2, max_q=5
                )
                st.session_state.arima_model.fit(order)
                st.success(f"✅ Auto ARIMA selected: ARIMA{order}")
            else:
                st.session_state.arima_model.fit((p, d, q))
                st.success(f"✅ Manual ARIMA fitted: ARIMA({p},{d},{q})")
            
            st.session_state.model_fitted = True
            
            # Generate forecast
            alpha_map = {"80%": 0.20, "90%": 0.10, "95%": 0.05, "99%": 0.01}
            alpha = alpha_map.get(confidence_level, 0.05)
            
            st.session_state.forecast_df = st.session_state.arima_model.forecast(
                steps=forecast_horizon,
                alpha=alpha
            )
            st.session_state.forecast_generated = True
            st.success(f"✅ Forecast generated for {forecast_horizon} days")
        
        except Exception as e:
            st.error(f"❌ Error fitting model: {str(e)}")
            st.session_state.model_fitted = False

# ═══════════════════════════════════════════════════════════════════════════════
# MAIN CONTENT - METRICS & ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════

# Section 1: Basic Metrics
st.markdown("### 📊 Data Summary")

col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    st.metric(
        label="Ticker",
        value=ticker,
        help="Selected security"
    )

with col2:
    st.metric(
        label="Lookback",
        value=f"{lookback_years}y",
        help="Historical data period"
    )

with col3:
    st.metric(
        label="Model Mode",
        value="Manual" if model_mode == "Manual ARIMA" else "Auto",
        help="ARIMA parameter selection method"
    )

with col4:
    st.metric(
        label="Forecast Days",
        value=forecast_horizon,
        help="Forecast horizon"
    )

with col5:
    st.metric(
        label="Train/Test",
        value=f"{train_pct}% / {100-train_pct}%",
        help="Data split ratio"
    )

st.markdown("---")

# Section 2: Analysis Tabs
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
    
    if st.session_state.data_loaded and st.session_state.forecast_generated:
        try:
            # Create Plotly figure
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
            if st.session_state.arima_model.fitted_model is not None:
                fitted_values = st.session_state.arima_model.fitted_model.fittedvalues
                fig.add_trace(go.Scatter(
                    x=fitted_values.index,
                    y=fitted_values.values,
                    mode='lines',
                    name='Fitted Values',
                    line=dict(color='green', width=2, dash='dash'),
                    hovertemplate='<b>Date</b>: %{x|%Y-%m-%d}<br><b>Fitted</b>: ₹%{y:.2f}<extra></extra>'
                ))
            
            # Forecast
            forecast_data = st.session_state.forecast_df
            fig.add_trace(go.Scatter(
                x=forecast_data.index,
                y=forecast_data['forecast'].values,
                mode='lines',
                name='Forecast',
                line=dict(color=GOLD_COLOR, width=2),
                hovertemplate='<b>Date</b>: %{x|%Y-%m-%d}<br><b>Forecast</b>: ₹%{y:.2f}<extra></extra>'
            ))
            
            # Confidence interval (upper)
            fig.add_trace(go.Scatter(
                x=forecast_data.index,
                y=forecast_data['upper_ci'].values,
                fill=None,
                mode='lines',
                line_color='rgba(0,0,0,0)',
                showlegend=False,
                hoverinfo='skip'
            ))
            
            # Confidence interval (lower & fill)
            fig.add_trace(go.Scatter(
                x=forecast_data.index,
                y=forecast_data['lower_ci'].values,
                fill='tonexty',
                mode='lines',
                line_color='rgba(0,0,0,0)',
                name='95% Confidence Interval',
                fillcolor=f'rgba(255,215,0,0.2)',
                hovertemplate='<b>Date</b>: %{x|%Y-%m-%d}<br><b>CI</b>: [%{y:.2f}, %{y:.2f}]<extra></extra>'
            ))
            
            # Layout
            fig.update_layout(
                title=f'<b>{ALL_TICKERS[ticker]} - ARIMA Forecast</b>',
                xaxis_title='Date',
                yaxis_title='Price (₹)',
                template='plotly_white',
                hovermode='x unified',
                height=600,
                font=dict(family="Arial", size=12),
                margin=dict(l=50, r=50, t=80, b=50)
            )
            
            st.plotly_chart(fig, use_container_width=True)
            st.success(f"✓ Interactive chart generated for {ticker}")
        
        except Exception as e:
            st.error(f"❌ Error generating chart: {str(e)}")
    
    else:
        if not st.session_state.data_loaded:
            st.warning("⚠️ Click '🔄 FETCH DATA & RUN MODEL' to load data and generate chart")
        else:
            st.info("📊 Chart will appear after model fitting completes")

# ───────────────────────────────────────────────────────────────────────────────
# TAB 2: RESIDUAL DIAGNOSTICS
# ───────────────────────────────────────────────────────────────────────────────
with tab2:
    st.subheader("Residual Analysis - Box-Jenkins Diagnostics")
    st.info("""
    📊 **Four-Panel Diagnostic Grid:**
    
    1. **ACF Plot (Top-Left)**: Auto-correlation function
       - Shows if residuals are white noise
       - Should stay within 95% confidence bounds
    
    2. **PACF Plot (Top-Right)**: Partial auto-correlation
       - Identifies lag dependencies
       - Helps select AR order (p)
    
    3. **Histogram (Bottom-Left)**: Distribution of residuals
       - Should be roughly normal (bell curve)
       - Check for skewness and heavy tails
    
    4. **Q-Q Plot (Bottom-Right)**: Normality test
       - Points on diagonal = normally distributed residuals
       - Deviations indicate non-normal behavior
    """)
    
    if st.session_state.model_fitted:
        try:
            # Create diagnostic plots
            st.session_state.arima_model.plot_diagnostics(figsize=(14, 10))
            st.pyplot(plt.gcf(), use_container_width=True)
            st.success(f"✓ Diagnostics calculated for {ticker}")
        except Exception as e:
            st.error(f"❌ Error generating diagnostics: {str(e)}")
    else:
        st.warning("⚠️ Click '🔄 FETCH DATA & RUN MODEL' to generate diagnostics")

# ───────────────────────────────────────────────────────────────────────────────
# TAB 3: MODEL METRICS
# ───────────────────────────────────────────────────────────────────────────────
with tab3:
    st.subheader("Model Fit & Performance Metrics")
    
    if st.session_state.model_fitted:
        try:
            # Get model metrics
            residuals = st.session_state.arima_model.fitted_model.resid
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**📋 Model Fit Metrics**")
                st.metric("AIC", f"{st.session_state.arima_model.aic:.2f}", 
                         help="Akaike Information Criterion - lower is better")
                st.metric("BIC", f"{st.session_state.arima_model.bic:.2f}", 
                         help="Bayesian Information Criterion - lower is better")
                st.metric("R-Squared", f"{st.session_state.arima_model.fitted_model.rsquared:.4f}", 
                         help="Goodness of fit measure")
            
            with col2:
                st.write("**📊 Forecast Accuracy (Residuals)**")
                rmse = np.sqrt(np.mean(residuals**2))
                st.metric("RMSE", f"₹{rmse:.2f}", help="Root Mean Squared Error")
                mae = np.mean(np.abs(residuals))
                st.metric("MAE", f"₹{mae:.2f}", help="Mean Absolute Error")
                mape = np.mean(np.abs(residuals / st.session_state.series.mean())) * 100
                st.metric("MAPE", f"{mape:.2f}%", help="Mean Absolute Percentage Error")
            
            st.markdown("---")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**✓ Residual Statistics**")
                st.info(f"""
                - **Mean**: {np.mean(residuals):.4f} (should be ~0)
                - **Std Dev**: {np.std(residuals):.4f}
                - **Min**: {np.min(residuals):.4f}
                - **Max**: {np.max(residuals):.4f}
                """)
            
            with col2:
                st.write("**⚙️ Model Configuration**")
                if st.session_state.arima_model.parameters:
                    order = st.session_state.arima_model.parameters.get('order', (p, d, q))
                    st.metric("ARIMA Order", f"{order}", help="Selected parameters")
                else:
                    st.metric("ARIMA Order", f"({p},{d},{q})", help="Specified parameters")
                
                st.metric("Train Size", f"{int(len(st.session_state.series) * train_pct / 100)}", 
                         help="# observations in training set")
                st.metric("Test Size", f"{int(len(st.session_state.series) * (100-train_pct) / 100)}", 
                         help="# observations in test set")
        
        except Exception as e:
            st.error(f"❌ Error calculating metrics: {str(e)}")
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
    - **Forecast**: Point forecast (mean prediction)
    - **Lower CI**: Confidence interval lower bound
    - **Upper CI**: Confidence interval upper bound
    
    **Interpretation:**
    If forecast = ₹2,500 with CI [₹2,450 - ₹2,550]:
    - We predict ₹2,500 on that date
    - We are 95% confident the actual price will be between ₹2,450-₹2,550
    """)
    
    if st.session_state.forecast_generated:
        try:
            # Format forecast dataframe for display
            forecast_display = st.session_state.forecast_df.copy()
            forecast_display.index = forecast_display.index.strftime('%Y-%m-%d')
            forecast_display.columns = ['Forecast (₹)', 'Lower CI (₹)', 'Upper CI (₹)']
            
            # Round to 2 decimal places
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
            
            st.success("✓ Forecast generated - table shows point estimates and confidence intervals")
        
        except Exception as e:
            st.error(f"❌ Error displaying forecast: {str(e)}")
    else:
        st.warning("⚠️ Click '🔄 FETCH DATA & RUN MODEL' to generate forecast table")

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
    
    ### 📊 Forecast Interpretation:
    
    **Point Forecast**: Most likely value
    **95% CI**: 95% confident actual will fall within bounds
    **Wider CI**: Higher uncertainty (consider risk!)
    **Narrower CI**: More confidence in forecast
    
    ---
    
    **📖 Further Reading:**
    - Box, G. E. P., & Jenkins, G. M. (1970). *Time Series Analysis, Forecasting and Control*
    - Brockwell, P. J., & Davis, R. A. (2016). *Introduction to Time Series and Forecasting*
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
# DEBUG MODE (Development)
# ═══════════════════════════════════════════════════════════════════════════════

if st.sidebar.checkbox("🔧 Show Debug Info", key="debug_checkbox"):
    st.sidebar.markdown("---")
    st.sidebar.write("**DEBUG INFORMATION**")
    st.sidebar.write(f"Ticker: `{ticker}`")
    st.sidebar.write(f"Lookback: `{lookback_years}y`")
    st.sidebar.write(f"Transformation: `{transformation}`")
    st.sidebar.write(f"Model Mode: `{model_mode}`")
    st.sidebar.write(f"Data Loaded: `{st.session_state.data_loaded}`")
    st.sidebar.write(f"Model Fitted: `{st.session_state.model_fitted}`")
    st.sidebar.write(f"Forecast Generated: `{st.session_state.forecast_generated}`")
    
    if model_mode == "Manual ARIMA":
        st.sidebar.write(f"ARIMA Order: `({p},{d},{q})`")
    
    st.sidebar.write(f"Forecast Horizon: `{forecast_horizon} days`")
    st.sidebar.write(f"Confidence Level: `{confidence_level}`")
    st.sidebar.write(f"Train/Test Split: `{train_pct}% / {100-train_pct}%`")
    st.sidebar.write(f"Refresh Button Clicked: `{refresh_button}`")
