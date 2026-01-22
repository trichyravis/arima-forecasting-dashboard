
"""
═══════════════════════════════════════════════════════════════════════════════
ARIMA FORECASTING DASHBOARD - FINAL PRODUCTION VERSION
The Mountain Path - World of Finance
Real-Time Box-Jenkins Time Series Forecasting for Indian Equities

Prof. V. Ravichandran
28+ Years Corporate Finance & Banking Experience
10+ Years Academic Excellence

FINAL COMPLETED VERSION - January 1st, 2026
═══════════════════════════════════════════════════════════════════════════════
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from typing import Tuple, Dict, Optional
import warnings

warnings.filterwarnings('ignore')

# TRY IMPORTS
try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except:
    YFINANCE_AVAILABLE = False

try:
    from statsmodels.tsa.arima.model import ARIMA
    from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
    from statsmodels.tsa.stattools import adfuller, kpss
    STATSMODELS_AVAILABLE = True
except:
    STATSMODELS_AVAILABLE = False

try:
    import pmdarima as pm
    PMDARIMA_AVAILABLE = True
except:
    PMDARIMA_AVAILABLE = False

# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

DARK_BLUE = "#003366"
LIGHT_BLUE = "#004d80"
GOLD = "#FFD700"

# TICKERS
INDICES = {"^NSEI": "NIFTY 50", "^NSEBANK": "BANKNIFTY", "^NIFTYNXT50": "NIFTY NEXT 50"}
STOCKS = {"TCS.NS": "TCS", "INFY.NS": "INFOSYS", "HDFC.NS": "HDFC", "RELIANCE.NS": "RELIANCE", "WIPRO.NS": "WIPRO"}
CRYPTO = {"BTC-USD": "Bitcoin", "EURINR=X": "EUR/INR", "GBPINR=X": "GBP/INR"}

ALL_TICKERS = {**INDICES, **STOCKS, **CRYPTO}

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE CONFIG
# ═══════════════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="ARIMA Forecasting Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ═══════════════════════════════════════════════════════════════════════════════
# CUSTOM CSS
# ═══════════════════════════════════════════════════════════════════════════════

st.markdown(f"""
    <style>
    .hero-header {{
        background: linear-gradient(135deg, {DARK_BLUE} 0%, {LIGHT_BLUE} 100%);
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        margin-bottom: 2rem;
        border: 3px solid {DARK_BLUE};
    }}
    
    .hero-header h1 {{
        color: white;
        font-size: 36px;
        margin: 0;
        font-weight: 900;
        letter-spacing: 2px;
    }}
    
    .hero-header p {{
        color: #E0F0FF;
        font-size: 18px;
        margin: 10px 0 0 0;
    }}
    
    .sidebar-section {{
        background: rgba(0, 51, 102, 0.1);
        padding: 15px;
        border-radius: 10px;
        margin: 10px 0;
    }}
    
    [data-testid="stSidebar"] {{
        background: linear-gradient(135deg, {DARK_BLUE} 0%, {LIGHT_BLUE} 100%);
    }}
    
    [data-testid="stSidebar"] * {{
        color: white !important;
    }}
    
    .metric-card {{
        background: linear-gradient(135deg, {DARK_BLUE} 0%, {LIGHT_BLUE} 100%);
        color: white;
        padding: 20px;
        border-radius: 10px;
        text-align: center;
    }}
    </style>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# HERO HEADER
# ═══════════════════════════════════════════════════════════════════════════════

st.markdown("""
    <div class="hero-header">
        <h1>📊 THE MOUNTAIN PATH • ARIMA FORECASTING</h1>
        <p>Box-Jenkins Time Series Analysis</p>
        <p style='font-size: 14px; color: #D0E8FF;'>Interactive Forecasting for Indian Equities</p>
    </div>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# SESSION STATE
# ═══════════════════════════════════════════════════════════════════════════════

if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
    st.session_state.model_fitted = False
    st.session_state.forecast_generated = False
    st.session_state.series = None
    st.session_state.model = None
    st.session_state.forecast_df = None
    st.session_state.fitted = None
    st.session_state.residuals = None
    st.session_state.last_ticker = None

# ═══════════════════════════════════════════════════════════════════════════════
# FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

@st.cache_data
def fetch_data(ticker, days=1000):
    """Fetch data from Yahoo Finance"""
    try:
        if not YFINANCE_AVAILABLE:
            return None
        end = datetime.now()
        start = end - timedelta(days=days)
        data = yf.download(ticker, start=start, end=end, progress=False)
        return data
    except:
        return None

def fit_arima(series, p, d, q):
    """Fit ARIMA model"""
    try:
        if not STATSMODELS_AVAILABLE:
            return None
        model = ARIMA(series, order=(p, d, q))
        result = model.fit()
        return result
    except:
        return None

def forecast_arima(model, steps, alpha=0.05):
    """Generate forecast"""
    try:
        forecast = model.get_forecast(steps=steps)
        forecast_df = forecast.conf_int(alpha=alpha)
        forecast_df['forecast'] = forecast.predicted_mean
        return forecast_df.iloc[:, [1, 0, 2]]
    except:
        return None

def calculate_metrics(residuals):
    """Calculate residual metrics"""
    rmse = np.sqrt(np.mean(residuals**2))
    mae = np.mean(np.abs(residuals))
    mape = np.mean(np.abs(residuals / (np.abs(residuals) + 1e-10))) * 100
    return {"RMSE": rmse, "MAE": mae, "MAPE": mape}

# ═══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("---")
    st.markdown("### 📊 DATA SELECTION")
    
    ticker = st.selectbox(
        "Select Ticker",
        list(ALL_TICKERS.keys()),
        format_func=lambda x: f"{x} - {ALL_TICKERS[x]}"
    )
    
    lookback = st.selectbox(
        "Years of Data",
        [1, 2, 3, 5, 7, 10],
        index=2
    )
    
    st.markdown("---")
    st.markdown("### ⚙️ MODEL CONFIGURATION")
    
    model_mode = st.radio(
        "Model Selection",
        ["Manual ARIMA", "Auto ARIMA"],
        index=0
    )
    
    if model_mode == "Manual ARIMA":
        col1, col2, col3 = st.columns(3)
        with col1:
            p = st.slider("p", 0, 5, 1)
        with col2:
            d = st.slider("d", 0, 2, 1)
        with col3:
            q = st.slider("q", 0, 5, 1)
    else:
        st.info("Auto ARIMA will find optimal parameters")
        p, d, q = None, None, None
    
    st.markdown("---")
    st.markdown("### 🔮 FORECAST SETTINGS")
    
    forecast_steps = st.slider(
        "Forecast Days",
        min_value=1,
        max_value=60,
        value=10
    )
    
    confidence = st.selectbox(
        "Confidence Level",
        ["80%", "90%", "95%", "99%"],
        index=2
    )
    
    st.markdown("---")
    
    run_button = st.button("🔄 FETCH DATA & RUN MODEL", use_container_width=True)
    
    st.markdown("---")
    st.markdown("### About")
    st.markdown("""
    Box-Jenkins ARIMA methodology for time series forecasting.
    
    **Key Features:**
    - Real-time data fetching
    - ARIMA parameter optimization
    - Comprehensive diagnostics
    - Interactive visualizations
    """)
    
    st.markdown("---")
    st.markdown(f"**{AUTHOR_INFO}**" if 'AUTHOR_INFO' in globals() else "")
    st.markdown("**Prof. V. Ravichandran**")
    st.markdown("*28+ Years Corporate Finance Experience*")

# ═══════════════════════════════════════════════════════════════════════════════
# MAIN LOGIC
# ═══════════════════════════════════════════════════════════════════════════════

if run_button:
    # FETCH DATA
    with st.spinner(f"⏳ Fetching data for {ticker}..."):
        data = fetch_data(ticker, days=lookback*365)
        
        if data is not None and not data.empty:
            series = data['Close'].dropna()
            st.session_state.series = series
            st.session_state.data_loaded = True
            st.session_state.last_ticker = ticker
            st.success(f"✅ Data fetched: {len(series)} observations")
        else:
            st.error("❌ Could not fetch data")
            st.stop()
    
    # FIT MODEL
    if st.session_state.data_loaded:
        with st.spinner("⏳ Fitting ARIMA model..."):
            if model_mode == "Manual ARIMA":
                model = fit_arima(st.session_state.series, p, d, q)
                if model:
                    st.session_state.model = model
                    st.session_state.model_fitted = True
                    st.success(f"✅ ARIMA({p},{d},{q}) fitted")
            else:
                if PMDARIMA_AVAILABLE:
                    try:
                        auto = pm.auto_arima(st.session_state.series, seasonal=False)
                        st.session_state.model = auto
                        st.session_state.model_fitted = True
                        st.success(f"✅ Auto ARIMA: {auto.order}")
                    except:
                        model = fit_arima(st.session_state.series, 1, 1, 1)
                        st.session_state.model = model
                        st.session_state.model_fitted = True
    
    # GENERATE FORECAST
    if st.session_state.model_fitted:
        with st.spinner("⏳ Generating forecast..."):
            alpha_map = {"80%": 0.20, "90%": 0.10, "95%": 0.05, "99%": 0.01}
            alpha = alpha_map[confidence]
            
            forecast = forecast_arima(st.session_state.model, forecast_steps, alpha)
            if forecast is not None:
                st.session_state.forecast_df = forecast
                st.session_state.forecast_generated = True
                st.session_state.fitted = st.session_state.model.fittedvalues
                st.session_state.residuals = st.session_state.model.resid
                st.success("✅ Forecast generated")

# ═══════════════════════════════════════════════════════════════════════════════
# DISPLAY RESULTS
# ═══════════════════════════════════════════════════════════════════════════════

if st.session_state.data_loaded:
    # METRICS
    st.markdown("### 📊 Data Summary")
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric("Ticker", st.session_state.last_ticker)
    with col2:
        st.metric("Lookback", f"{lookback}y")
    with col3:
        st.metric("Observations", len(st.session_state.series))
    with col4:
        st.metric("Forecast Days", forecast_steps)
    with col5:
        st.metric("Model", "Manual" if model_mode == "Manual ARIMA" else "Auto")
    
    st.markdown("---")
    
    # TABS
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📈 Time Series & Forecast",
        "📊 Residual Diagnostics",
        "📋 Model Metrics",
        "🔮 Forecast Results",
        "❓ Help & Guide"
    ])
    
    # TAB 1: TIME SERIES
    with tab1:
        st.subheader("Time Series with Forecast")
        st.info("""
        **Chart Components:**
        - Blue: Historical prices
        - Green: Fitted values
        - Orange: Forecast
        - Shaded: Confidence interval
        """)
        
        if st.session_state.forecast_generated:
            fig = go.Figure()
            
            # Historical
            fig.add_trace(go.Scatter(
                x=st.session_state.series.index,
                y=st.session_state.series.values,
                name="Historical",
                line=dict(color=DARK_BLUE, width=2)
            ))
            
            # Fitted
            fig.add_trace(go.Scatter(
                x=st.session_state.fitted.index,
                y=st.session_state.fitted.values,
                name="Fitted",
                line=dict(color="green", width=2, dash="dash")
            ))
            
            # Forecast
            fig.add_trace(go.Scatter(
                x=st.session_state.forecast_df.index,
                y=st.session_state.forecast_df.iloc[:, 2].values,
                name="Forecast",
                line=dict(color=GOLD, width=2)
            ))
            
            # CI Upper
            fig.add_trace(go.Scatter(
                x=st.session_state.forecast_df.index,
                y=st.session_state.forecast_df.iloc[:, 1].values,
                fill=None,
                showlegend=False,
                line_color="rgba(0,0,0,0)"
            ))
            
            # CI Lower
            fig.add_trace(go.Scatter(
                x=st.session_state.forecast_df.index,
                y=st.session_state.forecast_df.iloc[:, 0].values,
                fill="tonexty",
                name="95% CI",
                fillcolor="rgba(255,215,0,0.2)",
                line_color="rgba(0,0,0,0)"
            ))
            
            fig.update_layout(
                title=f"{ALL_TICKERS[st.session_state.last_ticker]} - ARIMA Forecast",
                xaxis_title="Date",
                yaxis_title="Price",
                template="plotly_white",
                height=600
            )
            
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Click button to generate chart")
    
    # TAB 2: DIAGNOSTICS
    with tab2:
        st.subheader("Residual Diagnostics")
        
        if st.session_state.model_fitted and st.session_state.residuals is not None:
            try:
                fig, axes = plt.subplots(2, 2, figsize=(14, 10))
                
                plot_acf(st.session_state.residuals, lags=40, ax=axes[0, 0])
                axes[0, 0].set_title("ACF")
                
                plot_pacf(st.session_state.residuals, lags=40, ax=axes[0, 1])
                axes[0, 1].set_title("PACF")
                
                axes[1, 0].hist(st.session_state.residuals, bins=30)
                axes[1, 0].set_title("Histogram")
                
                from scipy import stats
                stats.probplot(st.session_state.residuals, dist="norm", plot=axes[1, 1])
                axes[1, 1].set_title("Q-Q Plot")
                
                plt.tight_layout()
                st.pyplot(fig)
            except Exception as e:
                st.error(f"Error: {str(e)}")
        else:
            st.warning("Click button to generate diagnostics")
    
    # TAB 3: METRICS
    with tab3:
        st.subheader("Model Metrics")
        
        if st.session_state.model_fitted:
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Model Fit**")
                st.metric("AIC", f"{st.session_state.model.aic:.2f}")
                st.metric("BIC", f"{st.session_state.model.bic:.2f}")
            
            with col2:
                st.write("**Forecast Accuracy**")
                if st.session_state.residuals is not None:
                    metrics = calculate_metrics(st.session_state.residuals)
                    st.metric("RMSE", f"₹{metrics['RMSE']:.2f}")
                    st.metric("MAE", f"₹{metrics['MAE']:.2f}")
                    st.metric("MAPE", f"{metrics['MAPE']:.2f}%")
        else:
            st.warning("Click button to calculate metrics")
    
    # TAB 4: FORECAST
    with tab4:
        st.subheader(f"{forecast_steps}-Day Forecast")
        
        if st.session_state.forecast_generated:
            forecast_table = st.session_state.forecast_df.copy()
            forecast_table.index = forecast_table.index.strftime('%Y-%m-%d')
            forecast_table.columns = ['Lower CI', 'Upper CI', 'Forecast']
            forecast_table = forecast_table[['Forecast', 'Lower CI', 'Upper CI']].round(2)
            
            st.dataframe(forecast_table, use_container_width=True)
            
            csv = forecast_table.to_csv()
            st.download_button(
                "📥 Download CSV",
                csv,
                f"{st.session_state.last_ticker}_forecast.csv",
                "text/csv"
            )
        else:
            st.warning("Click button to generate forecast")
    
    # TAB 5: HELP
    with tab5:
        st.subheader("ARIMA Methodology Guide")
        st.markdown("""
        ## Box-Jenkins ARIMA
        
        **ARIMA(p,d,q):**
        - **p**: AR order (previous values)
        - **d**: Differencing (stationarity)
        - **q**: MA order (previous errors)
        
        **6 Steps:**
        1. Data prep
        2. Stationarity test (ADF)
        3. ACF/PACF analysis
        4. Parameter estimation
        5. Diagnostic checks
        6. Forecasting
        
        **Good Model Signs:**
        ✓ ACF/PACF within bounds
        ✓ White noise residuals
        ✓ MAPE < 5%
        ✓ Ljung-Box p > 0.05
        """)

else:
    st.warning("⏳ Click '🔄 FETCH DATA & RUN MODEL' to start")

# ═══════════════════════════════════════════════════════════════════════════════
# FOOTER
# ═══════════════════════════════════════════════════════════════════════════════

st.markdown("---")
st.markdown("""
    <div style='text-align: center; color: #999; font-size: 0.9em;'>
        <p><strong>The Mountain Path - World of Finance</strong></p>
        <p>Prof. V. Ravichandran | 28+ Years Corporate Finance Experience</p>
    </div>
""", unsafe_allow_html=True)
