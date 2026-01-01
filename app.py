
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import yfinance as yf
import warnings
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.graphics.tsaplots import plot_acf
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
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

BRAND_NAME = "The Mountain Path - World of Finance"
HERO_TITLE = "ARIMA FORECASTING DASHBOARD"
HERO_SUBTITLE = "Real-Time Box-Jenkins Time Series Forecasting for Indian Equities"
HERO_DESCRIPTION = "Prof. V. Ravichandran | 28+ Years Corporate Finance & Banking Experience"

ALL_TICKERS = {
    "^NSEI": "NIFTY 50", 
    "^BSESN": "SENSEX", 
    "RELIANCE.NS": "Reliance Industries", 
    "TCS.NS": "TCS", 
    "HDFCBANK.NS": "HDFC Bank",
    "INFY.NS": "Infosys",
    "ICICIBANK.NS": "ICICI Bank"
}

st.set_page_config(page_title="ARIMA Dashboard - The Mountain Path", page_icon="🏔️", layout="wide")

# ═══════════════════════════════════════════════════════════════════════════════
# CSS FIXES (Sidebar Visibility & White Text)
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown(f"""
    <style>
    /* Hero Header */
    .hero-title {{ 
        background: linear-gradient(135deg, {DARK_BLUE} 0%, {LIGHT_BLUE} 100%); 
        padding: 2rem; border-radius: 20px; margin-bottom: 2rem; 
        box-shadow: 0 12px 30px rgba(0, 51, 102, 0.4); border: 4px solid {DARK_BLUE}; 
        color: white; text-align: center;
    }}
    
    /* Sidebar Styling */
    [data-testid="stSidebar"] {{ 
        background: linear-gradient(135deg, {DARK_BLUE} 0%, {LIGHT_BLUE} 100%) !important; 
    }}
    
    /* UNIVERSAL WHITE TEXT: Headers, Labels, and Radio Options */
    [data-testid="stSidebar"] h3, 
    [data-testid="stSidebar"] label, 
    [data-testid="stSidebar"] .stMarkdown p,
    [data-testid="stSidebar"] div[role="radiogroup"] {{ 
        color: white !important; 
        font-weight: 600 !important;
    }}

    /* Target specific radio button text labels */
    [data-testid="stSidebar"] div[data-testid="stWidgetLabel"] p,
    [data-testid="stSidebar"] div[data-testid="stMarkdownContainer"] p {{
        color: white !important;
    }}
    
    /* Target radio button item text specifically */
    [data-testid="stSidebar"] .st-ae div {{
        color: white !important;
    }}

    /* INPUT VISIBILITY: Keep text INSIDE dropdowns/inputs Dark for readability */
    div[data-baseweb="select"] > div,
    input {{ 
        color: {DARK_BLUE} !important; 
    }}

    /* Slider numbers fix */
    [data-testid="stSidebar"] .st-at {{
        color: white !important;
    }}

    /* Refresh Button Styling */
    .stButton>button {{
        background-color: {GOLD_COLOR} !important;
        color: {DARK_BLUE} !important;
        font-weight: bold !important;
        border-radius: 10px !important;
        border: none !important;
        width: 100%;
        margin-top: 1rem;
    }}
    </style>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# UI LAYOUT - HERO & SIDEBAR
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown(f"""
    <div class="hero-title">
        <h1>{HERO_TITLE}</h1>
        <p>{HERO_SUBTITLE}</p>
        <p>{HERO_DESCRIPTION}</p>
    </div>
""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("### 📊 Data Selection")
    ticker = st.selectbox("Select Ticker", options=list(ALL_TICKERS.keys()), format_func=lambda x: f"{x} - {ALL_TICKERS[x]}")
    lookback = st.selectbox("Years of Historical Data", [1, 2, 3, 5, 10], index=2)
    freq = st.radio("Data Frequency", ["Daily", "Weekly", "Monthly"])
    
    st.markdown("### ⚙️ Model Configuration")
    transformation = st.radio("Price Transformation", ["Price Level", "Log Prices", "Log Returns", "Percentage Returns"], index=2)
    model_mode = st.radio("Model Selection", ["Manual ARIMA", "Auto ARIMA"], index=1)
    
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
# MAIN PAGE METRICS
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown("### 📊 Current Selections")
m1, m2, m3, m4 = st.columns(4)
m1.metric("Security", ticker)
m2.metric("Lookback", f"{lookback}y")
m3.metric("Mode", model_mode)
m4.metric("Confidence", conf_level)

# ═══════════════════════════════════════════════════════════════════════════════
# MAIN LOGIC
# ═══════════════════════════════════════════════════════════════════════════════
tab1, tab2, tab3, tab4 = st.tabs(["📈 Forecast Chart", "🔍 Residual Diagnostics", "📊 Model Metrics", "📋 Export Data"])
results = None

if refresh_button:
    with st.spinner("Executing Box-Jenkins Methodology..."):
        data = yf.download(ticker, start=datetime.now()-timedelta(days=lookback*365))
        if not data.empty:
            raw_prices = data['Close'][ticker] if isinstance(data.columns, pd.MultiIndex) else data['Close']
            raw_prices = raw_prices.dropna()
            
            resample_map = {"Daily": "B", "Weekly": "W", "Monthly": "M"}
            raw_prices = raw_prices.resample(resample_map[freq]).last().ffill()
            
            if transformation == "Log Returns": train_series = np.log(raw_prices).diff().dropna()
            elif transformation == "Log Prices": train_series = np.log(raw_prices)
            elif transformation == "Percentage Returns": train_series = raw_prices.pct_change().dropna()
            else: train_series = raw_prices

            try:
                if model_mode == "Auto ARIMA":
                    model = pm.auto_arima(train_series, seasonal=False, stepwise=True)
                    fc_vals = model.predict(n_periods=forecast_horizon)
                    order, aic, resid = model.order, model.aic(), model.resid()
                else:
                    fit = ARIMA(train_series, order=(p, d, q)).fit()
                    fc_vals = fit.forecast(steps=forecast_horizon)
                    order, aic, resid = (p, d, q), fit.aic, fit.resid

                # Inversion logic
                last_p = raw_prices.iloc[-1]
                if transformation == "Log Returns": forecast_prices = last_p * np.exp(np.cumsum(fc_vals))
                elif transformation == "Log Prices": forecast_prices = np.exp(fc_vals)
                elif transformation == "Percentage Returns": forecast_prices = last_p * (1 + np.cumsum(fc_vals))
                else: forecast_prices = fc_vals
                
                f_dates = pd.date_range(raw_prices.index[-1], periods=forecast_horizon + 1, freq=resample_map[freq])[1:]
                fc_df = pd.DataFrame({"Forecasted Price": np.array(forecast_prices).flatten()}, index=f_dates)
                
                results = {"raw": raw_prices, "fc_df": fc_df, "order": order, "aic": aic, "resid": resid}
            except Exception as e:
                st.error(f"Computation Error: {e}")

# ═══════════════════════════════════════════════════════════════════════════════
# DISPLAY TABS
# ═══════════════════════════════════════════════════════════════════════════════
if results:
    with tab1:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=results["raw"].index, y=results["raw"], name="Historical Price", line=dict(color=DARK_BLUE)))
        fig.add_trace(go.Scatter(x=results["fc_df"].index, y=results["fc_df"]["Forecasted Price"], name="ARIMA Forecast", line=dict(color='orange', width=3)))
        st.plotly_chart(fig, use_container_width=True)
        
    with tab2:
        st.subheader("🔍 Residual Diagnostics")
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        axes[0, 0].plot(results["resid"]); axes[0, 0].set_title("Standardized Residuals")
        sns.histplot(results["resid"], kde=True, ax=axes[0, 1]); axes[0, 1].set_title("Residual Normality")
        plot_acf(results["resid"], ax=axes[1, 0], lags=20); axes[1, 0].set_title("Residual ACF")
        stats.probplot(results["resid"], dist="norm", plot=axes[1, 1]); axes[1, 1].set_title("Normal Q-Q Plot")
        plt.tight_layout()
        st.pyplot(fig)
        
        lb_test = acorr_ljungbox(results["resid"], lags=[10], return_df=True)
        p_val = lb_test['lb_pvalue'].values[0]
        if p_val > 0.05: st.success(f"✅ Ljung-Box Test (p={p_val:.3f}): Residuals are White Noise.")
        else: st.warning(f"⚠️ Ljung-Box Test (p={p_val:.3f}): Residuals have structure.")

    with tab3:
        st.metric("Optimal Model Order", str(results["order"]))
        st.metric("AIC Score", f"{results['aic']:.2f}")
        
    with tab4:
        st.dataframe(results["fc_df"].style.format("{:.2f}"), use_container_width=True)
        buffer = BytesIO()
        with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
            results["fc_df"].to_excel(writer, sheet_name='Forecast')
        st.download_button(label="📥 Download Excel Report", data=buffer.getvalue(), file_name=f"{ticker}_forecast.xlsx")
else:
    st.info("Adjust settings and click the button in the sidebar to generate a forecast.")

st.markdown("---")
st.markdown("### Box-Jenkins ARIMA Methodology")
