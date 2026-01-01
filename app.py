
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
from scipy import stats
import plotly.graph_objects as go
import pmdarima as pm
from io import BytesIO

# Try to import seaborn for better visuals, with fallback
try:
    import seaborn as sns
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False

# Suppress warnings
warnings.filterwarnings("ignore")

# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION & NIFTY 50 TICKERS
# ═══════════════════════════════════════════════════════════════════════════════
DARK_BLUE = "#003366"
LIGHT_BLUE = "#0066CC"
GOLD_COLOR = "#FFD700"
BRAND_NAME = "The Mountain Path - World of Finance"

NIFTY_50_STOCKS = {
    "^NSEI": "NIFTY 50 INDEX", "^BSESN": "SENSEX INDEX",
    "ADANIENT.NS": "Adani Enterprises", "ADANIPORTS.NS": "Adani Ports", "APOLLOHOSP.NS": "Apollo Hospitals",
    "ASIANPAINT.NS": "Asian Paints", "AXISBANK.NS": "Axis Bank", "BAJAJ-AUTO.NS": "Bajaj Auto",
    "BAJFINANCE.NS": "Bajaj Finance", "BAJAJFINSV.NS": "Bajaj Finserv", "BPCL.NS": "BPCL",
    "BHARTIARTL.NS": "Bharti Airtel", "BRITANNIA.NS": "Britannia", "CIPLA.NS": "Cipla",
    "COALINDIA.NS": "Coal India", "DIVISLAB.NS": "Divi's Lab", "DRREDDY.NS": "Dr. Reddy's",
    "EICHERMOT.NS": "Eicher Motors", "GRASIM.NS": "Grasim Industries", "HCLTECH.NS": "HCL Tech",
    "HDFCBANK.NS": "HDFC Bank", "HDFCLIFE.NS": "HDFC Life", "HEROMOTOCO.NS": "Hero MotoCorp",
    "HINDALCO.NS": "Hindalco", "HINDUNILVR.NS": "Hindustan Unilever", "ICICIBANK.NS": "ICICI Bank",
    "ITC.NS": "ITC", "INDUSINDBK.NS": "IndusInd Bank", "INFY.NS": "Infosys",
    "JSWSTEEL.NS": "JSW Steel", "KOTAKBANK.NS": "Kotak Mahindra", "LT.NS": "L&T",
    "LTIM.NS": "LTIMindtree", "M&M.NS": "M&M", "MARUTI.NS": "Maruti Suzuki",
    "NESTLEIND.NS": "Nestle India", "NTPC.NS": "NTPC", "ONGC.NS": "ONGC",
    "POWERGRID.NS": "Power Grid", "RELIANCE.NS": "Reliance Industries", "SBILIFE.NS": "SBI Life",
    "SBIN.NS": "State Bank of India", "SUNPHARMA.NS": "Sun Pharma", "TCS.NS": "TCS",
    "TATACONSUM.NS": "Tata Consumer", "TATAMOTORS.NS": "Tata Motors", "TATASTEEL.NS": "Tata Steel",
    "TECHM.NS": "Tech Mahindra", "TITAN.NS": "Titan Company", "ULTRACEMCO.NS": "UltraTech Cement",
    "UPL.NS": "UPL", "WIPRO.NS": "Wipro"
}

st.set_page_config(page_title="ARIMA Dashboard - The Mountain Path", page_icon="🏔️", layout="wide")

# ═══════════════════════════════════════════════════════════════════════════════
# CSS STYLING (Fixed for Sidebar & Radio Visibility)
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown(f"""
    <style>
    .hero-title {{ background: linear-gradient(135deg, {DARK_BLUE} 0%, {LIGHT_BLUE} 100%); padding: 2rem; border-radius: 20px; margin-bottom: 2rem; box-shadow: 0 12px 30px rgba(0, 51, 102, 0.4); border: 4px solid {DARK_BLUE}; color: white; text-align: center; }}
    [data-testid="stSidebar"] {{ background: linear-gradient(135deg, {DARK_BLUE} 0%, {LIGHT_BLUE} 100%) !important; }}
    [data-testid="stSidebar"] h3, [data-testid="stSidebar"] label, [data-testid="stSidebar"] .stMarkdown p, [data-testid="stSidebar"] div[role="radiogroup"] p, [data-testid="stSidebar"] div[data-testid="stWidgetLabel"] p {{ color: white !important; font-weight: 600 !important; }}
    [data-testid="stSidebar"] .st-ae div {{ color: white !important; }}
    div[data-baseweb="select"] > div, input {{ color: {DARK_BLUE} !important; }}
    [data-testid="stSidebar"] .st-at {{ color: white !important; }}
    .stButton>button {{ background-color: {GOLD_COLOR} !important; color: {DARK_BLUE} !important; font-weight: bold !important; border-radius: 10px !important; width: 100%; }}
    </style>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# UI LAYOUT - HERO & SIDEBAR
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown(f"<div class='hero-title'><h1>ARIMA FORECASTING DASHBOARD</h1><p>Real-Time Box-Jenkins Time Series Forecasting for Nifty 50 Stocks</p><p>Prof. V. Ravichandran | 28+ Years Finance Experience</p></div>", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("### 📊 Data Selection")
    ticker = st.selectbox("Select Security", options=list(NIFTY_50_STOCKS.keys()), format_func=lambda x: f"{x} - {NIFTY_50_STOCKS[x]}")
    lookback = st.selectbox("Years of Historical Data", [1, 2, 3, 5, 10], index=2)
    freq = st.radio("Data Frequency", ["Daily", "Weekly", "Monthly"])
    
    st.markdown("### ⚙️ Model Configuration")
    transformation = st.radio("Price Transformation", ["Price Level", "Log Prices", "Log Returns", "Percentage Returns"], index=2)
    model_mode = st.radio("Model Selection", ["Manual ARIMA", "Auto ARIMA"], index=1)
    
    p, d, q = 1, 1, 1
    if model_mode == "Manual ARIMA":
        col1, col2, col3 = st.columns(3)
        p = col1.slider("p (AR)", 0, 5, 1)
        d = col2.slider("d (I)", 0, 2, 1)
        q = col3.slider("q (MA)", 0, 5, 1)
    
    st.markdown("### 🔮 Forecast Settings")
    forecast_horizon = st.slider("Forecast Horizon (Periods)", 1, 60, 10)
    refresh_button = st.button("🔄 FETCH DATA & RUN MODEL")

    st.markdown("---")
    st.markdown("### Prof. V. Ravichandran")
    st.markdown("*28+ Years Finance Experience*")
    st.markdown(f"<a href='https://www.linkedin.com/in/trichyravis' target='_blank' style='display: block; padding: 0.5rem; background: #0077b5; color: white; text-align: center; text-decoration: none; border-radius: 5px; font-weight: bold;'>🔗 LinkedIn Profile</a>", unsafe_allow_html=True)

# Dashboard Summary Metrics
st.markdown("### 📊 Current Parameters")
m_cols = st.columns(4)
m_cols[0].metric("Security", ticker)
m_cols[1].metric("History", f"{lookback}y")
m_cols[2].metric("Model Mode", model_mode)
m_cols[3].metric("Frequency", freq)

# ═══════════════════════════════════════════════════════════════════════════════
# MODELING LOGIC
# ═══════════════════════════════════════════════════════════════════════════════
tab1, tab2, tab3, tab4, tab5 = st.tabs(["📈 Forecast", "🔍 Diagnostics", "📊 Metrics", "📋 Export", "📚 Educational Hub"])
results = None

if refresh_button:
    with st.spinner("Processing Box-Jenkins Analysis..."):
        data = yf.download(ticker, start=datetime.now()-timedelta(days=lookback*365))
        if not data.empty:
            raw_prices = data['Close'][ticker] if isinstance(data.columns, pd.MultiIndex) else data['Close']
            raw_prices = raw_prices.dropna()
            res_map = {"Daily": "B", "Weekly": "W", "Monthly": "M"}
            raw_prices = raw_prices.resample(res_map[freq]).last().ffill()
            
            if transformation == "Log Returns": train_series = np.log(raw_prices).diff().dropna()
            elif transformation == "Log Prices": train_series = np.log(raw_prices)
            elif transformation == "Percentage Returns": train_series = raw_prices.pct_change().dropna()
            else: train_series = raw_prices

            try:
                if model_mode == "Auto ARIMA":
                    model = pm.auto_arima(train_series, seasonal=False, stepwise=True)
                    fc_vals = model.predict(n_periods=forecast_horizon)
                    order, aic, resid = model.order, model.aic(), model.resid()
                    fit_obj = model
                else:
                    fit = ARIMA(train_series, order=(p, d, q)).fit()
                    fc_vals = fit.forecast(steps=forecast_horizon)
                    order, aic, resid = (p, d, q), fit.aic, fit.resid
                    fit_obj = fit

                last_p = raw_prices.iloc[-1]
                if transformation == "Log Returns": forecast_prices = last_p * np.exp(np.cumsum(fc_vals))
                elif transformation == "Log Prices": forecast_prices = np.exp(fc_vals)
                elif transformation == "Percentage Returns": forecast_prices = last_p * (1 + np.cumsum(fc_vals))
                else: forecast_prices = fc_vals
                
                f_dates = pd.date_range(raw_prices.index[-1], periods=forecast_horizon + 1, freq=res_map[freq])[1:]
                fc_df = pd.DataFrame({"Forecasted Price": np.array(forecast_prices).flatten()}, index=f_dates)
                results = {"raw": raw_prices, "fc_df": fc_df, "order": order, "aic": aic, "resid": resid, "fit_obj": fit_obj, "train_series": train_series}
            except Exception as e:
                st.error(f"Computation Error: {e}")

# ═══════════════════════════════════════════════════════════════════════════════
# OUTPUT DISPLAY
# ═══════════════════════════════════════════════════════════════════════════════
if results:
    with tab1:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=results["raw"].index, y=results["raw"], name="Historical Price", line=dict(color=DARK_BLUE)))
        fig.add_trace(go.Scatter(x=results["fc_df"].index, y=results["fc_df"]["Forecasted Price"], name="ARIMA Forecast", line=dict(color='orange', width=3)))
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        st.subheader("🔍 Residual Diagnostics")
        fig_diag, axes = plt.subplots(2, 2, figsize=(12, 8))
        axes[0, 0].plot(results["resid"]); axes[0, 0].set_title("Standardized Residuals")
        if HAS_SEABORN: sns.histplot(results["resid"], kde=True, ax=axes[0, 1])
        plot_acf(results["resid"], ax=axes[1, 0], lags=min(20, len(results["resid"])//2)); axes[1, 0].set_title("Residual ACF")
        stats.probplot(results["resid"], dist="norm", plot=axes[1, 1]); axes[1, 1].set_title("Normal Q-Q Plot")
        plt.tight_layout(); st.pyplot(fig_diag)

    with tab3:
        st.subheader("📊 Comprehensive Model Metrics")
        c1, c2, c3 = st.columns(3)
        
        def get_stat(obj, attr, is_method=True):
            if hasattr(obj, attr):
                val = getattr(obj, attr)
                return val() if is_method and callable(val) else val
            return None

        llf = get_stat(results["fit_obj"], "llf", False)
        if llf is None: llf = get_stat(results["fit_obj"], "loglikelihood", True)
        bic = get_stat(results["fit_obj"], "bic", True)
        if bic is None: bic = get_stat(results["fit_obj"], "bic", False)

        c1.markdown("#### Selection Criteria")
        c1.metric("Optimal Order", str(results["order"]))
        c1.metric("AIC", f"{results['aic']:.2f}")
        if bic: c1.metric("BIC", f"{bic:.2f}")

        fitted = results["fit_obj"].fittedvalues() if hasattr(results["fit_obj"], 'fittedvalues') and callable(results["fit_obj"].fittedvalues) else results["fit_obj"].fittedvalues
        actual = results["train_series"]
        rmse = np.sqrt(np.mean((fitted - actual)**2))
        
        # Robust MAPE Fix to prevent inf%
        mask = actual != 0
        if np.any(mask):
            mape_val = np.mean(np.abs((actual[mask] - fitted[mask]) / actual[mask])) * 100
            mape_str = f"{mape_val:.2f}%"
        else:
            mape_str = "N/A"
        
        c2.markdown("#### Performance")
        c2.metric("RMSE", f"{rmse:.4f}")
        c2.metric("MAPE", mape_str)

        c3.markdown("#### Quality")
        if llf: c3.metric("Log-Likelihood", f"{llf:.2f}")
        lb_p = acorr_ljungbox(results["resid"], lags=[min(10, len(results['resid'])//2)], return_df=True)['lb_pvalue'].values[0]
        c3.metric("Ljung-Box p-val", f"{lb_p:.3f}")

    with tab4:
        st.dataframe(results["fc_df"].style.format("{:.2f}"), use_container_width=True)
        buffer = BytesIO()
        with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
            results["fc_df"].to_excel(writer, sheet_name='Forecast')
        st.download_button(label="📥 Download Excel Report", data=buffer.getvalue(), file_name=f"{ticker}_forecast.xlsx")

    with tab5:
        st.header("📖 ARIMA Learning Center")
        st.markdown("""
        ### 🏔️ The Box-Jenkins Methodology
        The **Box-Jenkins** methodology is the mathematical cornerstone of this dashboard. It follows a 3-stage iterative cycle:
        
        1. **Identification**: Checking for **Stationarity**. We use differencing ($d$) and transformations (like Log Returns) to ensure the data has a stable mean and variance.
        2. **Estimation**: Fitting the model using $p$ (AutoRegressive) and $q$ (Moving Average) parameters to capture market patterns.
        3. **Diagnostic Checking**: Evaluating the **Residuals**. If residuals are random "White Noise," the model is capture-ready.
        
        ### 🎯 Key Parameters
        - **p (AR - Autoregressive):** Past price influence. It captures how today's price is dependent on previous days.
        - **d (I - Integrated):** The number of differencing steps required to remove trends and achieve stationarity.
        - **q (MA - Moving Average):** Past error influence. It models the impact of sudden market shocks.
        
        ### 📊 Performance Indicators
        - **AIC/BIC**: These criteria reward accuracy but penalize over-complexity. **Lower is better.**
        - **MAPE (Mean Absolute Percentage Error)**: Represents the average error as a percentage of actual values. A score below 5% is considered excellent.
        """)

st.markdown("---")
st.markdown(f"<p style='text-align: center; color: gray;'>{BRAND_NAME} | Built for Educational Excellence</p>", unsafe_allow_html=True)
