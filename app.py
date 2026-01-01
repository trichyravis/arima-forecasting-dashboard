
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

# Full Nifty 50 List
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
# CSS STYLING
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown(f"""
    <style>
    .hero-title {{ 
        background: linear-gradient(135deg, {DARK_BLUE} 0%, {LIGHT_BLUE} 100%); 
        padding: 2rem; border-radius: 20px; margin-bottom: 2rem; 
        box-shadow: 0 12px 30px rgba(0, 51, 102, 0.4); border: 4px solid {DARK_BLUE}; 
        color: white; text-align: center;
    }}
    [data-testid="stSidebar"] {{ background: linear-gradient(135deg, {DARK_BLUE} 0%, {LIGHT_BLUE} 100%) !important; }}
    [data-testid="stSidebar"] h3, [data-testid="stSidebar"] label, [data-testid="stSidebar"] p, 
    [data-testid="stSidebar"] div[role="radiogroup"] p, [data-testid="stSidebar"] div[data-testid="stWidgetLabel"] p {{ 
        color: white !important; font-weight: 600 !important;
    }}
    [data-testid="stSidebar"] .st-ae div {{ color: white !important; }}
    div[data-baseweb="select"] > div, input {{ color: {DARK_BLUE} !important; }}
    [data-testid="stSidebar"] .st-at {{ color: white !important; }}
    .stButton>button {{ background-color: {GOLD_COLOR} !important; color: {DARK_BLUE} !important; font-weight: bold !important; width: 100%; }}
    </style>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# UI LAYOUT
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown(f"<div class='hero-title'><h1>ARIMA FORECASTING DASHBOARD</h1><p>Real-Time Box-Jenkins Time Series Forecasting for Nifty 50</p><p>Prof. V. Ravichandran | 28+ Years Experience</p></div>", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("### 📊 Data Selection")
    ticker = st.selectbox("Select Nifty 50 Stock", options=list(NIFTY_50_STOCKS.keys()), format_func=lambda x: f"{x} - {NIFTY_50_STOCKS[x]}")
    lookback = st.selectbox("Years of Historical Data", [1, 2, 3, 5, 10], index=2)
    freq = st.radio("Data Frequency", ["Daily", "Weekly", "Monthly"])
    
    st.markdown("### ⚙️ Model Configuration")
    transformation = st.radio("Price Transformation", ["Price Level", "Log Prices", "Log Returns", "Percentage Returns"], index=2)
    model_mode = st.radio("Model Selection", ["Manual ARIMA", "Auto ARIMA"], index=1)
    
    p, d, q = 1, 1, 1
    if model_mode == "Manual ARIMA":
        c1, c2, c3 = st.columns(3)
        p, d, q = c1.slider("p", 0, 5, 1), c2.slider("d", 0, 2, 1), c3.slider("q", 0, 5, 1)
    
    st.markdown("### 🔮 Forecast Settings")
    forecast_horizon = st.slider("Forecast Horizon (Periods)", 1, 60, 10)
    refresh_button = st.button("🔄 FETCH DATA & RUN MODEL")

    st.markdown("---")
    st.markdown("### Prof. V. Ravichandran")
    st.markdown(f"<a href='https://www.linkedin.com/in/trichyravis' target='_blank' style='display: block; padding: 0.5rem; background: #0077b5; color: white; text-align: center; text-decoration: none; border-radius: 5px; font-weight: bold;'>🔗 LinkedIn Profile</a>", unsafe_allow_html=True)

# Main Metrics Summary
st.markdown("### 📊 Current Parameters")
m1, m2, m3, m4 = st.columns(4)
m1.metric("Stock Selected", ticker)
m2.metric("History", f"{lookback}y")
m3.metric("Transformation", transformation)
m4.metric("Mode", model_mode)

# ═══════════════════════════════════════════════════════════════════════════════
# PROCESSING
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
                    fc, order, aic, resid = model.predict(n_periods=forecast_horizon), model.order, model.aic(), model.resid()
                else:
                    fit = ARIMA(train_series, order=(p, d, q)).fit()
                    fc, order, aic, resid = fit.forecast(steps=forecast_horizon), (p, d, q), fit.aic, fit.resid

                last_p = raw_prices.iloc[-1]
                if transformation == "Log Returns": inv_fc = last_p * np.exp(np.cumsum(fc))
                elif transformation == "Log Prices": inv_fc = np.exp(fc)
                elif transformation == "Percentage Returns": inv_fc = last_p * (1 + np.cumsum(fc))
                else: inv_fc = fc
                
                f_dates = pd.date_range(raw_prices.index[-1], periods=forecast_horizon + 1, freq=res_map[freq])[1:]
                fc_df = pd.DataFrame({"Forecasted Price": np.array(inv_fc).flatten()}, index=f_dates)
                results = {"raw": raw_prices, "fc_df": fc_df, "order": order, "aic": aic, "resid": resid}
            except Exception as e: st.error(f"Computation Error: {e}")

if results:
    with tab1:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=results["raw"].index, y=results["raw"], name="Historical"))
        fig.add_trace(go.Scatter(x=results["fc_df"].index, y=results["fc_df"]["Forecasted Price"], name="Forecast", line=dict(color='orange', width=3)))
        st.plotly_chart(fig, use_container_width=True)
    with tab2:
        fig_diag, axes = plt.subplots(2, 2, figsize=(12, 8))
        axes[0, 0].plot(results["resid"]); axes[0, 0].set_title("Standardized Residuals")
        if HAS_SEABORN: sns.histplot(results["resid"], kde=True, ax=axes[0, 1])
        plot_acf(results["resid"], ax=axes[1, 0], lags=20); axes[1, 0].set_title("Residual ACF")
        stats.probplot(results["resid"], dist="norm", plot=axes[1, 1]); axes[1, 1].set_title("Normal Q-Q Plot")
        plt.tight_layout(); st.pyplot(fig_diag)
    with tab3:
        st.metric("Optimal ARIMA Order", str(results["order"]))
        st.metric("AIC Score (Lower is Better)", f"{results['aic']:.2f}")
    with tab4:
        st.dataframe(results["fc_df"].style.format("{:.2f}"), use_container_width=True)
        buffer = BytesIO()
        with pd.ExcelWriter(buffer, engine='openpyxl') as writer: results["fc_df"].to_excel(writer, sheet_name='Forecast')
        st.download_button(label="📥 Download Excel Report", data=buffer.getvalue(), file_name=f"{ticker}_forecast.xlsx")

with tab5:
    st.header("📖 ARIMA Learning Center")
    st.markdown("""
    ### 1. The Box-Jenkins Methodology
    The Box-Jenkins approach consists of a three-stage iterative process:
    - **Identification:** Checking for stationarity and determining the initial (p, d, q).
    - **Estimation:** Finding the coefficients that minimize the error.
    - **Diagnostic Checking:** Testing if the residuals are 'White Noise'.
    """)
    
    st.markdown("""
    ### 2. Parameter Definitions
    - **p (AR - Autoregressive):** Looks at the relationship between the current value and its own past values.
    - **d (I - Integrated):** The number of differencing steps required to remove trends and seasonality.
    - **q (MA - Moving Average):** Models the error term as a linear combination of error terms occurring contemporaneously and at various times in the past.
    
    ### 3. Model Fitness (AIC)
    The **Akaike Information Criterion (AIC)** is used to compare models. It estimates the relative amount of information lost by a given model: the less information a model loses, the higher the quality of that model.
    """)
