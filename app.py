
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
# CSS STYLING
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown(f"""
    <style>
    .hero-title {{ background: linear-gradient(135deg, {DARK_BLUE} 0%, {LIGHT_BLUE} 100%); padding: 2rem; border-radius: 20px; margin-bottom: 2rem; box-shadow: 0 12px 30px rgba(0, 51, 102, 0.4); border: 4px solid {DARK_BLUE}; color: white; text-align: center; }}
    [data-testid="stSidebar"] {{ background: linear-gradient(135deg, {DARK_BLUE} 0%, {LIGHT_BLUE} 100%) !important; }}
    [data-testid="stSidebar"] h3, [data-testid="stSidebar"] label, [data-testid="stSidebar"] p, [data-testid="stSidebar"] div[role="radiogroup"] p, [data-testid="stSidebar"] div[data-testid="stWidgetLabel"] p {{ color: white !important; font-weight: 600 !important; }}
    [data-testid="stSidebar"] .st-ae div {{ color: white !important; }}
    div[data-baseweb="select"] > div, input {{ color: {DARK_BLUE} !important; }}
    [data-testid="stSidebar"] .st-at {{ color: white !important; }}
    .stButton>button {{ background-color: {GOLD_COLOR} !important; color: {DARK_BLUE} !important; font-weight: bold !important; border-radius: 10px !important; width: 100%; }}
    </style>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# HERO SECTION & SIDEBAR
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown(f"<div class='hero-title'><h1>ARIMA FORECASTING DASHBOARD</h1><p>Real-Time Box-Jenkins Time Series Forecasting for Indian Equities</p><p>Prof. V. Ravichandran | 28+ Years Finance Experience</p></div>", unsafe_allow_html=True)

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
        c1, c2, c3 = st.columns(3)
        p, d, q = c1.slider("p", 0, 5, 1), c2.slider("d", 0, 2, 1), c3.slider("q", 0, 5, 1)
    st.markdown("### 🔮 Forecast Settings")
    forecast_horizon = st.slider("Forecast Horizon (Periods)", 1, 60, 10)
    refresh_button = st.button("🔄 FETCH DATA & RUN MODEL")
    st.markdown("---")
    st.markdown(f"<a href='https://www.linkedin.com/in/trichyravis' target='_blank' style='display: block; padding: 0.5rem; background: #0077b5; color: white; text-align: center; text-decoration: none; border-radius: 5px; font-weight: bold;'>🔗 LinkedIn Profile</a>", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# CURRENT PARAMETERS SUMMARY
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown("### 📊 Current Selection Parameters")
pm1, pm2, pm3, pm4 = st.columns(4)
pm1.metric("Ticker", ticker)
pm2.metric("History", f"{lookback}y")
pm3.metric("Mode", model_mode)
pm4.metric("Horizon", f"{forecast_horizon}")

# ═══════════════════════════════════════════════════════════════════════════════
# PROCESSING LOGIC
# ═══════════════════════════════════════════════════════════════════════════════
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["📈 Forecast", "🧪 Backtesting", "🔍 Diagnostics", "📊 Metrics", "📋 Export", "📚 Educational Hub"])
results = None

if refresh_button:
    with st.spinner("Processing Box-Jenkins Analysis..."):
        data = yf.download(ticker, start=datetime.now()-timedelta(days=lookback*365))
        if not data.empty:
            raw_prices = data['Close'][ticker] if isinstance(data.columns, pd.MultiIndex) else data['Close']
            raw_prices = raw_prices.dropna()
            res_map = {"Daily": "B", "Weekly": "W", "Monthly": "M"}
            raw_prices = raw_prices.resample(res_map[freq]).last().ffill()
            
            # Backtesting Data Split
            bt_len = min(20, len(raw_prices)//5)
            bt_train_raw = raw_prices[:-bt_len]
            bt_actual_raw = raw_prices[-bt_len:]
            
            def get_series(series, t):
                if t == "Log Returns": return np.log(series).diff().dropna()
                if t == "Log Prices": return np.log(series)
                if t == "Percentage Returns": return series.pct_change().dropna()
                return series

            train_series = get_series(raw_prices, transformation)
            bt_train_series = get_series(bt_train_raw, transformation)

            try:
                # Main Model Fit
                if model_mode == "Auto ARIMA":
                    model = pm.auto_arima(train_series, seasonal=False)
                    fc = model.predict(n_periods=forecast_horizon)
                    order, aic, resid, fit_obj = model.order, model.aic(), model.resid(), model
                    
                    bt_model = pm.auto_arima(bt_train_series, seasonal=False)
                    bt_fc = bt_model.predict(n_periods=bt_len)
                else:
                    fit = ARIMA(train_series, order=(p, d, q)).fit()
                    fc = fit.forecast(steps=forecast_horizon)
                    order, aic, resid, fit_obj = (p, d, q), fit.aic, fit.resid, fit
                    
                    bt_fit = ARIMA(bt_train_series, order=(p, d, q)).fit()
                    bt_fc = bt_fit.forecast(steps=bt_len)

                # Inversion logic
                def invert_fc(fc_vals, last_p, t):
                    if t == "Log Returns": return last_p * np.exp(np.cumsum(fc_vals))
                    if t == "Log Prices": return np.exp(fc_vals)
                    if t == "Percentage Returns": return last_p * (1 + np.cumsum(fc_vals))
                    return fc_vals

                inv_fc = invert_fc(fc, raw_prices.iloc[-1], transformation)
                inv_bt = invert_fc(bt_fc, bt_train_raw.iloc[-1], transformation)
                
                f_dates = pd.date_range(raw_prices.index[-1], periods=forecast_horizon + 1, freq=res_map[freq])[1:]
                fc_df = pd.DataFrame({"Forecasted Price": np.array(inv_fc).flatten()}, index=f_dates)
                
                # Backtest Comparison DF
                bt_comp = pd.DataFrame({
                    "Actual Price": bt_actual_raw.values,
                    "Predicted Price": np.array(inv_bt).flatten()
                }, index=bt_actual_raw.index)
                bt_comp["Variance (%)"] = ((bt_comp["Predicted Price"] - bt_comp["Actual Price"]) / bt_comp["Actual Price"]) * 100
                
                results = {"raw": raw_prices, "fc_df": fc_df, "bt_comp": bt_comp, "order": order, "aic": aic, "resid": resid, "fit_obj": fit_obj, "train_series": train_series}
            except Exception as e: st.error(f"Error: {e}")

if results:
    with tab1:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=results["raw"].index, y=results["raw"], name="Historical", line=dict(color=DARK_BLUE)))
        fig.add_trace(go.Scatter(x=results["fc_df"].index, y=results["fc_df"]["Forecasted Price"], name="ARIMA Forecast", line=dict(color='orange', width=3)))
        st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.subheader("🧪 Backtesting Results: Forecast vs. Actual")
        fig_bt = go.Figure()
        fig_bt.add_trace(go.Scatter(x=results["raw"].index, y=results["raw"], name="Actual", line=dict(color=DARK_BLUE)))
        fig_bt.add_trace(go.Scatter(x=results["bt_comp"].index, y=results["bt_comp"]["Predicted Price"], name="Model Prediction", line=dict(color='red', dash='dash')))
        st.plotly_chart(fig_bt, use_container_width=True)
        st.write("**Detailed Variance Analysis**")
        st.dataframe(results["bt_comp"].style.format("{:.2f}"))

    with tab3:
        fig_diag, axes = plt.subplots(2, 2, figsize=(12, 8))
        axes[0, 0].plot(results["resid"]); axes[0, 0].set_title("Standardized Residuals")
        if HAS_SEABORN: sns.histplot(results["resid"], kde=True, ax=axes[0, 1])
        plot_acf(results["resid"], ax=axes[1, 0], lags=min(20, len(results["resid"])//2)); axes[1, 0].set_title("Residual ACF")
        stats.probplot(results["resid"], dist="norm", plot=axes[1, 1]); axes[1, 1].set_title("Normal Q-Q Plot")
        plt.tight_layout(); st.pyplot(fig_diag)

    with tab4:
        st.subheader("📊 Model Performance Metrics")
        c1, c2, c3 = st.columns(3)
        c1.metric("Order", str(results["order"]))
        c1.metric("AIC", f"{results['aic']:.2f}")
        
        fitted = results["fit_obj"].fittedvalues() if hasattr(results["fit_obj"], 'fittedvalues') and callable(results["fit_obj"].fittedvalues) else results["fit_obj"].fittedvalues
        rmse = np.sqrt(np.mean((fitted - results["train_series"])**2))
        c2.metric("RMSE", f"{rmse:.4f}")
        
        lb_p = acorr_ljungbox(results["resid"], lags=[10], return_df=True)['lb_pvalue'].values[0]
        c3.metric("Ljung-Box p-val", f"{lb_p:.3f}")

    with tab5:
        st.dataframe(results["fc_df"].style.format("{:.2f}"), use_container_width=True)
        buffer = BytesIO()
        with pd.ExcelWriter(buffer, engine='openpyxl') as writer: results["fc_df"].to_excel(writer, sheet_name='Forecast')
        st.download_button("📥 Download Excel Report", data=buffer.getvalue(), file_name=f"{ticker}_forecast.xlsx")

with tab6:
    st.header("📖 ARIMA Learning Center")
    
    st.markdown("""
    ### Stages of Box-Jenkins Methodology
    1. **Identification**: Transform data to be stationary ($d$).
    2. **Estimation**: Find best $p$ and $q$ parameters.
    3. **Diagnostics**: Ensure residuals are White Noise (ACF & Q-Q Plots).
    """)
