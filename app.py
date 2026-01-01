
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
    "INFY.NS": "Infosys",
    "ICICIBANK.NS": "ICICI Bank"
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
    [data-testid="stSidebar"] {{ 
        background: linear-gradient(135deg, {DARK_BLUE} 0%, {LIGHT_BLUE} 100%) !important; 
    }}
    [data-testid="stSidebar"] h3, [data-testid="stSidebar"] label, [data-testid="stSidebar"] .stMarkdown p, [data-testid="stSidebar"] div[role="radiogroup"] {{ 
        color: white !important; font-weight: 600 !important;
    }}
    [data-testid="stSidebar"] div[data-testid="stWidgetLabel"] p, [data-testid="stSidebar"] div[data-testid="stMarkdownContainer"] p {{
        color: white !important;
    }}
    [data-testid="stSidebar"] .st-ae div {{ color: white !important; }}
    div[data-baseweb="select"] > div, input {{ color: {DARK_BLUE} !important; }}
    [data-testid="stSidebar"] .st-at {{ color: white !important; }}
    .stButton>button {{
        background-color: {GOLD_COLOR} !important;
        color: {DARK_BLUE} !important;
        font-weight: bold !important;
        border-radius: 10px !important;
        width: 100%;
        margin-top: 1rem;
    }}
    </style>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# HERO SECTION
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown(f"""
    <div class="hero-title">
        <h1>ARIMA FORECASTING DASHBOARD</h1>
        <p>Real-Time Box-Jenkins Time Series Forecasting for Indian Equities</p>
        <p>Prof. V. Ravichandran | 28+ Years Corporate Finance & Banking Experience</p>
    </div>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════════════════════════
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
        p = col1.slider("p (AR)", 0, 5, 1)
        d = col2.slider("d (I)", 0, 2, 1)
        q = col3.slider("q (MA)", 0, 5, 1)
    
    st.markdown("### 🔮 Forecast Settings")
    forecast_horizon = st.slider("Forecast Horizon (Periods)", 1, 60, 10)
    refresh_button = st.button("🔄 FETCH DATA & RUN MODEL")

# ═══════════════════════════════════════════════════════════════════════════════
# MAIN LOGIC
# ═══════════════════════════════════════════════════════════════════════════════
tab1, tab2, tab3, tab4, tab5 = st.tabs(["📈 Forecast", "🔍 Diagnostics", "📊 Metrics", "📋 Export", "📚 Educational Hub"])
results = None

if refresh_button:
    with st.spinner("Executing Box-Jenkins Methodology..."):
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
                    model_fit = pm.auto_arima(train_series, seasonal=False, stepwise=True)
                    fc_vals = model_fit.predict(n_periods=forecast_horizon)
                    order, aic, resid = model_fit.order, model_fit.aic(), model_fit.resid()
                else:
                    model_manual = ARIMA(train_series, order=(p, d, q)).fit()
                    fc_vals = model_manual.forecast(steps=forecast_horizon)
                    order, aic, resid = (p, d, q), model_manual.aic, model_manual.resid

                last_p = raw_prices.iloc[-1]
                if transformation == "Log Returns": forecast_prices = last_p * np.exp(np.cumsum(fc_vals))
                elif transformation == "Log Prices": forecast_prices = np.exp(fc_vals)
                elif transformation == "Percentage Returns": forecast_prices = last_p * (1 + np.cumsum(fc_vals))
                else: forecast_prices = fc_vals
                
                f_dates = pd.date_range(raw_prices.index[-1], periods=forecast_horizon + 1, freq=res_map[freq])[1:]
                fc_df = pd.DataFrame({"Forecasted Price": np.array(forecast_prices).flatten()}, index=f_dates)
                results = {"raw": raw_prices, "fc_df": fc_df, "order": order, "aic": aic, "resid": resid}
            except Exception as e:
                st.error(f"Computation Error: {e}")

# ═══════════════════════════════════════════════════════════════════════════════
# TABS CONTENT
# ═══════════════════════════════════════════════════════════════════════════════
if results:
    with tab1:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=results["raw"].index, y=results["raw"], name="Historical Price", line=dict(color=DARK_BLUE)))
        fig.add_trace(go.Scatter(x=results["fc_df"].index, y=results["fc_df"]["Forecasted Price"], name="ARIMA Forecast", line=dict(color='orange', width=3)))
        st.plotly_chart(fig, use_container_width=True)
    with tab2:
        fig_diag, axes = plt.subplots(2, 2, figsize=(12, 8))
        axes[0, 0].plot(results["resid"]); axes[0, 0].set_title("Standardized Residuals")
        if HAS_SEABORN: sns.histplot(results["resid"], kde=True, ax=axes[0, 1])
        else: axes[0, 1].hist(results["resid"], bins=20)
        axes[0, 1].set_title("Residual Distribution")
        plot_acf(results["resid"], ax=axes[1, 0], lags=20); axes[1, 0].set_title("Residual ACF")
        stats.probplot(results["resid"], dist="norm", plot=axes[1, 1]); axes[1, 1].set_title("Normal Q-Q Plot")
        plt.tight_layout()
        st.pyplot(fig_diag)
    with tab3:
        st.metric("Optimal Model Order", str(results["order"]))
        st.metric("AIC Score", f"{results['aic']:.2f}")
    with tab4:
        st.dataframe(results["fc_df"].style.format("{:.2f}"), use_container_width=True)
        buffer = BytesIO()
        with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
            results["fc_df"].to_excel(writer, sheet_name='Forecast')
        st.download_button(label="📥 Download Excel Report", data=buffer.getvalue(), file_name=f"{ticker}_forecast.xlsx")

# 📚 EDUCATIONAL HUB CONTENT
with tab5:
    st.header("📖 ARIMA Learning Center")
    st.write("Welcome to the educational guide. This dashboard is built on the **Box-Jenkins Methodology**, the gold standard for statistical time series forecasting.")
    
    

    st.subheader("1. What is ARIMA?")
    st.write("""
    **ARIMA** stands for **A**uto**R**egressive **I**ntegrated **M**oving **A**verage. It predicts future values based on past values (Autoregression) and past errors (Moving Average).
    
    * **p (AutoRegressive - AR):** The number of lag observations included in the model. It looks at how much yesterday's price affects today's.
    * **d (Integrated - I):** The number of times the raw observations are differenced to make the data stationary (stable mean/variance).
    * **q (Moving Average - MA):** The size of the moving average window applied to forecast errors.
    """)

    st.subheader("2. Key Terminology")
    col1, col2 = st.columns(2)
    with col1:
        st.info("**Stationarity:** A property where the mean and variance do not change over time. Stock prices are rarely stationary, which is why we use 'Log Returns' or 'Differencing'.")
        st.info("**AIC (Akaike Information Criterion):** A measure of model quality. **Lower is better.** It rewards accuracy but penalizes over-complexity.")
    with col2:
        st.info("**White Noise:** If residuals are white noise, it means the model has captured all the patterns. Only random, unpredictable error remains.")
        st.info("**Log Returns:** Calculating the percentage change in log space. This stabilizes the variance of financial data, making the ARIMA model more reliable.")

    st.subheader("3. Understanding Diagnostics")
    st.write("""
    * **ACF Plot:** Used to identify the 'q' parameter. If bars stay within the blue shaded area, the residuals are random.
    * **Q-Q Plot:** If the points lie on the red diagonal line, the errors are normally distributed—a sign of a healthy model.
    * **Standardized Residuals:** Should look like random 'noise' around zero with no visible trends or patterns.
    """)
    
    st.success("💡 **Pro Tip:** Start with 'Auto ARIMA'. It automatically tests hundreds of (p,d,q) combinations to find the one with the lowest AIC score!")

st.markdown("---")
st.markdown(f"<p style='text-align: center; color: gray;'>{BRAND_NAME} | Built for Educational Excellence</p>", unsafe_allow_html=True)
