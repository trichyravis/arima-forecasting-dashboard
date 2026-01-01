
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
# CONFIGURATION & BRANDING
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
# CSS STYLING (Fixed for Sidebar Visibility & Selection Clarity)
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
    /* Force Sidebar labels, headers, and radio options to White */
    [data-testid="stSidebar"] h3, 
    [data-testid="stSidebar"] label, 
    [data-testid="stSidebar"] .stMarkdown p, 
    [data-testid="stSidebar"] div[role="radiogroup"] p,
    [data-testid="stSidebar"] div[data-testid="stWidgetLabel"] p {{ 
        color: white !important; font-weight: 600 !important;
    }}
    /* Keep selectbox/input text Dark Blue for readability on white background */
    div[data-baseweb="select"] > div, input {{ color: {DARK_BLUE} !important; }}
    /* Slider value labels */
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
    
    # RESTORED: Manual Parameter Selection Sliders
    p, d, q = 1, 1, 1
    if model_mode == "Manual ARIMA":
        col1, col2, col3 = st.columns(3)
        p = col1.slider("p (AR)", 0, 5, 1)
        d = col2.slider("d (I)", 0, 2, 1)
        q = col3.slider("q (MA)", 0, 5, 1)
    
    st.markdown("### 🔮 Forecast Settings")
    forecast_horizon = st.slider("Forecast Horizon (Periods)", 1, 60, 10)
    conf_level = st.selectbox("Confidence Level", ["80%", "90%", "95%", "99%"], index=2)
    refresh_button = st.button("🔄 FETCH DATA & RUN MODEL")

# RESTORED: Selection parameters summary on the main dashboard
st.markdown("### 📊 Current Parameters")
m1, m2, m3, m4 = st.columns(4)
m1.metric("Security", ticker)
m2.metric("Data History", f"{lookback}y")
m3.metric("Model Mode", model_mode)
m4.metric("Confidence", conf_level)

# ═══════════════════════════════════════════════════════════════════════════════
# MODELING LOGIC
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
                # FIX: Ensure forecast numerical values are properly flattened for display
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
        st.subheader("🔍 Residual Diagnostics")
        fig_diag, axes = plt.subplots(2, 2, figsize=(12, 8))
        axes[0, 0].plot(results["resid"]); axes[0, 0].set_title("Standardized Residuals")
        if HAS_SEABORN: sns.histplot(results["resid"], kde=True, ax=axes[0, 1])
        else: axes[0, 1].hist(results["resid"], bins=20)
        axes[0, 1].set_title("Residual Distribution")
        plot_acf(results["resid"], ax=axes[1, 0], lags=20); axes[1, 0].set_title("Residual ACF")
        stats.probplot(results["resid"], dist="norm", plot=axes[1, 1]); axes[1, 1].set_title("Normal Q-Q Plot")
        plt.tight_layout()
        st.pyplot(fig_diag)
        
        lb_test = acorr_ljungbox(results["resid"], lags=[10], return_df=True)
        p_val = lb_test['lb_pvalue'].values[0]
        if p_val > 0.05: st.success(f"✅ Ljung-Box Test (p={p_val:.3f}): Residuals are White Noise.")
        else: st.warning(f"⚠️ Ljung-Box Test (p={p_val:.3f}): Residuals have structure.")

    with tab3:
        st.metric("Optimal Model Order", str(results["order"]))
        st.metric("AIC Score", f"{results['aic']:.2f}")
        
    with tab4:
        st.subheader("Forecasted Results Table")
        st.dataframe(results["fc_df"].style.format("{:.2f}"), use_container_width=True)
        buffer = BytesIO()
        with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
            results["fc_df"].to_excel(writer, sheet_name='Forecast')
        st.download_button(label="📥 Download Excel Report", data=buffer.getvalue(), file_name=f"{ticker}_forecast.xlsx")

# Educational Hub content 
with tab5:
    st.header("📖 ARIMA Learning Center")
    st.write("This dashboard utilizes the **Box-Jenkins Methodology**, which is a systematic process of identifying, fitting, and checking time series models.")
    
    
    st.subheader("What do the parameters mean?")
    st.markdown("""
    * **p (AutoRegressive):** Uses the relationship between an observation and a number of lagged observations. 
    * **d (Integrated):** Uses differencing of raw observations to make the time series stationary. 
    * **q (Moving Average):** Uses the dependency between an observation and a residual error from a moving average model applied to lagged observations. 
    """)
    st.info("💡 **AIC (Akaike Information Criterion):** Measures model quality by rewarding accuracy and penalizing over-complexity. Lower AIC values indicate a better-fitting model.")

st.markdown("---")
st.markdown(f"<p style='text-align: center; color: gray;'>{BRAND_NAME} | Built for Educational Excellence</p>", unsafe_allow_html=True)
