
import streamlit as st
import pandas as pd  # Fixed: changed from 'import pd'
import numpy as np
from datetime import datetime, timedelta
import yfinance as yf
import warnings
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import matplotlib.pyplot as plt
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

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE CONFIG & CSS FIXES
# ═══════════════════════════════════════════════════════════════════════════════
st.set_page_config(page_title="ARIMA Dashboard - The Mountain Path", page_icon="🏔️", layout="wide")

st.markdown(f"""
    <style>
    /* Hero Header */
    .hero-title {{ 
        background: linear-gradient(135deg, {DARK_BLUE} 0%, {LIGHT_BLUE} 100%); 
        padding: 2rem; border-radius: 20px; margin-bottom: 2rem; 
        box-shadow: 0 12px 30px rgba(0, 51, 102, 0.4); border: 4px solid {DARK_BLUE}; 
        color: white; text-align: right;
    }}
    
    /* Sidebar Styling */
    [data-testid="stSidebar"] {{ 
        background: linear-gradient(135deg, {DARK_BLUE} 0%, {LIGHT_BLUE} 100%) !important; 
    }}
    
    /* UNIVERSAL WHITE TEXT: Headers, Labels, and Radio Button Options */
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
# HELPER FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════
def fetch_stock_data(ticker, years):
    end_date = datetime.today()
    start_date = end_date - timedelta(days=years * 365)
    try:
        data = yf.download(ticker, start=start_date, end=end_date)
        if data.empty: return None
        # Handle yfinance MultiIndex or single index
        if isinstance(data.columns, pd.MultiIndex):
            return data['Close'][ticker].dropna()
        return data['Close'].dropna()
    except:
        return None

# ═══════════════════════════════════════════════════════════════════════════════
# UI LAYOUT
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
    
    st.markdown("### 🔮 Forecast Settings")
    forecast_horizon = st.slider("Forecast Horizon (Periods)", 1, 60, 10)
    refresh_button = st.button("🔄 FETCH DATA & RUN MODEL")

# ═══════════════════════════════════════════════════════════════════════════════
# MAIN LOGIC & OUTPUT
# ═══════════════════════════════════════════════════════════════════════════════
tab1, tab2, tab3 = st.tabs(["📈 Forecast Chart", "📊 Model Metrics", "📋 Export Data"])
results = None

if refresh_button:
    with st.spinner("Executing Box-Jenkins Methodology..."):
        raw_prices = fetch_stock_data(ticker, lookback)
        
        if raw_prices is not None:
            # Resampling based on user selection
            if freq == "Weekly": raw_prices = raw_prices.resample('W').last()
            elif freq == "Monthly": raw_prices = raw_prices.resample('M').last()
            
            # Apply Transformation
            if transformation == "Log Returns":
                train_series = np.log(raw_prices).diff().dropna()
            elif transformation == "Log Prices":
                train_series = np.log(raw_prices)
            elif transformation == "Percentage Returns":
                train_series = raw_prices.pct_change().dropna()
            else:
                train_series = raw_prices
            
            try:
                if model_mode == "Auto ARIMA":
                    model = pm.auto_arima(train_series, seasonal=False, stepwise=True)
                    fc = model.predict(n_periods=forecast_horizon)
                    order, aic = model.order, model.aic()
                else:
                    # Default manual order for demonstration; can be expanded for full (p,d,q) sliders
                    fit = ARIMA(train_series, order=(1,1,1)).fit()
                    fc = fit.forecast(steps=forecast_horizon)
                    order, aic = (1,1,1), fit.aic

                # Inversion logic to return to Price Level
                if transformation == "Log Returns":
                    last_price = raw_prices.iloc[-1]
                    inv_fc = last_price * np.exp(np.cumsum(fc))
                elif transformation == "Log Prices":
                    inv_fc = np.exp(fc)
                elif transformation == "Percentage Returns":
                    last_price = raw_prices.iloc[-1]
                    inv_fc = last_price * (1 + np.cumsum(fc))
                else:
                    inv_fc = fc
                
                # Create date range for forecast
                freq_map = {"Daily": "B", "Weekly": "W", "Monthly": "M"}
                f_dates = pd.date_range(raw_prices.index[-1], periods=forecast_horizon + 1, freq=freq_map[freq])[1:]
                fc_df = pd.DataFrame({"Forecasted Price": inv_fc}, index=f_dates)
                
                results = {"raw": raw_prices, "fc_df": fc_df, "order": order, "aic": aic}
            except Exception as e:
                st.error(f"Computation Error: {e}")

if results:
    with tab1:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=results["raw"].index, y=results["raw"], name="Historical"))
        fig.add_trace(go.Scatter(x=results["fc_df"].index, y=results["fc_df"]["Forecasted Price"], name="Forecast", line=dict(color='orange', width=3)))
        fig.update_layout(title=f"ARIMA{results['order']} Model for {ticker}", template="plotly_white")
        st.plotly_chart(fig, use_container_width=True)
        
    with tab2:
        st.metric("Optimal Model Order", str(results["order"]))
        st.metric("AIC Score", f"{results['aic']:.2f}")
        
    with tab3:
        st.subheader("Forecasted Results Table")
        st.dataframe(results["fc_df"], use_container_width=True)
        
        # Excel Download Logic
        buffer = BytesIO()
        with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
            results["fc_df"].to_excel(writer, sheet_name='Forecast')
        
        st.download_button(
            label="📥 Download Excel Report",
            data=buffer.getvalue(),
            file_name=f"{ticker}_forecast.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
else:
    st.info("Click 'FETCH DATA & RUN MODEL' in the sidebar to start the analysis.")

st.markdown("---")
st.markdown("### Box-Jenkins ARIMA Methodology")
