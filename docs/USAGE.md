# Usage Guide - ARIMA Forecasting Dashboard

## Getting Started

### 1. Start the Dashboard

```bash
streamlit run app.py
```

Navigate to `http://localhost:8501`

### 2. Select Data Source

**Options:**
- **Yahoo Finance:** Real-time stock data (NIFTY, SENSEX, etc.)
- **Upload CSV:** Your own historical data
- **Upload Excel:** Excel file with prices

### 3. Configure ARIMA Model

**Parameters:**
- **p:** AR order (auto-selected 0-5)
- **d:** Differencing order (auto-selected 0-2)
- **q:** MA order (auto-selected 0-5)

Click **"Auto Select Parameters"** to let the algorithm choose.

### 4. Generate Forecast

1. Select number of periods to forecast (1-60 days)
2. Choose confidence level (90%, 95%, 99%)
3. Click **"Generate Forecast"**

### 5. Analyze Results

**Tabs:**
- **Forecast:** View forecast with confidence intervals
- **Model Metrics:** AIC, BIC, RMSE, MAE values
- **Diagnostics:** ACF/PACF, residual analysis
- **Comparison:** Compare multiple model parameters

## Features

### Data Quality Analysis

Automatically checks:
- ✅ Missing values
- ✅ Outliers
- ✅ Data length
- ✅ Date consistency

### Stationarity Testing

Tests include:
- **ADF Test:** Augmented Dickey-Fuller
- **KPSS Test:** Kwiatkowski-Phillips-Schmidt-Shin
- **Recommendation:** Suggests differencing order

### Model Validation

Diagnostic plots:
- **Residuals:** Check randomness
- **ACF/PACF:** Identify patterns
- **Q-Q Plot:** Check normality
- **Histogram:** Distribution analysis

### Forecast Interpretation

**What the forecast shows:**
- **Blue line:** Actual historical data
- **Red line:** Forecast values
- **Gray band:** 95% confidence interval

## Step-by-Step Example

### Example: Forecasting NIFTY 50

```
Step 1: Select "Yahoo Finance" as source
Step 2: Enter ticker "^NSEI" (NIFTY 50)
Step 3: Click "Load Data"
Step 4: Review data quality (should show ~250 observations)
Step 5: Click "Auto Select Parameters"
Step 6: Set forecast period to 30 days
Step 7: Click "Generate Forecast"
Step 8: Review tabs for results
```

## Understanding Results

### AIC (Akaike Information Criterion)
- Lower is better
- Balances fit vs. complexity
- Typical range: 1000-3000

### BIC (Bayesian Information Criterion)
- Similar to AIC, slightly stricter
- Better for model selection
- Penalizes complexity more

### RMSE (Root Mean Square Error)
- Average prediction error in same units as data
- Lower is better
- Typical range: 0.5-2.0% of mean price

### MAE (Mean Absolute Error)
- Average absolute error
- More robust to outliers
- Same units as RMSE

## Tips & Tricks

### 1. Data Quality
- Use at least 100 observations
- Check for gaps or missing values
- Remove outliers if suspicious

### 2. Parameter Selection
- Let auto-select choose best parameters
- Check different (p, d, q) combinations
- Compare AIC/BIC values

### 3. Forecast Horizon
- Short-term (1-10 days): More accurate
- Medium-term (10-30 days): Moderate accuracy
- Long-term (30+ days): Less reliable, wider CI

### 4. Interpretation
- Wider confidence intervals = more uncertainty
- Check residual plots for autocorrelation
- Validate on holdout test data

## Troubleshooting

### Issue: "Insufficient data"
**Solution:** Use at least 50 observations

### Issue: "Series is not stationary"
**Solution:** Increase differencing order (d parameter)

### Issue: "Forecast seems unrealistic"
**Solution:** Check residual plots, try different parameters

### Issue: "Slow performance"
**Solution:** Clear cache, reduce data period, close other apps

## Exporting Results

**Download Options:**
- Forecast as CSV
- Model summary as PDF
- Diagnostic plots as PNG/SVG

## Advanced Usage

### Custom Data Upload

Format your CSV:
```
Date,Close
2023-01-01,100.5
2023-01-02,101.2
2023-01-03,100.8
```

Required columns:
- **Date:** YYYY-MM-DD format
- **Close:** Numeric price values

### Seasonal ARIMA (SARIMA)

For seasonal data (e.g., quarterly patterns):
- Set seasonal period (4 for quarterly, 12 for monthly)
- Add seasonal parameters (P, D, Q)
- Auto-select will handle automatically

### Model Comparison

Compare multiple ARIMA models:
1. Fit Model A (p=1, d=1, q=1)
2. Note AIC score
3. Fit Model B (p=2, d=1, q=1)
4. Compare results
5. Choose lower AIC

## Next Steps

- Read [API_REFERENCE.md](API_REFERENCE.md) for technical details
- Check [educational/FORECAST_INTERPRETATION.pdf](educational/FORECAST_INTERPRETATION.pdf) for deeper understanding
- Review [TROUBLESHOOTING.md](TROUBLESHOOTING.md) for common issues

---

**Happy Forecasting!** 📊
