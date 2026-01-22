
"""
ARIMA Modeling: Complete Implementation Code
Financial Risk Management and Time Series Forecasting
Prof. V. Ravichandran - The Mountain Path: World of Finance

This module contains all practical implementations from the comprehensive ARIMA guide,
ready for deployment in financial risk management systems.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Time series analysis libraries
from statsmodels.tsa.stattools import adfuller, acf, pacf, kpss
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.arima_model import auto_arima
from statsmodels.graphics.gofplots import qqplot
from statsmodels.stats.diagnostic import acorr_ljungbox

# Machine learning and evaluation
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
import scipy.stats as stats

# Set visualization style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 8)
plt.rcParams['font.size'] = 10

# ============================================================================
# SECTION 1: DATA GENERATION AND EXPLORATION
# ============================================================================

def load_financial_data(n_obs=500, stock_type='equity'):
    """
    Load or generate financial time series data
    
    Parameters:
    -----------
    n_obs : int
        Number of observations (default: 500 daily observations)
    stock_type : str
        Type of financial data ('equity', 'fx', 'bond', 'commodity')
    
    Returns:
    --------
    pd.DataFrame
        DataFrame with Price, Returns, Log_Price, Log_Returns columns
    """
    np.random.seed(42)
    dates = pd.date_range(start='2022-01-01', periods=n_obs, freq='D')
    
    if stock_type == 'equity':
        # Equity: positive drift, moderate volatility
        returns = np.random.normal(0.0005, 0.02, n_obs)  # 0.05% mean, 2% vol
        base_price = 100
    elif stock_type == 'fx':
        # FX: near zero drift, lower volatility
        returns = np.random.normal(0.00001, 0.005, n_obs)  # ~0%, 0.5% vol
        base_price = 1.0
    elif stock_type == 'bond':
        # Bond yield: mean-reverting
        returns = np.random.normal(0.00, 0.003, n_obs)
        base_price = 100.0
    else:  # commodity
        # Commodity: higher volatility
        returns = np.random.normal(0.0001, 0.03, n_obs)
        base_price = 50.0
    
    prices = base_price * np.exp(np.cumsum(returns))
    
    data = pd.DataFrame({
        'Date': dates,
        'Price': prices,
        'Returns': returns
    })
    data.set_index('Date', inplace=True)
    
    # Log transformation
    data['Log_Price'] = np.log(data['Price'])
    data['Log_Returns'] = data['Log_Price'].diff()
    
    return data


def explore_data(data, series_name='Series'):
    """
    Comprehensive data exploration with visualizations
    """
    print(f"\n{'='*70}")
    print(f"DATA EXPLORATION: {series_name}")
    print(f"{'='*70}")
    
    # Summary statistics
    print(f"\nSummary Statistics:")
    print(data.describe())
    
    # Visualization
    fig, axes = plt.subplots(3, 1, figsize=(14, 10))
    
    # Price series
    axes[0].plot(data.index, data['Price'], color='navy', linewidth=1.5)
    axes[0].set_title(f'{series_name}: Price Level', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('Price')
    axes[0].grid(True, alpha=0.3)
    
    # Log price
    axes[1].plot(data.index, data['Log_Price'], color='darkgreen', linewidth=1.5)
    axes[1].set_title(f'{series_name}: Log Price', fontsize=12, fontweight='bold')
    axes[1].set_ylabel('Log Price')
    axes[1].grid(True, alpha=0.3)
    
    # Returns
    axes[2].plot(data.index, data['Log_Returns'], color='darkred', linewidth=1)
    axes[2].set_title(f'{series_name}: Log Returns', fontsize=12, fontweight='bold')
    axes[2].set_ylabel('Log Returns')
    axes[2].set_xlabel('Date')
    axes[2].grid(True, alpha=0.3)
    axes[2].axhline(y=0, color='black', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.show()
    
    print(f"\nAutocorrelation of Returns (first 10 lags):")
    acf_values = acf(data['Log_Returns'].dropna(), nlags=10)
    for i, val in enumerate(acf_values[:11]):
        print(f"  Lag {i}: {val:.4f}")


# ============================================================================
# SECTION 2: STATIONARITY TESTING
# ============================================================================

def adf_test(series, name='Series', verbose=True):
    """
    Augmented Dickey-Fuller Test for stationarity
    
    H0: Unit root exists (nonstationary)
    H1: Unit root does not exist (stationary)
    """
    result = adfuller(series.dropna(), autolag='AIC')
    
    if verbose:
        print(f"\nADF Test Results for {name}:")
        print(f"{'─'*60}")
        print(f"  ADF Test Statistic:        {result[0]:>12.6f}")
        print(f"  P-value:                   {result[1]:>12.6f}")
        print(f"  Number of Lags Used:       {result[2]:>12d}")
        print(f"  Number of Observations:    {result[3]:>12d}")
        print(f"\n  Critical Values:")
        for key, value in result[4].items():
            print(f"    {key:3}: {value:>10.3f}")
        
        print(f"\n  Result: ", end='')
        if result[1] <= 0.05:
            print(f"✓ STATIONARY (reject H0, p < 0.05)")
            is_stationary = True
        else:
            print(f"✗ NONSTATIONARY (fail to reject H0, p ≥ 0.05)")
            is_stationary = False
    
    return result, is_stationary


def kpss_test(series, name='Series', verbose=True):
    """
    KPSS Test for stationarity (complementary to ADF)
    
    H0: Series is stationary
    H1: Series has a unit root
    """
    result = kpss(series.dropna(), regression='c', nlags='auto')
    
    if verbose:
        print(f"\nKPSS Test Results for {name}:")
        print(f"{'─'*60}")
        print(f"  KPSS Test Statistic:       {result[0]:>12.6f}")
        print(f"  P-value:                   {result[1]:>12.6f}")
        print(f"  Number of Lags Used:       {result[2]:>12d}")
        
        print(f"\n  Critical Values:")
        for key, value in result[3].items():
            print(f"    {key:3}: {value:>10.3f}")
        
        print(f"\n  Result: ", end='')
        if result[1] <= 0.05:
            print(f"✗ NONSTATIONARY (reject H0, has unit root, p < 0.05)")
            is_stationary = False
        else:
            print(f"✓ STATIONARY (fail to reject H0, p ≥ 0.05)")
            is_stationary = True
    
    return result, is_stationary


def stationarity_analysis(data):
    """
    Comprehensive stationarity analysis
    """
    print("\n" + "="*70)
    print("STATIONARITY ANALYSIS")
    print("="*70)
    
    # Test original log prices
    adf_result, adf_stat = adf_test(data['Log_Price'], 'Log Price')
    kpss_result, kpss_stat = kpss_test(data['Log_Price'], 'Log Price')
    
    # Test log returns
    adf_result, adf_stat = adf_test(data['Log_Returns'].dropna(), 'Log Returns')
    kpss_result, kpss_stat = kpss_test(data['Log_Returns'].dropna(), 'Log Returns')
    
    # Test first difference
    first_diff = data['Log_Price'].diff().dropna()
    adf_result, adf_stat = adf_test(first_diff, 'First Difference of Log Price')
    kpss_result, kpss_stat = kpss_test(first_diff, 'First Difference of Log Price')


# ============================================================================
# SECTION 3: ACF AND PACF ANALYSIS
# ============================================================================

def plot_acf_pacf(series, lags=40, figsize=(15, 6), title=''):
    """
    Plot ACF and PACF for parameter identification
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # ACF
    plot_acf(series.dropna(), lags=lags, ax=axes[0], title=f'ACF {title}')
    axes[0].set_xlabel('Lag')
    axes[0].set_ylabel('ACF')
    axes[0].grid(True, alpha=0.3)
    
    # PACF
    plot_pacf(series.dropna(), lags=lags, ax=axes[1], title=f'PACF {title}', method='ywm')
    axes[1].set_xlabel('Lag')
    axes[1].set_ylabel('PACF')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


def acf_pacf_interpretation(series, lags=20):
    """
    Compute and interpret ACF/PACF patterns
    """
    print("\n" + "="*70)
    print("ACF/PACF INTERPRETATION FOR PARAMETER SELECTION")
    print("="*70)
    
    acf_values = acf(series.dropna(), nlags=lags, fft=False)
    pacf_values = pacf(series.dropna(), nlags=lags, method='ywm')
    
    print(f"\nACF Values (first 15 lags):")
    print(f"{'Lag':<5} {'ACF':<12} {'Significant':<15}")
    print(f"{'-'*32}")
    
    for i in range(min(15, len(acf_values))):
        significant = '✓' if abs(acf_values[i]) > 1.96/np.sqrt(len(series)) else '✗'
        print(f"{i:<5} {acf_values[i]:<12.4f} {significant:<15}")
    
    print(f"\nPACF Values (first 15 lags):")
    print(f"{'Lag':<5} {'PACF':<12} {'Significant':<15}")
    print(f"{'-'*32}")
    
    for i in range(min(15, len(pacf_values))):
        significant = '✓' if abs(pacf_values[i]) > 1.96/np.sqrt(len(series)) else '✗'
        print(f"{i:<5} {pacf_values[i]:<12.4f} {significant:<15}")


# ============================================================================
# SECTION 4: MODEL SELECTION AND FITTING
# ============================================================================

def find_optimal_arima(series, seasonal=False, max_p=5, max_d=2, max_q=5):
    """
    Use Auto ARIMA to identify optimal parameters
    """
    print("\n" + "="*70)
    print("AUTO ARIMA: OPTIMAL PARAMETER SELECTION")
    print("="*70)
    
    if seasonal:
        try:
            model = auto_arima(
                series.dropna(),
                seasonal=True,
                m=4,
                p_range=range(0, max_p),
                d_range=range(0, max_d),
                q_range=range(0, max_q),
                P_range=range(0, 2),
                D_range=range(0, 2),
                Q_range=range(0, 2),
                stepwise=True,
                trace=True,
                error_action='ignore',
                suppress_warnings=True,
                max_order=15
            )
        except:
            print("Seasonal ARIMA failed, falling back to non-seasonal")
            seasonal = False
    
    if not seasonal:
        model = auto_arima(
            series.dropna(),
            p_range=range(0, max_p),
            d_range=range(0, max_d),
            q_range=range(0, max_q),
            stepwise=True,
            trace=True,
            error_action='ignore',
            suppress_warnings=True,
            max_order=15
        )
    
    print(f"\n{'='*70}")
    print(f"Optimal ARIMA Order: {model.order}")
    print(f"AIC: {model.aic:.2f}")
    print(f"BIC: {model.bic:.2f}")
    print(f"{'='*70}")
    
    return model


def fit_arima_model(series, order, name=''):
    """
    Fit ARIMA model and return results
    """
    model = ARIMA(series.dropna(), order=order)
    results = model.fit()
    
    print(f"\nARIMA{order} Model Fit ({name})")
    print(f"{'─'*60}")
    print(f"AIC: {results.aic:>12.2f}")
    print(f"BIC: {results.bic:>12.2f}")
    print(f"HQIC: {results.hqic:>11.2f}")
    
    return results


def compare_models(series, models_to_test, test_size=0.2):
    """
    Fit multiple ARIMA models and compare performance
    """
    n = len(series)
    train_size = int(n * (1 - test_size))
    
    train = series.iloc[:train_size]
    test = series.iloc[train_size:]
    
    print("\n" + "="*80)
    print("MODEL COMPARISON")
    print("="*80)
    print(f"\nTrain size: {train_size}, Test size: {len(test)}")
    print(f"\n{'Order':<15} {'AIC':<12} {'BIC':<12} {'RMSE':<12} {'MAE':<12} {'MAPE':<12}")
    print(f"{'-'*75}")
    
    results_list = []
    
    for order in models_to_test:
        try:
            model = ARIMA(train, order=order)
            results = model.fit()
            
            # Forecast
            forecast = results.get_forecast(steps=len(test)).predicted_mean
            
            # Metrics
            rmse = np.sqrt(mean_squared_error(test, forecast))
            mae = mean_absolute_error(test, forecast)
            mape = mean_absolute_percentage_error(test, forecast)
            
            print(f"ARIMA{order:<5} {results.aic:<12.2f} {results.bic:<12.2f} {rmse:<12.6f} {mae:<12.6f} {mape:<12.4%}")
            
            results_list.append({
                'Order': order,
                'AIC': results.aic,
                'BIC': results.bic,
                'RMSE': rmse,
                'MAE': mae,
                'MAPE': mape,
                'Model': results,
                'Forecast': forecast,
                'Test': test
            })
        except Exception as e:
            print(f"ARIMA{order:<5} Failed to fit: {str(e)[:30]}")
    
    # Identify best model
    best_idx = np.argmin([r['AIC'] for r in results_list])
    best_result = results_list[best_idx]
    
    print(f"\n{'='*80}")
    print(f"BEST MODEL: ARIMA{best_result['Order']} (by AIC)")
    print(f"{'='*80}")
    
    return results_list, best_result


# ============================================================================
# SECTION 5: DIAGNOSTIC ANALYSIS
# ============================================================================

def diagnostic_plots(model_results):
    """
    Generate comprehensive diagnostic plots
    """
    print("\nGenerating diagnostic plots...")
    fig = model_results.plot_diagnostics(figsize=(15, 10))
    plt.tight_layout()
    plt.show()


def residual_analysis(model_results, name=''):
    """
    Comprehensive residual analysis with tests
    """
    print("\n" + "="*70)
    print(f"RESIDUAL ANALYSIS - {name}")
    print("="*70)
    
    residuals = model_results.resid
    
    # Summary statistics
    print(f"\nResidual Statistics:")
    print(f"  Mean:              {residuals.mean():>12.6f}")
    print(f"  Std Dev:           {residuals.std():>12.6f}")
    print(f"  Min:               {residuals.min():>12.6f}")
    print(f"  Max:               {residuals.max():>12.6f}")
    print(f"  Skewness:          {stats.skew(residuals):>12.4f}")
    print(f"  Kurtosis:          {stats.kurtosis(residuals):>12.4f}")
    
    # Ljung-Box test
    lb_test = acorr_ljungbox(residuals, lags=[10], return_df=True)
    
    print(f"\nLjung-Box Test (H0: No Autocorrelation):")
    print(f"  Test Statistic:    {lb_test['lb_stat'].values[0]:>12.4f}")
    print(f"  P-value:           {lb_test['lb_pvalue'].values[0]:>12.6f}")
    
    if lb_test['lb_pvalue'].values[0] > 0.05:
        print(f"  Result: ✓ PASS - No significant autocorrelation (good)")
    else:
        print(f"  Result: ✗ FAIL - Significant autocorrelation detected (model may be misspecified)")
    
    # Normality test
    jb_stat, jb_pvalue = stats.jarque_bera(residuals)
    
    print(f"\nJarque-Bera Test (H0: Normally Distributed):")
    print(f"  Test Statistic:    {jb_stat:>12.4f}")
    print(f"  P-value:           {jb_pvalue:>12.6f}")
    
    if jb_pvalue > 0.05:
        print(f"  Result: ✓ PASS - Residuals approximately normal")
    else:
        print(f"  Result: ⚠ WARNING - Residuals deviate from normality (fat tails common in finance)")
    
    # ADF test on residuals
    adf_result, adf_stat = adf_test(residuals, f'Residuals from {name}', verbose=False)
    
    print(f"\nADF Test on Residuals:")
    print(f"  Test Statistic:    {adf_result[0]:>12.6f}")
    print(f"  P-value:           {adf_result[1]:>12.6f}")
    
    if adf_result[1] < 0.05:
        print(f"  Result: ✓ PASS - Residuals are stationary")
    else:
        print(f"  Result: ✗ FAIL - Residuals are nonstationary (model misspecified)")


# ============================================================================
# SECTION 6: FORECASTING
# ============================================================================

def forecast_arima(model_results, steps=20, alpha=0.05):
    """
    Generate forecasts with confidence intervals
    """
    forecast_obj = model_results.get_forecast(steps=steps)
    forecast_df = forecast_obj.summary_frame(alpha=alpha)
    
    return forecast_df


def plot_forecast(historical_series, forecast_df, last_n=100, title='ARIMA Forecast'):
    """
    Visualization of forecast with confidence intervals
    """
    fig, ax = plt.subplots(figsize=(16, 7))
    
    # Historical data
    ax.plot(historical_series.index[-last_n:], 
            historical_series.iloc[-last_n:].values, 
            'o-', label='Historical Data', linewidth=2, markersize=4, color='navy')
    
    # Forecast
    forecast_index = pd.date_range(start=historical_series.index[-1], 
                                   periods=len(forecast_df)+1, freq='D')[1:]
    ax.plot(forecast_index, forecast_df['mean'], 'o-', label='Forecast', 
            linewidth=2, markersize=5, color='darkred')
    
    # Confidence intervals
    ax.fill_between(forecast_index, 
                    forecast_df['mean_ci_lower'], 
                    forecast_df['mean_ci_upper'], 
                    alpha=0.2, color='darkred', label='95% Confidence Interval')
    
    ax.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    ax.set_xlabel('Date', fontsize=11)
    ax.set_ylabel('Series Value', fontsize=11)
    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.legend(fontsize=10, loc='best')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


# ============================================================================
# SECTION 7: ROLLING WINDOW BACKTESTING
# ============================================================================

def rolling_forecast_arima(series, train_window, forecast_steps, order, desc=''):
    """
    Rolling window forecast for backtesting
    Simulates real-time forecasting with expanding/rolling windows
    """
    rolling_forecasts = []
    rolling_actuals = []
    rolling_ci_lower = []
    rolling_ci_upper = []
    dates_forecast = []
    
    n = len(series)
    
    print(f"\nRolling window forecast ({desc}):")
    print(f"{'─'*60}")
    print(f"Training window: {train_window}")
    print(f"Forecast horizon: {forecast_steps}")
    print(f"Total iterations: {n - train_window - forecast_steps + 1}")
    
    for i, start_idx in enumerate(range(train_window, n - forecast_steps)):
        
        if (i + 1) % max(1, (n - train_window - forecast_steps) // 5) == 0:
            print(f"Progress: {i+1}/{n - train_window - forecast_steps}")
        
        # Training window
        train_data = series.iloc[start_idx-train_window:start_idx]
        
        try:
            # Fit ARIMA model
            model = ARIMA(train_data, order=order)
            results = model.fit()
            
            # Forecast
            forecast = results.get_forecast(steps=forecast_steps)
            forecast_values = forecast.predicted_mean.values
            forecast_ci = forecast.conf_int()
            
            # Store results
            rolling_forecasts.append(forecast_values[forecast_steps-1])
            rolling_actuals.append(series.iloc[start_idx + forecast_steps - 1])
            rolling_ci_lower.append(forecast_ci.iloc[forecast_steps-1, 0])
            rolling_ci_upper.append(forecast_ci.iloc[forecast_steps-1, 1])
            dates_forecast.append(series.index[start_idx])
            
        except Exception as e:
            continue
    
    # Create results dataframe
    rolling_results = pd.DataFrame({
        'Date': dates_forecast,
        'Forecast': rolling_forecasts,
        'Actual': rolling_actuals,
        'CI_Lower': rolling_ci_lower,
        'CI_Upper': rolling_ci_upper,
        'Error': np.array(rolling_actuals) - np.array(rolling_forecasts),
        'Within_CI': (np.array(rolling_actuals) >= np.array(rolling_ci_lower)) & 
                     (np.array(rolling_actuals) <= np.array(rolling_ci_upper))
    })
    
    return rolling_results


def evaluate_rolling_forecast(rolling_results):
    """
    Comprehensive evaluation of rolling forecast performance
    """
    print("\n" + "="*70)
    print("ROLLING FORECAST EVALUATION")
    print("="*70)
    
    # Metrics
    rmse = np.sqrt(mean_squared_error(rolling_results['Actual'], rolling_results['Forecast']))
    mae = mean_absolute_error(rolling_results['Actual'], rolling_results['Forecast'])
    mape = mean_absolute_percentage_error(rolling_results['Actual'], rolling_results['Forecast'])
    coverage = rolling_results['Within_CI'].sum() / len(rolling_results)
    
    print(f"\nForecast Accuracy Metrics:")
    print(f"  RMSE (Root Mean Squared Error): {rmse:.6f}")
    print(f"  MAE (Mean Absolute Error):      {mae:.6f}")
    print(f"  MAPE (Mean Absolute % Error):   {mape:.4%}")
    
    print(f"\nConfidence Interval Analysis:")
    print(f"  Coverage Ratio:                 {coverage:.2%}")
    print(f"  Expected Coverage (95% CI):     95.00%")
    print(f"  Difference:                     {(coverage - 0.95)*100:+.2f}%")
    
    if abs(coverage - 0.95) < 0.05:
        print(f"  ✓ Coverage ratio is within acceptable range")
    else:
        print(f"  ⚠ Coverage ratio deviates from expected (model may be misspecified)")
    
    # Error statistics
    print(f"\nError Distribution:")
    print(f"  Mean Error:                     {rolling_results['Error'].mean():.6f}")
    print(f"  Std Dev of Errors:              {rolling_results['Error'].std():.6f}")
    print(f"  Min Error:                      {rolling_results['Error'].min():.6f}")
    print(f"  Max Error:                      {rolling_results['Error'].max():.6f}")


def plot_rolling_forecast(rolling_results):
    """
    Visualization of rolling forecast results
    """
    fig, axes = plt.subplots(2, 1, figsize=(16, 10))
    
    # Time series plot
    axes[0].plot(rolling_results['Date'], rolling_results['Actual'], 
                'o-', label='Actual', color='navy', markersize=4, linewidth=1.5)
    axes[0].plot(rolling_results['Date'], rolling_results['Forecast'], 
                's--', label='Forecast', color='darkred', markersize=4, linewidth=1.5)
    axes[0].fill_between(rolling_results['Date'], 
                        rolling_results['CI_Lower'], 
                        rolling_results['CI_Upper'], 
                        alpha=0.2, color='darkred', label='95% Confidence Interval')
    axes[0].set_ylabel('Series Value', fontsize=11)
    axes[0].set_title('Rolling Window Forecast: Actual vs Predicted', fontsize=12, fontweight='bold')
    axes[0].legend(fontsize=10, loc='best')
    axes[0].grid(True, alpha=0.3)
    
    # Error distribution
    colors = ['green' if x < 0 else 'red' for x in rolling_results['Error']]
    axes[1].bar(rolling_results['Date'], rolling_results['Error'], color=colors, alpha=0.6, width=1)
    axes[1].axhline(y=0, color='black', linestyle='-', linewidth=1)
    axes[1].set_xlabel('Date', fontsize=11)
    axes[1].set_ylabel('Forecast Error', fontsize=11)
    axes[1].set_title('Forecast Errors Distribution', fontsize=12, fontweight='bold')
    axes[1].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.show()


# ============================================================================
# SECTION 8: PRODUCTION-READY CLASS
# ============================================================================

class ARIMAForecastingEngine:
    """
    Production-ready ARIMA forecasting system for financial applications
    """
    
    def __init__(self, series, model_name='ARIMA_Model', order=None):
        """
        Initialize the forecasting engine
        
        Parameters:
        -----------
        series : pd.Series
            Time series data
        model_name : str
            Name identifier for the model
        order : tuple
            ARIMA(p,d,q) order specification
        """
        self.series = series
        self.model_name = model_name
        self.order = order
        self.model = None
        self.results = None
        self.is_stationary = None
        self.last_fit_date = None
        
    def check_stationarity(self, alpha=0.05, verbose=True):
        """
        Conduct ADF test for stationarity
        """
        adf_result = adfuller(self.series.dropna(), autolag='AIC')
        self.is_stationary = adf_result[1] < alpha
        
        if verbose:
            print(f"\n{self.model_name}: Stationarity Check")
            print(f"  ADF p-value: {adf_result[1]:.6f}")
            print(f"  Stationary: {'Yes ✓' if self.is_stationary else 'No ✗'}")
        
        return self.is_stationary
    
    def fit_model(self, verbose=True):
        """
        Fit ARIMA model to data
        """
        if self.order is None:
            raise ValueError("ARIMA order must be specified before fitting")
        
        self.model = ARIMA(self.series.dropna(), order=self.order)
        self.results = self.model.fit()
        self.last_fit_date = datetime.now()
        
        if verbose:
            print(f"\n{self.model_name} Fitted Successfully")
            print(f"  Order: ARIMA{self.order}")
            print(f"  AIC: {self.results.aic:.2f}")
            print(f"  BIC: {self.results.bic:.2f}")
            print(f"  Fit Date: {self.last_fit_date.strftime('%Y-%m-%d %H:%M:%S')}")
    
    def forecast(self, steps=10, alpha=0.05):
        """
        Generate forecasts with confidence intervals
        """
        if self.results is None:
            self.fit_model(verbose=False)
        
        forecast_obj = self.results.get_forecast(steps=steps)
        forecast_df = forecast_obj.summary_frame(alpha=alpha)
        
        return forecast_df
    
    def backtest(self, test_size=0.2, verbose=True):
        """
        Backtesting on hold-out test set
        """
        n = len(self.series)
        train_size = int(n * (1 - test_size))
        
        train = self.series.iloc[:train_size]
        test = self.series.iloc[train_size:]
        
        # Fit on training set
        model = ARIMA(train, order=self.order)
        results = model.fit()
        
        # Forecast
        forecast = results.get_forecast(steps=len(test)).predicted_mean
        
        # Metrics
        rmse = np.sqrt(mean_squared_error(test, forecast))
        mae = mean_absolute_error(test, forecast)
        mape = mean_absolute_percentage_error(test, forecast)
        
        if verbose:
            print(f"\n{self.model_name} Backtest Results")
            print(f"  Train size: {train_size}, Test size: {len(test)}")
            print(f"  RMSE: {rmse:.6f}")
            print(f"  MAE: {mae:.6f}")
            print(f"  MAPE: {mape:.4%}")
        
        return {
            'rmse': rmse,
            'mae': mae,
            'mape': mape,
            'test': test,
            'forecast': forecast
        }
    
    def get_summary(self):
        """
        Return model summary
        """
        if self.results is None:
            raise ValueError("Model not fitted yet")
        
        return self.results.summary()


# ============================================================================
# MAIN EXECUTION EXAMPLE
# ============================================================================

if __name__ == "__main__":
    
    print("\n" + "="*70)
    print("ARIMA MODELING - COMPREHENSIVE IMPLEMENTATION")
    print("Prof. V. Ravichandran - The Mountain Path: World of Finance")
    print("="*70)
    
    # 1. Data Loading
    print("\n[1] LOADING DATA")
    data = load_financial_data(n_obs=500, stock_type='equity')
    explore_data(data, 'Stock Prices')
    
    # 2. Stationarity Testing
    print("\n[2] STATIONARITY TESTING")
    stationarity_analysis(data)
    
    # 3. ACF/PACF Analysis
    print("\n[3] ACF/PACF ANALYSIS")
    plot_acf_pacf(data['Log_Returns'].dropna(), lags=40, title='Log Returns')
    acf_pacf_interpretation(data['Log_Returns'].dropna(), lags=15)
    
    # 4. Model Selection
    print("\n[4] MODEL SELECTION")
    auto_model = find_optimal_arima(data['Log_Returns'].dropna(), seasonal=False)
    
    # 5. Model Comparison
    print("\n[5] MODEL COMPARISON")
    candidates = [(0,0,0), (1,0,0), (0,0,1), (1,0,1), (2,0,2)]
    results_list, best_result = compare_models(data['Log_Returns'].dropna(), candidates)
    
    # 6. Diagnostics
    print("\n[6] DIAGNOSTIC ANALYSIS")
    diagnostic_plots(best_result['Model'])
    residual_analysis(best_result['Model'], name=f"ARIMA{best_result['Order']}")
    
    # 7. Forecasting
    print("\n[7] FORECASTING")
    forecast_df = forecast_arima(best_result['Model'], steps=30)
    print(f"\nForecast Summary (first 10 periods):")
    print(forecast_df[['mean', 'mean_ci_lower', 'mean_ci_upper']].head(10))
    plot_forecast(data['Log_Returns'], forecast_df, last_n=100, 
                  title=f"ARIMA{best_result['Order']} Forecast - 30-Day Ahead")
    
    # 8. Rolling Window Backtesting
    print("\n[8] ROLLING WINDOW BACKTESTING")
    rolling_results = rolling_forecast_arima(
        series=data['Log_Returns'].dropna(),
        train_window=200,
        forecast_steps=5,
        order=best_result['Order'],
        desc="Stock Returns Forecast"
    )
    evaluate_rolling_forecast(rolling_results)
    plot_rolling_forecast(rolling_results)
    
    # 9. Production Engine
    print("\n[9] PRODUCTION FORECASTING ENGINE")
    engine = ARIMAForecastingEngine(
        series=data['Log_Returns'].dropna(),
        model_name='Production_Stock_Returns',
        order=best_result['Order']
    )
    engine.check_stationarity()
    engine.fit_model()
    backtest_results = engine.backtest()
    
    print("\n" + "="*70)
    print("ARIMA ANALYSIS COMPLETE")
    print("="*70)
