"""
Diagnostics Module - src/models/diagnostics.py

Statistical diagnostics and model validation for ARIMA models.
Performs tests for residual analysis, forecast validation, and model assumptions.

Author: Prof. V. Ravichandran
The Mountain Path - World of Finance
28+ Years Corporate Finance & Banking
10+ Years Academic Excellence
"""

import pandas as pd
import numpy as np
from scipy import stats
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.stattools import adfuller, kpss, acf, pacf
from typing import Dict, Tuple, Optional
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ResidualDiagnostics:
    """
    Comprehensive residual diagnostics for ARIMA models.
    
    Tests:
    - Normality (Shapiro-Wilk, Jarque-Bera)
    - Autocorrelation (Ljung-Box)
    - Heteroscedasticity (ARCH)
    - Mean (t-test)
    """
    
    def __init__(self, residuals: pd.Series):
        """
        Initialize diagnostics.
        
        Args:
            residuals: Series of model residuals
        """
        self.residuals = residuals
        self.n = len(residuals)
        logger.info(f"ResidualDiagnostics initialized with {self.n} residuals")
    
    def test_normality(self) -> Dict:
        """
        Test for normality of residuals.
        
        Uses Shapiro-Wilk and Jarque-Bera tests.
        Null hypothesis: Residuals are normally distributed
        
        Returns:
            Dictionary with test results
        """
        logger.info("Testing normality...")
        
        results = {}
        
        # Shapiro-Wilk Test (best for n < 5000)
        if self.n < 5000:
            shapiro_stat, shapiro_pval = stats.shapiro(self.residuals)
            results['shapiro_wilk'] = {
                'statistic': shapiro_stat,
                'p_value': shapiro_pval,
                'is_normal': shapiro_pval > 0.05
            }
        
        # Jarque-Bera Test
        jb_stat, jb_pval = stats.jarque_bera(self.residuals)
        results['jarque_bera'] = {
            'statistic': jb_stat,
            'p_value': jb_pval,
            'is_normal': jb_pval > 0.05
        }
        
        # Skewness and Kurtosis
        skewness = stats.skew(self.residuals)
        kurtosis = stats.kurtosis(self.residuals)
        results['skewness'] = skewness
        results['kurtosis'] = kurtosis
        results['interpretation'] = {
            'skewness_interpretation': 'Normal' if abs(skewness) < 0.5 else 'Skewed',
            'kurtosis_interpretation': 'Normal' if abs(kurtosis) < 3 else 'Heavy-tailed'
        }
        
        logger.info(f"Normality - JB p-value: {jb_pval:.4f} {'✓' if jb_pval > 0.05 else '✗'}")
        
        return results
    
    def test_autocorrelation(self, lags: int = 20) -> Dict:
        """
        Test for autocorrelation in residuals using Ljung-Box test.
        
        Null hypothesis: No autocorrelation at specified lags
        
        Args:
            lags: Number of lags to test (default 20)
        
        Returns:
            Dictionary with test results
        """
        logger.info(f"Testing autocorrelation (lags={lags})...")
        
        try:
            # Ljung-Box test
            lb_result = acorr_ljungbox(self.residuals, lags=lags, return_df=True)
            
            # Check if autocorrelated
            is_autocorrelated = (lb_result['lb_pvalue'] < 0.05).any()
            
            results = {
                'ljung_box': lb_result.to_dict(),
                'is_autocorrelated': is_autocorrelated,
                'num_lags_significant': (lb_result['lb_pvalue'] < 0.05).sum()
            }
            
            logger.info(f"Autocorrelation - {results['num_lags_significant']}/{lags} lags significant")
            
            return results
        
        except Exception as e:
            logger.error(f"Error in autocorrelation test: {str(e)}")
            return {}
    
    def test_mean(self) -> Dict:
        """
        Test if residual mean is statistically zero.
        
        Uses one-sample t-test.
        Null hypothesis: Mean = 0
        
        Returns:
            Dictionary with test results
        """
        logger.info("Testing residual mean...")
        
        # One-sample t-test
        t_stat, p_value = stats.ttest_1samp(self.residuals, 0)
        
        results = {
            'mean': self.residuals.mean(),
            'std': self.residuals.std(),
            't_statistic': t_stat,
            'p_value': p_value,
            'mean_is_zero': p_value > 0.05
        }
        
        logger.info(f"Mean test - mean={results['mean']:.6f}, p-value={p_value:.4f}")
        
        return results
    
    def test_heteroscedasticity(self) -> Dict:
        """
        Test for heteroscedasticity (changing variance).
        
        Uses White test approximation.
        
        Returns:
            Dictionary with test results
        """
        logger.info("Testing heteroscedasticity...")
        
        # Simple test: split residuals in half and compare variances
        n = len(self.residuals)
        first_half = self.residuals[:n//2]
        second_half = self.residuals[n//2:]
        
        # Levene's test
        levene_stat, levene_pval = stats.levene(first_half, second_half)
        
        # F-test
        var_ratio = second_half.var() / first_half.var()
        f_stat = max(var_ratio, 1/var_ratio)
        f_pval = 2 * (1 - stats.f.cdf(f_stat, n//2-1, n//2-1))
        
        results = {
            'levene_test': {
                'statistic': levene_stat,
                'p_value': levene_pval,
                'is_homoscedastic': levene_pval > 0.05
            },
            'f_test': {
                'statistic': f_stat,
                'p_value': f_pval,
                'variance_ratio': var_ratio
            }
        }
        
        logger.info(f"Heteroscedasticity - Levene p-value: {levene_pval:.4f}")
        
        return results
    
    def get_summary(self) -> Dict:
        """
        Get comprehensive diagnostic summary.
        
        Returns:
            Dictionary with all diagnostic tests
        """
        summary = {
            'normality': self.test_normality(),
            'autocorrelation': self.test_autocorrelation(),
            'mean': self.test_mean(),
            'heteroscedasticity': self.test_heteroscedasticity()
        }
        
        return summary
    
    def print_summary(self) -> None:
        """Print diagnostic summary to console."""
        summary = self.get_summary()
        
        print("\n" + "="*70)
        print("RESIDUAL DIAGNOSTICS SUMMARY")
        print("="*70)
        
        # Normality
        print("\n1. NORMALITY TESTS")
        print("-" * 70)
        if 'shapiro_wilk' in summary['normality']:
            sw = summary['normality']['shapiro_wilk']
            print(f"   Shapiro-Wilk: p-value={sw['p_value']:.4f} {'✓' if sw['is_normal'] else '✗'}")
        
        jb = summary['normality']['jarque_bera']
        print(f"   Jarque-Bera:  p-value={jb['p_value']:.4f} {'✓' if jb['is_normal'] else '✗'}")
        print(f"   Skewness: {summary['normality']['skewness']:.4f}")
        print(f"   Kurtosis: {summary['normality']['kurtosis']:.4f}")
        
        # Autocorrelation
        print("\n2. AUTOCORRELATION TEST (Ljung-Box)")
        print("-" * 70)
        lb = summary['autocorrelation']
        if lb:
            print(f"   Significant lags: {lb['num_lags_significant']}/20")
            print(f"   Autocorrelated: {'✗ No' if not lb['is_autocorrelated'] else '✓ Yes'}")
        
        # Mean
        print("\n3. MEAN TEST")
        print("-" * 70)
        mean = summary['mean']
        print(f"   Mean: {mean['mean']:.6f}")
        print(f"   Std:  {mean['std']:.6f}")
        print(f"   p-value: {mean['p_value']:.4f}")
        print(f"   Mean = 0: {'✓ Yes' if mean['mean_is_zero'] else '✗ No'}")
        
        # Heteroscedasticity
        print("\n4. HETEROSCEDASTICITY TEST")
        print("-" * 70)
        hetero = summary['heteroscedasticity']
        print(f"   Levene p-value: {hetero['levene_test']['p_value']:.4f}")
        print(f"   F-test p-value: {hetero['f_test']['p_value']:.4f}")
        print(f"   Homoscedastic: {'✓ Yes' if hetero['levene_test']['is_homoscedastic'] else '✗ No'}")
        
        print("\n" + "="*70 + "\n")


class ForecastDiagnostics:
    """
    Forecast validation and accuracy metrics.
    
    Calculates:
    - RMSE, MAE, MAPE
    - Theil's U statistic
    - Directional accuracy
    """
    
    def __init__(self, actual: pd.Series, predicted: pd.Series):
        """
        Initialize forecast diagnostics.
        
        Args:
            actual: Actual values
            predicted: Predicted values
        """
        self.actual = actual
        self.predicted = predicted
        self.errors = actual - predicted
        
        logger.info(f"ForecastDiagnostics initialized with {len(actual)} observations")
    
    def calculate_metrics(self) -> Dict:
        """
        Calculate forecast accuracy metrics.
        
        Returns:
            Dictionary with metrics
        """
        logger.info("Calculating forecast metrics...")
        
        # Basic metrics
        mae = np.mean(np.abs(self.errors))
        rmse = np.sqrt(np.mean(self.errors**2))
        
        # MAPE (Mean Absolute Percentage Error)
        mape = np.mean(np.abs(self.errors / self.actual)) * 100
        
        # MASE (Mean Absolute Scaled Error)
        naive_forecast_error = np.abs(self.actual.diff().dropna()).mean()
        mase = mae / naive_forecast_error if naive_forecast_error != 0 else np.inf
        
        # Theil's U
        numerator = np.sum(self.errors**2)
        denominator = np.sum(self.actual**2) + np.sum(self.predicted**2)
        theils_u = np.sqrt(numerator / denominator)
        
        # Directional accuracy
        actual_direction = (self.actual.diff() > 0).astype(int)
        predicted_direction = (self.predicted.diff() > 0).astype(int)
        directional_accuracy = (actual_direction == predicted_direction).mean() * 100
        
        metrics = {
            'mae': mae,
            'rmse': rmse,
            'mape': mape,
            'mase': mase,
            'theils_u': theils_u,
            'directional_accuracy': directional_accuracy,
            'mean_error': self.errors.mean(),
            'std_error': self.errors.std()
        }
        
        logger.info(f"Metrics - RMSE: {rmse:.4f}, MAPE: {mape:.2f}%")
        
        return metrics
    
    def get_summary(self) -> str:
        """
        Get forecast metrics summary string.
        
        Returns:
            Formatted summary string
        """
        metrics = self.calculate_metrics()
        
        summary = f"""
FORECAST ACCURACY METRICS
========================
MAE:                    {metrics['mae']:.4f}
RMSE:                   {metrics['rmse']:.4f}
MAPE:                   {metrics['mape']:.2f}%
MASE:                   {metrics['mase']:.4f}
Theil's U:              {metrics['theils_u']:.4f}
Directional Accuracy:   {metrics['directional_accuracy']:.2f}%
Mean Error:             {metrics['mean_error']:.6f}
Std Error:              {metrics['std_error']:.4f}
"""
        return summary


class ModelComparison:
    """
    Compare multiple ARIMA models.
    
    Supports:
    - Information criteria comparison
    - Statistical testing
    - Forecast comparison
    """
    
    def __init__(self):
        """Initialize model comparison."""
        self.models = {}
        logger.info("ModelComparison initialized")
    
    def add_model(self, name: str, model) -> None:
        """
        Add model to comparison.
        
        Args:
            name: Model name
            model: Fitted ARIMA model object
        """
        self.models[name] = model
        logger.info(f"Added model: {name}")
    
    def compare_information_criteria(self) -> pd.DataFrame:
        """
        Compare models by AIC and BIC.
        
        Returns:
            DataFrame with comparison
        """
        comparison = []
        
        for name, model in self.models.items():
            comparison.append({
                'Model': name,
                'AIC': model.aic,
                'BIC': model.bic
            })
        
        df = pd.DataFrame(comparison).sort_values('AIC')
        
        return df
    
    def get_comparison_summary(self) -> str:
        """
        Get comparison summary.
        
        Returns:
            Summary string
        """
        df = self.compare_information_criteria()
        
        summary = "\nMODEL COMPARISON\n"
        summary += "=" * 50 + "\n"
        summary += df.to_string(index=False)
        summary += "\n" + "=" * 50 + "\n"
        
        return summary


# Example usage
if __name__ == "__main__":
    import pandas as pd
    from src.models.arima import ARIMAModel
    from src.data.cache import CachedDataLoader, DataCache
    from src.data.loader import DataLoader
    
    # Fetch data
    cache = DataCache()
    loader = DataLoader()
    cached_loader = CachedDataLoader(loader, cache)
    
    df = cached_loader.fetch_with_cache("^NSEI", start_date="2023-01-01")
    series = loader.prepare_series(df, column="Close")
    
    # Fit model
    model = ARIMAModel(series)
    order, _, _ = model.auto_select_parameters()
    model.fit(order)
    
    # Residual diagnostics
    residuals = model.get_residuals()
    diag = ResidualDiagnostics(residuals)
    diag.print_summary()
    
    # Forecast diagnostics
    train_size = int(len(series) * 0.8)
    train_series = series[:train_size]
    test_series = series[train_size:]
    
    # Refit on training data
    train_model = ARIMAModel(train_series)
    train_model.fit(order)
    
    # Generate forecast
    forecast = train_model.forecast(steps=len(test_series))
    
    # Compare
    forecast_diag = ForecastDiagnostics(test_series, forecast['forecast'])
    print(forecast_diag.get_summary())

