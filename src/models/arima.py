"""
ARIMA Modeling Module - src/models/arima.py

Comprehensive ARIMA (Box-Jenkins) implementation for time series forecasting.
Handles parameter selection, model fitting, and forecasting.

Author: Prof. V. Ravichandran
The Mountain Path - World of Finance
28+ Years Corporate Finance & Banking
10+ Years Academic Excellence
"""

import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller, kpss
import pmdarima as auto_arima
from typing import Tuple, Optional, Dict, List
import logging
import warnings

warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ARIMAModel:
    """
    ARIMA (AutoRegressive Integrated Moving Average) Model.
    
    Implements Box-Jenkins methodology for time series forecasting.
    
    Features:
    - Automatic parameter selection via pmdarima
    - Manual parameter fitting
    - Forecast generation
    - Residual analysis
    - Model diagnostics
    """
    
    def __init__(self, series: pd.Series, name: str = "ARIMA"):
        """
        Initialize ARIMA model.
        
        Args:
            series: Time series data (pandas Series)
            name: Model name for logging
        """
        self.series = series.copy()
        self.name = name
        self.model = None
        self.fitted_model = None
        self.forecast_df = None
        self.aic = None
        self.bic = None
        self.parameters = None
        
        logger.info(f"ARIMAModel initialized with {len(series)} observations")
    
    def check_stationarity(self) -> Dict[str, any]:
        """
        Check if series is stationary using ADF and KPSS tests.
        
        Returns:
            Dictionary with test results
        """
        logger.info("Testing stationarity...")
        
        results = {}
        
        # Augmented Dickey-Fuller Test
        adf_result = adfuller(self.series, autolag='AIC')
        results['adf'] = {
            'statistic': adf_result[0],
            'p_value': adf_result[1],
            'critical_values': adf_result[4],
            'is_stationary': adf_result[1] < 0.05
        }
        
        # KPSS Test
        try:
            kpss_result = kpss(self.series, regression='c', nlags='auto')
            results['kpss'] = {
                'statistic': kpss_result[0],
                'p_value': kpss_result[1],
                'critical_values': kpss_result[3],
                'is_stationary': kpss_result[1] > 0.05
            }
        except Exception as e:
            logger.warning(f"KPSS test failed: {str(e)}")
            results['kpss'] = None
        
        # Summary
        is_stationary = results['adf']['is_stationary']
        results['summary'] = {
            'is_stationary': is_stationary,
            'recommendation': 'd=0' if is_stationary else 'd=1 (or more)'
        }
        
        logger.info(f"Stationarity: {is_stationary} - Recommend d={0 if is_stationary else 1}")
        
        return results
    
    def auto_select_parameters(
        self,
        max_p: int = 5,
        max_d: int = 2,
        max_q: int = 5,
        seasonal: bool = False,
        m: int = 12
    ) -> Tuple[Tuple[int, int, int], float, float]:
        """
        Auto-select ARIMA parameters using auto_arima.
        
        Args:
            max_p: Maximum AR order
            max_d: Maximum differencing order
            max_q: Maximum MA order
            seasonal: Whether to include seasonal component
            m: Seasonal period (12 for monthly, 252 for daily stocks)
        
        Returns:
            Tuple of (parameters, AIC, BIC)
        """
        logger.info(f"Auto-selecting parameters (p:{max_p}, d:{max_d}, q:{max_q})...")
        
        try:
            stepwise_model = auto_arima.auto_arima(
                self.series,
                start_p=0, start_q=0, start_P=0, start_Q=0,
                max_p=max_p, max_d=max_d, max_q=max_q,
                max_P=2 if seasonal else 0,
                max_D=1 if seasonal else 0,
                max_Q=2 if seasonal else 0,
                m=m if seasonal else 1,
                seasonal=seasonal,
                stepwise=True,
                trace=False,
                error_action='ignore',
                suppress_warnings=True,
                random_state=42,
                n_fits=50
            )
            
            order = stepwise_model.order
            seasonal_order = stepwise_model.seasonal_order if seasonal else (0, 0, 0, 0)
            
            self.parameters = {
                'order': order,
                'seasonal_order': seasonal_order,
                'aic': stepwise_model.aic(),
                'bic': stepwise_model.bic()
            }
            
            logger.info(f"Selected parameters - ARIMA{order}")
            if seasonal:
                logger.info(f"Seasonal order: {seasonal_order}")
            
            return order, stepwise_model.aic(), stepwise_model.bic()
        
        except Exception as e:
            logger.error(f"Error in parameter selection: {str(e)}")
            raise
    
    def fit(
        self,
        order: Tuple[int, int, int],
        seasonal_order: Optional[Tuple[int, int, int, int]] = None
    ) -> Dict:
        """
        Fit ARIMA model with specified parameters.
        
        Args:
            order: ARIMA (p, d, q) parameters
            seasonal_order: Seasonal (P, D, Q, m) parameters (optional)
        
        Returns:
            Dictionary with model statistics
        """
        logger.info(f"Fitting ARIMA{order} model...")
        
        try:
            if seasonal_order:
                self.model = ARIMA(
                    self.series,
                    order=order,
                    seasonal_order=seasonal_order
                )
            else:
                self.model = ARIMA(self.series, order=order)
            
            self.fitted_model = self.model.fit()
            
            self.aic = self.fitted_model.aic
            self.bic = self.fitted_model.bic
            
            logger.info(f"Model fitted successfully")
            logger.info(f"AIC: {self.aic:.2f}, BIC: {self.bic:.2f}")
            
            return {
                'aic': self.aic,
                'bic': self.bic,
                'rsquared': self.fitted_model.rsquared,
                'parameters': order
            }
        
        except Exception as e:
            logger.error(f"Error fitting model: {str(e)}")
            raise
    
    def forecast(
        self,
        steps: int = 10,
        alpha: float = 0.05
    ) -> pd.DataFrame:
        """
        Generate forecast for future periods.
        
        Args:
            steps: Number of periods to forecast
            alpha: Confidence level for intervals (0.05 = 95%)
        
        Returns:
            DataFrame with forecasts and confidence intervals
        """
        if self.fitted_model is None:
            raise ValueError("Model not fitted. Call fit() first.")
        
        logger.info(f"Generating {steps}-step forecast...")
        
        try:
            # Get forecast
            forecast_result = self.fitted_model.get_forecast(steps=steps)
            forecast_df = forecast_result.conf_int(alpha=alpha)
            forecast_df['forecast'] = forecast_result.predicted_mean
            
            # Rename columns
            forecast_df.columns = ['lower_ci', 'upper_ci', 'forecast']
            forecast_df = forecast_df[['forecast', 'lower_ci', 'upper_ci']]
            
            # Create future dates
            last_date = self.series.index[-1]
            freq = pd.infer_freq(self.series.index)
            future_dates = pd.date_range(start=last_date, periods=steps+1, freq=freq)[1:]
            forecast_df.index = future_dates
            
            self.forecast_df = forecast_df
            
            logger.info(f"Forecast generated for {steps} periods")
            
            return forecast_df
        
        except Exception as e:
            logger.error(f"Error generating forecast: {str(e)}")
            raise
    
    def get_residuals(self) -> pd.Series:
        """
        Get model residuals.
        
        Returns:
            Series of residuals
        """
        if self.fitted_model is None:
            raise ValueError("Model not fitted. Call fit() first.")
        
        return self.fitted_model.resid
    
    def get_summary(self) -> str:
        """
        Get model summary statistics.
        
        Returns:
            Model summary string
        """
        if self.fitted_model is None:
            raise ValueError("Model not fitted. Call fit() first.")
        
        return str(self.fitted_model.summary())
    
    def plot_diagnostics(self, figsize: Tuple[int, int] = (12, 8)):
        """
        Plot diagnostic plots (requires matplotlib).
        
        Args:
            figsize: Figure size (width, height)
        """
        if self.fitted_model is None:
            raise ValueError("Model not fitted. Call fit() first.")
        
        try:
            self.fitted_model.plot_diagnostics(figsize=figsize)
            logger.info("Diagnostic plots generated")
        except Exception as e:
            logger.error(f"Error generating diagnostics: {str(e)}")
    
    def calculate_metrics(self) -> Dict:
        """
        Calculate model performance metrics.
        
        Returns:
            Dictionary with metrics
        """
        if self.fitted_model is None:
            raise ValueError("Model not fitted. Call fit() first.")
        
        residuals = self.fitted_model.resid
        
        metrics = {
            'aic': self.aic,
            'bic': self.bic,
            'rmse': np.sqrt(np.mean(residuals**2)),
            'mae': np.mean(np.abs(residuals)),
            'residual_mean': np.mean(residuals),
            'residual_std': np.std(residuals),
            'ljung_box_pvalue': self.fitted_model.plot_diagnostics()[3]  # Approx
        }
        
        return metrics


class ARIMAEnsemble:
    """
    Ensemble of multiple ARIMA models for robust forecasting.
    
    Combines multiple parameter configurations for better predictions.
    """
    
    def __init__(self, series: pd.Series):
        """Initialize ensemble."""
        self.series = series
        self.models = []
        self.weights = []
    
    def fit_multiple(
        self,
        parameter_list: List[Tuple[int, int, int]]
    ) -> None:
        """
        Fit multiple ARIMA models.
        
        Args:
            parameter_list: List of (p, d, q) tuples
        """
        aics = []
        
        for order in parameter_list:
            try:
                model = ARIMAModel(self.series)
                model.fit(order)
                self.models.append(model)
                aics.append(model.aic)
            except:
                pass
        
        # Weight by inverse AIC
        aics = np.array(aics)
        self.weights = (1 / aics) / np.sum(1 / aics)
        
        logger.info(f"Fitted {len(self.models)} models with weights")
    
    def forecast(self, steps: int = 10) -> pd.DataFrame:
        """
        Generate weighted ensemble forecast.
        
        Args:
            steps: Number of periods to forecast
        
        Returns:
            Ensemble forecast DataFrame
        """
        forecasts = []
        
        for model in self.models:
            forecast = model.forecast(steps=steps)
            forecasts.append(forecast['forecast'].values)
        
        forecasts = np.array(forecasts)
        
        # Weighted average
        ensemble_forecast = np.average(forecasts, axis=0, weights=self.weights)
        
        result_df = pd.DataFrame({
            'forecast': ensemble_forecast,
            'models_count': len(self.models)
        })
        
        return result_df


# Example usage
if __name__ == "__main__":
    from src.data.cache import CachedDataLoader, DataCache
    from src.data.loader import DataLoader
    
    # Fetch data
    cache = DataCache()
    loader = DataLoader()
    cached_loader = CachedDataLoader(loader, cache)
    
    df = cached_loader.fetch_with_cache("^NSEI", start_date="2023-01-01")
    series = loader.prepare_series(df, column="Close")
    
    # Create and fit model
    arima_model = ARIMAModel(series, name="NIFTY")
    
    # Check stationarity
    stationarity = arima_model.check_stationarity()
    print(f"\nStationarity: {stationarity['summary']}")
    
    # Auto-select parameters
    order, aic, bic = arima_model.auto_select_parameters(max_p=5, max_d=2, max_q=5)
    print(f"\nBest parameters: {order}")
    print(f"AIC: {aic:.2f}, BIC: {bic:.2f}")
    
    # Fit model
    arima_model.fit(order)
    
    # Forecast
    forecast_df = arima_model.forecast(steps=30)
    print(f"\n30-day forecast:\n{forecast_df.head(10)}")
    
    # Metrics
    metrics = arima_model.calculate_metrics()
    print(f"\nModel metrics:")
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")

