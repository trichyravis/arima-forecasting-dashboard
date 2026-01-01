"""
Visualization Diagnostics Module - src/visualization/diagnostics.py

Interactive diagnostic visualizations for ARIMA model validation.
Integrated with Streamlit for dashboard display.

Author: Prof. V. Ravichandran
The Mountain Path - World of Finance
28+ Years Corporate Finance & Banking
10+ Years Academic Excellence
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import matplotlib.pyplot as plt
from typing import Optional, Dict, Tuple
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TimeSeriesDiagnostics:
    """
    Time series analysis diagnostics.
    
    Provides visualization for:
    - Data quality
    - Trend analysis
    - Seasonal patterns
    - Stationarity indicators
    """
    
    @staticmethod
    def plot_data_quality(
        series: pd.Series,
        title: str = "Data Quality Analysis"
    ) -> go.Figure:
        """
        Analyze and visualize data quality.
        
        Args:
            series: Time series data
            title: Plot title
        
        Returns:
            Plotly figure
        """
        fig = make_subplots(
            rows=3, cols=1,
            subplot_titles=(
                'Time Series',
                'Missing Values',
                'Statistics'
            ),
            specs=[[{}], [{}], [{"type": "table"}]]
        )
        
        # Time series
        fig.add_trace(
            go.Scatter(
                x=series.index,
                y=series.values,
                mode='lines',
                name='Series',
                line=dict(color='#003366', width=2)
            ),
            row=1, col=1
        )
        
        # Missing values indicator
        missing_mask = series.isnull().astype(int)
        fig.add_trace(
            go.Scatter(
                x=series.index,
                y=missing_mask,
                fill='tozeroy',
                name='Missing',
                fillcolor='rgba(255, 107, 107, 0.3)',
                line=dict(color='rgba(255, 107, 107, 0)'),
                yaxis='y2'
            ),
            row=2, col=1
        )
        
        # Statistics table
        stats = {
            'Count': len(series),
            'Missing': series.isnull().sum(),
            'Missing %': f"{series.isnull().sum()/len(series)*100:.2f}%",
            'Mean': f"{series.mean():.4f}",
            'Std': f"{series.std():.4f}",
            'Min': f"{series.min():.4f}",
            'Max': f"{series.max():.4f}"
        }
        
        fig.add_trace(
            go.Table(
                header=dict(
                    values=['<b>Statistic</b>', '<b>Value</b>'],
                    fill_color='#003366',
                    font=dict(color='white')
                ),
                cells=dict(
                    values=[list(stats.keys()), list(stats.values())],
                    fill_color='rgba(240, 240, 240, 0.5)',
                    font=dict(color='#003366')
                )
            ),
            row=3, col=1
        )
        
        fig.update_yaxes(title_text="Value", row=1, col=1)
        fig.update_yaxes(title_text="Missing", row=2, col=1)
        fig.update_xaxes(title_text="Date", row=2, col=1)
        
        fig.update_layout(
            title=title,
            height=900,
            showlegend=True,
            template='plotly_white'
        )
        
        logger.info(f"Created data quality plot: {title}")
        
        return fig
    
    @staticmethod
    def plot_trend_seasonality(
        series: pd.Series,
        period: int = 252,
        title: str = "Trend & Seasonality"
    ) -> go.Figure:
        """
        Decompose and visualize trend & seasonality.
        
        Args:
            series: Time series data
            period: Seasonal period (252 for daily stock data)
            title: Plot title
        
        Returns:
            Plotly figure
        """
        from statsmodels.tsa.seasonal import seasonal_decompose
        
        try:
            decomposition = seasonal_decompose(
                series,
                model='multiplicative',
                period=min(period, len(series)//2)
            )
        except:
            decomposition = seasonal_decompose(
                series,
                model='additive',
                period=min(period, len(series)//2)
            )
        
        fig = make_subplots(
            rows=4, cols=1,
            subplot_titles=('Original', 'Trend', 'Seasonal', 'Residual'),
            shared_xaxes=True,
            vertical_spacing=0.08
        )
        
        # Original
        fig.add_trace(
            go.Scatter(x=series.index, y=series.values,
                      mode='lines', name='Original',
                      line=dict(color='#003366')),
            row=1, col=1
        )
        
        # Trend
        fig.add_trace(
            go.Scatter(x=decomposition.trend.index,
                      y=decomposition.trend.values,
                      mode='lines', name='Trend',
                      line=dict(color='#FFD700')),
            row=2, col=1
        )
        
        # Seasonal
        fig.add_trace(
            go.Scatter(x=decomposition.seasonal.index,
                      y=decomposition.seasonal.values,
                      mode='lines', name='Seasonal',
                      line=dict(color='#FF6B6B')),
            row=3, col=1
        )
        
        # Residual
        fig.add_trace(
            go.Scatter(x=decomposition.resid.index,
                      y=decomposition.resid.values,
                      mode='markers', name='Residual',
                      marker=dict(color='#4CAF50')),
            row=4, col=1
        )
        
        fig.update_yaxes(title_text="Value", row=1, col=1)
        fig.update_yaxes(title_text="Trend", row=2, col=1)
        fig.update_yaxes(title_text="Seasonal", row=3, col=1)
        fig.update_yaxes(title_text="Residual", row=4, col=1)
        fig.update_xaxes(title_text="Date", row=4, col=1)
        
        fig.update_layout(
            title=title,
            height=1000,
            template='plotly_white'
        )
        
        logger.info(f"Created trend-seasonality plot: {title}")
        
        return fig
    
    @staticmethod
    def plot_returns_analysis(
        series: pd.Series,
        title: str = "Returns Analysis"
    ) -> go.Figure:
        """
        Analyze returns distribution.
        
        Args:
            series: Time series (prices)
            title: Plot title
        
        Returns:
            Plotly figure
        """
        returns = series.pct_change().dropna() * 100
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'Returns Over Time',
                'Distribution',
                'Rolling Volatility',
                'Cumulative Returns'
            ),
            specs=[[{}, {}], [{}, {}]]
        )
        
        # Returns time series
        fig.add_trace(
            go.Scatter(x=returns.index, y=returns.values,
                      mode='markers', name='Returns',
                      marker=dict(color='#003366', size=4)),
            row=1, col=1
        )
        fig.add_hline(y=0, line_dash="dash", line_color="red", row=1, col=1)
        
        # Distribution
        fig.add_trace(
            go.Histogram(x=returns.values, nbinsx=50, name='Distribution',
                        marker=dict(color='#FFD700')),
            row=1, col=2
        )
        
        # Rolling volatility
        rolling_vol = returns.rolling(window=20).std()
        fig.add_trace(
            go.Scatter(x=rolling_vol.index, y=rolling_vol.values,
                      mode='lines', name='Rolling Vol',
                      line=dict(color='#FF6B6B')),
            row=2, col=1
        )
        
        # Cumulative returns
        cum_returns = (1 + returns/100).cumprod() - 1
        fig.add_trace(
            go.Scatter(x=cum_returns.index, y=cum_returns.values * 100,
                      mode='lines', name='Cumulative',
                      fill='tozeroy', fillcolor='rgba(0, 51, 102, 0.2)',
                      line=dict(color='#003366')),
            row=2, col=2
        )
        
        fig.update_yaxes(title_text="Returns %", row=1, col=1)
        fig.update_yaxes(title_text="Frequency", row=1, col=2)
        fig.update_yaxes(title_text="Volatility %", row=2, col=1)
        fig.update_yaxes(title_text="Cumulative %", row=2, col=2)
        
        fig.update_layout(
            title=title,
            height=800,
            template='plotly_white'
        )
        
        logger.info(f"Created returns analysis: {title}")
        
        return fig


class ArimaModeDiagnostics:
    """
    ARIMA-specific diagnostic visualizations.
    
    Provides:
    - ACF/PACF plots
    - Stationarity analysis
    - Parameter effect analysis
    """
    
    @staticmethod
    def create_acf_pacf_plots(
        series: pd.Series,
        lags: int = 40,
        title: str = "ACF & PACF"
    ) -> go.Figure:
        """
        Create ACF and PACF plots using matplotlib->plotly conversion.
        
        Args:
            series: Time series
            lags: Number of lags
            title: Plot title
        
        Returns:
            Plotly figure
        """
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=('ACF', 'PACF'),
            vertical_spacing=0.12
        )
        
        # Create ACF and PACF data
        acf_vals = plot_acf(series, lags=lags, plot=False)
        pacf_vals = plot_pacf(series, lags=lags, plot=False)
        
        lags_array = np.arange(len(acf_vals[0]))
        
        # Confidence interval (95%)
        ci = 1.96 / np.sqrt(len(series))
        
        # ACF
        fig.add_trace(
            go.Bar(x=lags_array, y=acf_vals[0], name='ACF',
                  marker=dict(color='#003366')),
            row=1, col=1
        )
        fig.add_hline(y=ci, line_dash="dash", line_color="red", row=1, col=1)
        fig.add_hline(y=-ci, line_dash="dash", line_color="red", row=1, col=1)
        
        # PACF
        fig.add_trace(
            go.Bar(x=lags_array, y=pacf_vals[0], name='PACF',
                  marker=dict(color='#FFD700')),
            row=2, col=1
        )
        fig.add_hline(y=ci, line_dash="dash", line_color="red", row=2, col=1)
        fig.add_hline(y=-ci, line_dash="dash", line_color="red", row=2, col=1)
        
        fig.update_yaxes(title_text="ACF", row=1, col=1)
        fig.update_yaxes(title_text="PACF", row=2, col=1)
        fig.update_xaxes(title_text="Lags", row=2, col=1)
        
        fig.update_layout(
            title=title,
            height=700,
            template='plotly_white'
        )
        
        logger.info(f"Created ACF/PACF plot: {title}")
        
        return fig
    
    @staticmethod
    def plot_stationarity_indicators(
        series: pd.Series,
        title: str = "Stationarity Indicators"
    ) -> go.Figure:
        """
        Visualize stationarity indicators.
        
        Args:
            series: Time series
            title: Plot title
        
        Returns:
            Plotly figure
        """
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'Original Series',
                'First Difference',
                'Original - Rolling Mean',
                'Original - Rolling Std'
            ),
            specs=[[{}, {}], [{}, {}]]
        )
        
        # Original
        fig.add_trace(
            go.Scatter(x=series.index, y=series.values,
                      mode='lines', name='Original',
                      line=dict(color='#003366')),
            row=1, col=1
        )
        
        # First difference
        diff1 = series.diff().dropna()
        fig.add_trace(
            go.Scatter(x=diff1.index, y=diff1.values,
                      mode='lines', name='1st Diff',
                      line=dict(color='#FF6B6B')),
            row=1, col=2
        )
        
        # Detrended (remove rolling mean)
        rm = series.rolling(window=30).mean()
        detrended = series - rm
        fig.add_trace(
            go.Scatter(x=detrended.index, y=detrended.values,
                      mode='lines', name='Detrended',
                      line=dict(color='#FFD700')),
            row=2, col=1
        )
        
        # Remove rolling std
        rs = series.rolling(window=30).std()
        deseasonalized = series / rs
        fig.add_trace(
            go.Scatter(x=deseasonalized.index, y=deseasonalized.values,
                      mode='lines', name='Scaled',
                      line=dict(color='#4CAF50')),
            row=2, col=2
        )
        
        fig.update_layout(
            title=title,
            height=800,
            template='plotly_white'
        )
        
        logger.info(f"Created stationarity plot: {title}")
        
        return fig


class ForecastDiagnostics:
    """
    Forecast-specific diagnostic visualizations.
    
    Includes:
    - Actual vs Forecast
    - Forecast error analysis
    - Confidence interval visualization
    """
    
    @staticmethod
    def plot_forecast_uncertainty(
        forecast_df: pd.DataFrame,
        actual: Optional[pd.Series] = None,
        title: str = "Forecast with Uncertainty"
    ) -> go.Figure:
        """
        Visualize forecast with uncertainty bands.
        
        Args:
            forecast_df: Forecast DataFrame with forecast, lower_ci, upper_ci
            actual: Optional actual values for comparison
            title: Plot title
        
        Returns:
            Plotly figure
        """
        fig = go.Figure()
        
        # Add actual if provided
        if actual is not None:
            fig.add_trace(go.Scatter(
                x=actual.index, y=actual.values,
                mode='lines', name='Actual',
                line=dict(color='#003366', width=2.5)
            ))
        
        # Add forecast
        fig.add_trace(go.Scatter(
            x=forecast_df.index, y=forecast_df['forecast'],
            mode='lines+markers', name='Forecast',
            line=dict(color='#FF6B6B', width=2.5),
            marker=dict(size=6)
        ))
        
        # Add confidence interval
        fig.add_trace(go.Scatter(
            x=forecast_df.index.tolist() + forecast_df.index.tolist()[::-1],
            y=forecast_df['upper_ci'].tolist() + forecast_df['lower_ci'].tolist()[::-1],
            fill='toself',
            fillcolor='rgba(255, 107, 107, 0.2)',
            line=dict(color='rgba(255, 107, 107, 0)'),
            name='95% CI',
            hoverinfo='skip'
        ))
        
        fig.update_layout(
            title=title,
            xaxis_title='Date',
            yaxis_title='Value',
            template='plotly_white',
            height=600,
            hovermode='x unified'
        )
        
        logger.info(f"Created forecast uncertainty plot: {title}")
        
        return fig
    
    @staticmethod
    def plot_error_metrics(
        errors: pd.Series,
        title: str = "Error Analysis"
    ) -> go.Figure:
        """
        Analyze forecast errors.
        
        Args:
            errors: Forecast errors (actual - predicted)
            title: Plot title
        
        Returns:
            Plotly figure
        """
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'Error Over Time',
                'Error Distribution',
                'Cumulative Error',
                'Error Autocorrelation'
            ),
            specs=[[{}, {}], [{}, {}]]
        )
        
        # Error over time
        fig.add_trace(
            go.Scatter(x=errors.index, y=errors.values,
                      mode='lines+markers', name='Error',
                      line=dict(color='#FF6B6B')),
            row=1, col=1
        )
        fig.add_hline(y=0, line_dash="dash", line_color="red", row=1, col=1)
        
        # Distribution
        fig.add_trace(
            go.Histogram(x=errors.values, nbinsx=30, name='Distribution',
                        marker=dict(color='#FFD700')),
            row=1, col=2
        )
        
        # Cumulative
        cum_error = errors.cumsum()
        fig.add_trace(
            go.Scatter(x=cum_error.index, y=cum_error.values,
                      mode='lines', name='Cumulative',
                      fill='tozeroy', fillcolor='rgba(76, 175, 80, 0.2)',
                      line=dict(color='#4CAF50')),
            row=2, col=1
        )
        
        # Autocorrelation of errors
        acf_errors = plot_acf(errors, lags=20, plot=False)
        fig.add_trace(
            go.Bar(x=np.arange(len(acf_errors[0])), y=acf_errors[0],
                  name='ACF', marker=dict(color='#003366')),
            row=2, col=2
        )
        
        fig.update_layout(
            title=title,
            height=800,
            template='plotly_white'
        )
        
        logger.info(f"Created error analysis: {title}")
        
        return fig


# Example usage
if __name__ == "__main__":
    import pandas as pd
    import numpy as np
    
    # Generate sample data
    dates = pd.date_range('2023-01-01', periods=300)
    series = 100 + np.cumsum(np.random.randn(300) * 0.5)
    series = pd.Series(series, index=dates)
    
    print("Creating diagnostic visualizations...")
    
    # Time series diagnostics
    fig1 = TimeSeriesDiagnostics.plot_data_quality(series)
    fig2 = TimeSeriesDiagnostics.plot_returns_analysis(series)
    
    # ARIMA diagnostics
    fig3 = ArimaModeDiagnostics.create_acf_pacf_plots(series)
    fig4 = ArimaModeDiagnostics.plot_stationarity_indicators(series)
    
    # Forecast diagnostics
    forecast_df = pd.DataFrame({
        'forecast': series.iloc[-1:].values[0] + np.cumsum(np.random.randn(30) * 0.2),
        'lower_ci': series.iloc[-1:].values[0] - np.arange(30) * 0.5,
        'upper_ci': series.iloc[-1:].values[0] + np.arange(30) * 0.5
    }, index=pd.date_range(start=dates[-1], periods=30, freq='D')[1:])
    
    fig5 = ForecastDiagnostics.plot_forecast_uncertainty(forecast_df, series)
    
    print("Diagnostic visualizations created successfully!")

