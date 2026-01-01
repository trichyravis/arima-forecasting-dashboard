"""
Charts & Visualization Module - src/visualization/charts.py

Comprehensive charting and plotting for ARIMA forecasting dashboard.
Includes time series plots, forecasts, ACF/PACF, and more.

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
from typing import Optional, List, Tuple
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TimeSeriesCharts:
    """
    Time series visualization charts.
    
    Supports:
    - Price series plots
    - Forecast with confidence intervals
    - Multiple series comparison
    """
    
    @staticmethod
    def plot_series(
        series: pd.Series,
        title: str = "Time Series",
        ylabel: str = "Value",
        show_ma: bool = False,
        ma_window: int = 20
    ) -> go.Figure:
        """
        Plot time series.
        
        Args:
            series: Time series data
            title: Plot title
            ylabel: Y-axis label
            show_ma: Show moving average
            ma_window: Moving average window
        
        Returns:
            Plotly figure
        """
        fig = go.Figure()
        
        # Add main series
        fig.add_trace(go.Scatter(
            x=series.index,
            y=series.values,
            mode='lines',
            name='Price',
            line=dict(color='#003366', width=2)
        ))
        
        # Add moving average if requested
        if show_ma:
            ma = series.rolling(window=ma_window).mean()
            fig.add_trace(go.Scatter(
                x=ma.index,
                y=ma.values,
                mode='lines',
                name=f'MA({ma_window})',
                line=dict(color='#FFD700', width=2, dash='dash')
            ))
        
        fig.update_layout(
            title=title,
            xaxis_title='Date',
            yaxis_title=ylabel,
            template='plotly_white',
            hovermode='x unified',
            height=500,
            font=dict(family="Arial", size=12, color='#003366'),
            plot_bgcolor='rgba(240, 240, 240, 0.5)'
        )
        
        logger.info(f"Created time series plot: {title}")
        
        return fig
    
    @staticmethod
    def plot_forecast(
        actual: pd.Series,
        forecast: pd.DataFrame,
        title: str = "ARIMA Forecast",
        lookback: int = 50
    ) -> go.Figure:
        """
        Plot actual vs forecast with confidence intervals.
        
        Args:
            actual: Actual values series
            forecast: Forecast DataFrame with columns ['forecast', 'lower_ci', 'upper_ci']
            title: Plot title
            lookback: Number of recent actual values to show
        
        Returns:
            Plotly figure
        """
        fig = go.Figure()
        
        # Get recent actual values
        recent_actual = actual.tail(lookback)
        
        # Add actual values
        fig.add_trace(go.Scatter(
            x=recent_actual.index,
            y=recent_actual.values,
            mode='lines',
            name='Actual',
            line=dict(color='#003366', width=2.5)
        ))
        
        # Add forecast
        fig.add_trace(go.Scatter(
            x=forecast.index,
            y=forecast['forecast'],
            mode='lines',
            name='Forecast',
            line=dict(color='#FF6B6B', width=2.5)
        ))
        
        # Add confidence interval
        fig.add_trace(go.Scatter(
            x=forecast.index.tolist() + forecast.index.tolist()[::-1],
            y=forecast['upper_ci'].tolist() + forecast['lower_ci'].tolist()[::-1],
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
            hovermode='x unified',
            height=600,
            font=dict(family="Arial", size=12, color='#003366'),
            plot_bgcolor='rgba(240, 240, 240, 0.5)'
        )
        
        logger.info(f"Created forecast plot: {title}")
        
        return fig
    
    @staticmethod
    def plot_multiple_series(
        series_dict: dict,
        title: str = "Multiple Series Comparison",
        ylabel: str = "Value"
    ) -> go.Figure:
        """
        Plot multiple series for comparison.
        
        Args:
            series_dict: Dictionary of {name: series}
            title: Plot title
            ylabel: Y-axis label
        
        Returns:
            Plotly figure
        """
        fig = go.Figure()
        
        colors = ['#003366', '#004d80', '#FF6B6B', '#FFD700', '#4CAF50']
        
        for i, (name, series) in enumerate(series_dict.items()):
            color = colors[i % len(colors)]
            fig.add_trace(go.Scatter(
                x=series.index,
                y=series.values,
                mode='lines',
                name=name,
                line=dict(color=color, width=2)
            ))
        
        fig.update_layout(
            title=title,
            xaxis_title='Date',
            yaxis_title=ylabel,
            template='plotly_white',
            hovermode='x unified',
            height=600,
            font=dict(family="Arial", size=12, color='#003366')
        )
        
        logger.info(f"Created comparison plot: {title}")
        
        return fig


class DiagnosticCharts:
    """
    Diagnostic charts for model validation.
    
    Includes:
    - Residual plots
    - ACF/PACF plots
    - Distribution analysis
    """
    
    @staticmethod
    def plot_residuals(
        residuals: pd.Series,
        title: str = "Residual Analysis"
    ) -> go.Figure:
        """
        Plot residuals.
        
        Args:
            residuals: Model residuals
            title: Plot title
        
        Returns:
            Plotly figure
        """
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Residuals Over Time', 'Residual Distribution',
                          'ACF', 'Q-Q Plot'),
            specs=[[{}, {}], [{}, {}]]
        )
        
        # Residuals over time
        fig.add_trace(
            go.Scatter(x=residuals.index, y=residuals.values,
                      mode='markers', name='Residuals',
                      marker=dict(color='#003366', size=5)),
            row=1, col=1
        )
        fig.add_hline(y=0, line_dash="dash", line_color="red", row=1, col=1)
        
        # Histogram
        fig.add_trace(
            go.Histogram(x=residuals.values, nbinsx=30, name='Distribution',
                        marker=dict(color='#FFD700')),
            row=1, col=2
        )
        
        # Normal distribution overlay
        mu, sigma = residuals.mean(), residuals.std()
        x_range = np.linspace(residuals.min(), residuals.max(), 100)
        normal_dist = (1/(sigma*np.sqrt(2*np.pi))*
                      np.exp(-0.5*((x_range-mu)/sigma)**2))
        normal_dist *= len(residuals) * (residuals.max()-residuals.min())/30
        
        fig.add_trace(
            go.Scatter(x=x_range, y=normal_dist, mode='lines',
                      name='Normal', line=dict(color='red')),
            row=1, col=2
        )
        
        # Q-Q plot
        sorted_residuals = np.sort(residuals)
        n = len(sorted_residuals)
        theoretical_quantiles = np.random.normal(0, 1, n)
        theoretical_quantiles = np.sort(theoretical_quantiles)
        
        fig.add_trace(
            go.Scatter(x=theoretical_quantiles, y=sorted_residuals,
                      mode='markers', name='Q-Q',
                      marker=dict(color='#003366', size=5)),
            row=2, col=2
        )
        
        # Add diagonal line
        min_val = min(theoretical_quantiles.min(), sorted_residuals.min())
        max_val = max(theoretical_quantiles.max(), sorted_residuals.max())
        fig.add_trace(
            go.Scatter(x=[min_val, max_val], y=[min_val, max_val],
                      mode='lines', name='Perfect Fit',
                      line=dict(color='red', dash='dash')),
            row=2, col=2
        )
        
        fig.update_xaxes(title_text="Date", row=1, col=1)
        fig.update_yaxes(title_text="Residual", row=1, col=1)
        fig.update_xaxes(title_text="Value", row=1, col=2)
        fig.update_yaxes(title_text="Frequency", row=1, col=2)
        fig.update_xaxes(title_text="Theoretical", row=2, col=2)
        fig.update_yaxes(title_text="Sample", row=2, col=2)
        
        fig.update_layout(
            title_text=title,
            height=800,
            showlegend=True,
            template='plotly_white'
        )
        
        logger.info(f"Created residual plot: {title}")
        
        return fig
    
    @staticmethod
    def plot_comparison(
        actual: pd.Series,
        predicted: pd.Series,
        title: str = "Actual vs Predicted"
    ) -> go.Figure:
        """
        Plot actual vs predicted values.
        
        Args:
            actual: Actual values
            predicted: Predicted values
            title: Plot title
        
        Returns:
            Plotly figure
        """
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=('Series Comparison', 'Forecast Error'),
            vertical_spacing=0.12
        )
        
        # Series comparison
        fig.add_trace(
            go.Scatter(x=actual.index, y=actual.values,
                      mode='lines', name='Actual',
                      line=dict(color='#003366', width=2)),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(x=predicted.index, y=predicted.values,
                      mode='lines', name='Predicted',
                      line=dict(color='#FF6B6B', width=2)),
            row=1, col=1
        )
        
        # Error
        error = actual - predicted
        fig.add_trace(
            go.Bar(x=error.index, y=error.values, name='Error',
                  marker=dict(color='#FFD700')),
            row=2, col=1
        )
        fig.add_hline(y=0, line_dash="dash", line_color="red", row=2, col=1)
        
        fig.update_yaxes(title_text="Value", row=1, col=1)
        fig.update_yaxes(title_text="Error", row=2, col=1)
        fig.update_xaxes(title_text="Date", row=2, col=1)
        
        fig.update_layout(
            title_text=title,
            height=700,
            hovermode='x unified',
            template='plotly_white'
        )
        
        logger.info(f"Created comparison plot: {title}")
        
        return fig


class MetricCharts:
    """
    Charts for displaying metrics and statistics.
    
    Includes:
    - Metric comparison tables
    - Information criteria comparison
    - Performance metrics
    """
    
    @staticmethod
    def plot_metrics_table(
        metrics: dict,
        title: str = "Model Metrics"
    ) -> go.Figure:
        """
        Create metrics table.
        
        Args:
            metrics: Dictionary of metrics
            title: Table title
        
        Returns:
            Plotly figure
        """
        keys = list(metrics.keys())
        values = [f"{v:.4f}" if isinstance(v, float) else str(v)
                 for v in metrics.values()]
        
        fig = go.Figure(data=[go.Table(
            header=dict(
                values=['<b>Metric</b>', '<b>Value</b>'],
                fill_color='#003366',
                align='left',
                font=dict(color='white', size=12)
            ),
            cells=dict(
                values=[keys, values],
                fill_color='rgba(240, 240, 240, 0.5)',
                align='left',
                font=dict(color='#003366', size=11),
                height=30
            )
        )])
        
        fig.update_layout(
            title=title,
            height=400,
            margin=dict(l=20, r=20, t=40, b=20)
        )
        
        logger.info(f"Created metrics table: {title}")
        
        return fig
    
    @staticmethod
    def plot_model_comparison(
        comparison_df: pd.DataFrame,
        metrics_cols: List[str] = ['AIC', 'BIC'],
        title: str = "Model Comparison"
    ) -> go.Figure:
        """
        Compare multiple models.
        
        Args:
            comparison_df: DataFrame with model metrics
            metrics_cols: Columns to compare
            title: Plot title
        
        Returns:
            Plotly figure
        """
        fig = go.Figure()
        
        for col in metrics_cols:
            if col in comparison_df.columns:
                fig.add_trace(go.Bar(
                    x=comparison_df.index,
                    y=comparison_df[col],
                    name=col
                ))
        
        fig.update_layout(
            title=title,
            xaxis_title='Model',
            yaxis_title='Value',
            barmode='group',
            template='plotly_white',
            height=500,
            font=dict(family="Arial", size=12, color='#003366')
        )
        
        logger.info(f"Created comparison chart: {title}")
        
        return fig
    
    @staticmethod
    def plot_accuracy_metrics(
        metrics: dict,
        title: str = "Forecast Accuracy"
    ) -> go.Figure:
        """
        Display accuracy metrics as gauge charts.
        
        Args:
            metrics: Dictionary of metrics
            title: Plot title
        
        Returns:
            Plotly figure
        """
        # Extract accuracy metric if available
        accuracy = metrics.get('directional_accuracy', 0)
        
        fig = go.Figure(data=[
            go.Indicator(
                mode='gauge+number+delta',
                value=accuracy,
                title={'text': 'Directional Accuracy (%)'},
                delta={'reference': 50},
                gauge={
                    'axis': {'range': [0, 100]},
                    'bar': {'color': '#003366'},
                    'steps': [
                        {'range': [0, 50], 'color': '#FFE0E0'},
                        {'range': [50, 75], 'color': '#FFFFE0'},
                        {'range': [75, 100], 'color': '#E0FFE0'}
                    ],
                    'threshold': {
                        'line': {'color': 'red', 'width': 4},
                        'thickness': 0.75,
                        'value': 50
                    }
                }
            )
        ])
        
        fig.update_layout(height=400, title=title)
        
        logger.info(f"Created accuracy gauge: {title}")
        
        return fig


# Example usage
if __name__ == "__main__":
    import pandas as pd
    import numpy as np
    
    # Generate sample data
    dates = pd.date_range('2023-01-01', periods=200)
    prices = 100 + np.cumsum(np.random.randn(200) * 0.5)
    series = pd.Series(prices, index=dates)
    
    # Generate forecast
    forecast_dates = pd.date_range(start=dates[-1], periods=31, freq='D')[1:]
    forecast_values = prices[-1] + np.cumsum(np.random.randn(30) * 0.3)
    forecast_df = pd.DataFrame({
        'forecast': forecast_values,
        'lower_ci': forecast_values - 5,
        'upper_ci': forecast_values + 5
    }, index=forecast_dates)
    
    print("Creating sample charts...")
    
    # Create charts
    fig1 = TimeSeriesCharts.plot_series(series, show_ma=True)
    fig2 = TimeSeriesCharts.plot_forecast(series, forecast_df)
    fig3 = DiagnosticCharts.plot_residuals(series.diff().dropna())
    
    print("Charts created successfully!")

