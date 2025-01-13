"""
GCA Visualizer Module

This module provides visualization functionality for group conversation analysis,
including heatmaps, network graphs, and various metrics visualizations.

Author: Jianjun Xiao
Email: et_shaw@126.com
Date: 2025-01-12
License: Apache 2.0
"""

from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go

from .config import Config, default_config
from .logger import logger


class GCAVisualizer:
    """Class for visualizing GCA analysis results.

    This class provides various visualization methods for analyzing group
    conversations, including participation patterns, interaction networks,
    and metrics distributions.

    Attributes:
        _config (Config): Configuration instance for visualization settings
        default_colors (np.ndarray): Array of default colors for plotting
        figsize (tuple): Default figure size for plots
    """

    def __init__(self, config: Config = None):
        """Initialize the visualizer with default settings.

        Args:
            config (Config, optional): Configuration instance. Defaults to None.
        """
        logger.info("Initializing GCA Visualizer")
        self._config = config or default_config
        self.default_colors = plt.cm.Set3(np.linspace(0, 1, 12))
        self.figsize = self._config.visualization.default_figsize
        plt.style.use('default')
        logger.debug("Style configuration set")
        
    def _validate_metrics_data(
        self,
        data: pd.DataFrame,
        metrics: Optional[List[str]] = None,
        operation: str = "plotting"
    ) -> List[str]:
        """Validate input data and metrics for plotting functions.

        Args:
            data: DataFrame containing metrics data
            metrics: Optional list of metrics to include
            operation: Description of the operation being performed (for error messages)

        Returns:
            List of validated metrics

        Raises:
            ValueError: If input data is empty or missing required metrics
        """
        if data.empty:
            raise ValueError(f"Input data is empty for {operation}")
        
        if metrics is None:
            return data.select_dtypes(include=[np.number]).columns.tolist()
        
        if not all(metric in data.columns for metric in metrics):
            raise ValueError(f"Data must contain all specified metrics for {operation}: {metrics}")
        
        return metrics

    def plot_metrics_radar(
        self,
        data: pd.DataFrame,
        metrics: List[str],
        title: str = None
    ) -> go.Figure:
        """Create a radar chart visualization of multiple metrics."""
        logger.info("Creating metrics radar chart")
        try:
            metrics = self._validate_metrics_data(data, metrics, "radar chart")
            
            fig = go.Figure()
            
            for idx, row in data.iterrows():
                fig.add_trace(go.Scatterpolar(
                    r=[row[m] for m in metrics],
                    theta=metrics,
                    fill='toself',
                    name=str(idx)
                ))
            
            fig.update_layout(
                polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
                showlegend=True,
                title=title or 'Metrics Radar Chart',
                height=600,
                width=800
            )
            
            logger.debug("Metrics radar chart created")
            return fig
        except Exception as e:
            logger.error(f"Error creating metrics radar chart: {str(e)}")
            raise

    def plot_metrics_distribution(
        self,
        data: pd.DataFrame,
        metrics: List[str] = None,
        title: str = None
    ) -> go.Figure:
        """Create a violin plot of metrics distributions."""
        logger.info("Creating metrics distribution plot")
        try:
            metrics = self._validate_metrics_data(data, metrics, "distribution plot")
            
            fig = go.Figure()
            
            for metric in metrics:
                fig.add_trace(go.Violin(
                    x=[metric] * len(data),
                    y=data[metric],
                    name=metric,
                    box_visible=True,
                    meanline_visible=True,
                    points="all",
                    jitter=0.05,
                    pointpos=-0.1,
                    marker=dict(size=4),
                    line_color='rgb(70,130,180)',
                    fillcolor='rgba(70,130,180,0.3)',
                    opacity=0.6,
                    side='positive',
                    width=1.8,
                    meanline=dict(color="black", width=2),
                    box=dict(line=dict(color="black", width=2))
                ))
            
            fig.update_layout(
                title=title or 'Metrics Distribution',
                showlegend=False,
                xaxis_title='Metrics',
                yaxis_title='Value',
                height=600,
                width=1000,
                template='plotly_white',
                margin=dict(l=50, r=50, t=50, b=50),
                xaxis=dict(showgrid=True, gridcolor='rgba(0,0,0,0.1)', tickangle=45),
                yaxis=dict(
                    showgrid=True,
                    gridcolor='rgba(0,0,0,0.1)',
                    zeroline=True,
                    zerolinecolor='rgba(0,0,0,0.2)',
                    zerolinewidth=1
                ),
                plot_bgcolor='rgba(0,0,0,0)'
            )
            
            logger.debug("Metrics distribution plot created")
            return fig
        except Exception as e:
            logger.error(f"Error creating metrics distribution plot: {str(e)}")
            raise
