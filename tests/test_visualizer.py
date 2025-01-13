"""
Test module for the GCA Visualizer

This module contains unit tests for the GCAVisualizer class.
"""

import os
import pytest
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from gca_analyzer.visualizer import GCAVisualizer
from gca_analyzer.config import Config

@pytest.fixture
def sample_metrics_df():
    """Create sample metrics DataFrame for testing."""
    return pd.DataFrame({
        'cohesion': [0.5, 0.6, 0.7],
        'response': [0.4, 0.5, 0.6],
        'participation': [0.3, 0.4, 0.5],
        'time': pd.date_range(start='2025-01-01', periods=3)
    })

@pytest.fixture
def visualizer():
    """Create a GCAVisualizer instance for testing."""
    config = Config()
    return GCAVisualizer(config=config)

def test_init():
    """Test visualizer initialization."""
    config = Config()
    viz = GCAVisualizer(config=config)
    assert viz._config == config
    assert viz.figsize == config.visualization.default_figsize
    
def test_validate_metrics_data(visualizer, sample_metrics_df):
    """Test metrics data validation."""
    # Should not raise any exceptions
    metrics = visualizer._validate_metrics_data(sample_metrics_df)
    assert 'cohesion' in metrics
    assert 'response' in metrics
    assert 'participation' in metrics
    
    # Test with specific metrics
    specific_metrics = ['cohesion', 'response']
    validated = visualizer._validate_metrics_data(sample_metrics_df, specific_metrics)
    assert validated == specific_metrics
    
    # Test with empty DataFrame
    empty_df = pd.DataFrame()
    with pytest.raises(ValueError, match="Input data is empty for plotting"):
        visualizer._validate_metrics_data(empty_df)
    
    # Test with missing metrics
    with pytest.raises(ValueError, match="Data must contain all specified metrics"):
        visualizer._validate_metrics_data(sample_metrics_df, ['nonexistent_metric'])

def test_plot_metrics_radar(visualizer, sample_metrics_df):
    """Test plotting metrics radar chart."""
    metrics = ['cohesion', 'response', 'participation']
    fig = visualizer.plot_metrics_radar(sample_metrics_df, metrics)
    assert isinstance(fig, go.Figure)
    assert len(fig.data) == len(sample_metrics_df)  # One trace per row
    
    # Test with custom title
    title = "Custom Radar Chart"
    fig = visualizer.plot_metrics_radar(sample_metrics_df, metrics, title=title)
    assert fig.layout.title.text == title

def test_plot_metrics_distribution(visualizer, sample_metrics_df):
    """Test plotting metrics distribution."""
    # Test with default metrics
    fig = visualizer.plot_metrics_distribution(sample_metrics_df)
    assert isinstance(fig, go.Figure)
    assert len(fig.data) == 3  # One trace per metric
    
    # Test with specific metrics
    specific_metrics = ['cohesion', 'response']
    fig = visualizer.plot_metrics_distribution(sample_metrics_df, specific_metrics)
    assert isinstance(fig, go.Figure)
    assert len(fig.data) == 2
    
    # Test with custom title
    title = "Custom Distribution"
    fig = visualizer.plot_metrics_distribution(sample_metrics_df, title=title)
    assert fig.layout.title.text == title

def test_error_handling(visualizer, sample_metrics_df):
    """Test error handling in visualization methods."""
    # Test with invalid metrics
    with pytest.raises(ValueError):
        visualizer.plot_metrics_radar(sample_metrics_df, ['invalid_metric'])
    
    # Test with empty DataFrame
    empty_df = pd.DataFrame()
    with pytest.raises(ValueError):
        visualizer.plot_metrics_distribution(empty_df)
    
    # Test with non-numeric data
    invalid_df = pd.DataFrame({'text': ['a', 'b', 'c']})
    metrics = visualizer._validate_metrics_data(invalid_df)
    assert len(metrics) == 0  # Should return empty list for non-numeric columns
