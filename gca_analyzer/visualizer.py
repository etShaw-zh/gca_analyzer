import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
import plotly.graph_objects as go
from typing import List, Optional, Dict, Any

class GCAVisualizer:
    """Class for visualizing GCA analysis results."""
    
    def __init__(self):
        """Initialize the visualizer with default settings."""
        self.default_colors = plt.cm.Set3(np.linspace(0, 1, 12))
        self.figsize = (10, 6)
        plt.style.use('default')  # Use default style instead of seaborn
        sns.set_theme(style="whitegrid")  # Set seaborn theme
        
    def plot_participation_heatmap(self, data: pd.DataFrame, title: str = None) -> go.Figure:
        """
        Plot participation heatmap.
        
        Args:
            data: DataFrame containing participation data
            title: Optional title for the plot
            
        Returns:
            plotly.graph_objects.Figure: Heatmap figure
        """
        if data.empty:
            raise ValueError("Input data is empty")
            
        if not all(col in data.columns for col in ['person_id', 'time']):
            raise ValueError("Data must contain 'person_id' and 'time' columns")
            
        # Create participation matrix
        pivot_data = pd.crosstab(data.person_id, data.time)
        
        # Create heatmap
        fig = go.Figure(data=go.Heatmap(
            z=pivot_data.values,
            x=pivot_data.columns,
            y=pivot_data.index,
            colorscale='Viridis'
        ))
        
        # Update layout
        fig.update_layout(
            title=title or 'Participation Heatmap',
            xaxis_title='Time',
            yaxis_title='Participant',
            height=600,
            width=800
        )
        
        return fig
        
    def plot_interaction_network(self, data: pd.DataFrame, title: str = None) -> go.Figure:
        """
        Plot interaction network.
        
        Args:
            data: DataFrame containing interaction data
            title: Optional title for the plot
            
        Returns:
            plotly.graph_objects.Figure: Network figure
        """
        if data.empty:
            raise ValueError("Input data is empty")
            
        if not all(col in data.columns for col in ['source', 'target', 'weight']):
            raise ValueError("Data must contain 'source', 'target', and 'weight' columns")
            
        # Create network
        G = nx.from_pandas_edgelist(data, 'source', 'target', 'weight')
        pos = nx.spring_layout(G)
        
        # Create edges
        edge_trace = go.Scatter(
            x=[], y=[],
            line=dict(width=0.5, color='#888'),
            hoverinfo='none',
            mode='lines')
            
        for edge in G.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_trace['x'] += (x0, x1, None)
            edge_trace['y'] += (y0, y1, None)
            
        # Create nodes
        node_trace = go.Scatter(
            x=[], y=[],
            mode='markers',
            hoverinfo='text',
            marker=dict(
                showscale=True,
                colorscale='YlGnBu',
                size=10
            ))
            
        for node in G.nodes():
            x, y = pos[node]
            node_trace['x'] += (x,)
            node_trace['y'] += (y,)
            
        # Create figure
        fig = go.Figure(data=[edge_trace, node_trace],
                       layout=go.Layout(
                           title=title or 'Interaction Network',
                           showlegend=False,
                           hovermode='closest',
                           margin=dict(b=20,l=5,r=5,t=40),
                           height=600,
                           width=800
                       ))
                       
        return fig
        
    def plot_metrics_radar(self, data: pd.DataFrame, metrics: List[str], title: str = None) -> go.Figure:
        """
        Plot metrics radar chart.
        
        Args:
            data: DataFrame containing metrics data
            metrics: List of metrics to plot
            title: Optional title for the plot
            
        Returns:
            plotly.graph_objects.Figure: Radar chart figure
        """
        if data.empty:
            raise ValueError("Input data is empty")
            
        if not all(metric in data.columns for metric in metrics):
            raise ValueError(f"Data must contain all specified metrics: {metrics}")
            
        # Create radar chart
        fig = go.Figure()
        
        for idx, row in data.iterrows():
            fig.add_trace(go.Scatterpolar(
                r=[row[m] for m in metrics],
                theta=metrics,
                fill='toself',
                name=str(idx)
            ))
            
        # Update layout
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1]
                )),
            showlegend=True,
            title=title or 'Metrics Radar Chart',
            height=600,
            width=800
        )
        
        return fig
        
    def plot_temporal_metrics(self, data: pd.DataFrame, metric: str, title: str = None) -> go.Figure:
        """
        Plot temporal metrics.
        
        Args:
            data: DataFrame containing temporal data
            metric: Name of the metric to plot
            title: Optional title for the plot
            
        Returns:
            plotly.graph_objects.Figure: Line plot figure
        """
        if data.empty:
            raise ValueError("Input data is empty")
            
        if metric not in data.columns:
            raise ValueError(f"Data must contain the specified metric: {metric}")
            
        if 'time' not in data.columns:
            raise ValueError("Data must contain a 'time' column")
            
        # Create line plot
        fig = go.Figure()
        
        for name, group in data.groupby('person_id'):
            fig.add_trace(go.Scatter(
                x=group['time'],
                y=group[metric],
                mode='lines+markers',
                name=str(name)
            ))
            
        # Update layout
        fig.update_layout(
            title=title or f'Temporal {metric}',
            xaxis_title='Time',
            yaxis_title=metric,
            height=600,
            width=800
        )
        
        return fig
