import unittest
import pandas as pd
import numpy as np
from gca_analyzer import GCAVisualizer
import matplotlib.pyplot as plt

class TestGCAVisualizer(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures before each test method."""
        self.visualizer = GCAVisualizer()
        
        # Create test data for participation heatmap
        self.participation_data = pd.DataFrame({
            'person_id': ['Student1', 'Student2', 'Teacher'] * 3 + ['Student1'],
            'time': list(range(1, 11)),
            'participation': [1] * 10
        })
        
        # Create test data for interaction network
        self.interaction_data = pd.DataFrame({
            'source': ['Student1', 'Student2', 'Teacher', 'Student1'],
            'target': ['Teacher', 'Student1', 'Student2', 'Student2'],
            'weight': [0.8, 0.6, 0.7, 0.5]
        })
        
        # Create test data for metrics radar chart
        self.metrics_data = pd.DataFrame({
            'Internal_cohesion': [0.8, 0.7, 0.9],
            'Overall_responsivity': [0.6, 0.8, 0.7],
            'Social_impact': [0.7, 0.6, 0.8],
            'Newness': [0.5, 0.6, 0.4],
            'Communication_density': [0.7, 0.8, 0.6]
        }, index=['Student1', 'Student2', 'Teacher'])
        
        # Create test data for temporal metrics
        self.temporal_data = pd.DataFrame({
            'person_id': ['Student1', 'Student2', 'Teacher'] * 3,
            'time': list(range(1, 10)),
            'metric_value': np.random.rand(9)
        })
        
        # Empty DataFrame for error testing
        self.empty_df = pd.DataFrame()
        
    def test_plot_participation_heatmap(self):
        """Test participation heatmap plotting."""
        fig = self.visualizer.plot_participation_heatmap(self.participation_data)
        self.assertIsNotNone(fig)
        
    def test_plot_interaction_network(self):
        """Test interaction network plotting."""
        fig = self.visualizer.plot_interaction_network(self.interaction_data)
        self.assertIsNotNone(fig)
        
    def test_plot_metrics_radar(self):
        """Test metrics radar chart plotting."""
        metrics = ['Internal_cohesion', 'Overall_responsivity', 'Social_impact', 
                  'Newness', 'Communication_density']
        fig = self.visualizer.plot_metrics_radar(self.metrics_data, metrics)
        self.assertIsNotNone(fig)
        
    def test_plot_temporal_metrics(self):
        """Test temporal metrics plotting."""
        fig = self.visualizer.plot_temporal_metrics(self.temporal_data, 'metric_value')
        self.assertIsNotNone(fig)
        
    def test_error_handling(self):
        """Test error handling in visualization."""
        # Test empty data
        with self.assertRaises(ValueError):
            self.visualizer.plot_participation_heatmap(self.empty_df)
            
        with self.assertRaises(ValueError):
            self.visualizer.plot_interaction_network(self.empty_df)
            
        with self.assertRaises(ValueError):
            metrics = ['Internal_cohesion', 'Overall_responsivity']
            self.visualizer.plot_metrics_radar(self.empty_df, metrics)
            
        with self.assertRaises(ValueError):
            self.visualizer.plot_temporal_metrics(self.empty_df, 'metric_value')
            
        # Test missing required columns
        invalid_data = pd.DataFrame({'wrong_column': [1, 2, 3]})
        with self.assertRaises(ValueError):
            self.visualizer.plot_participation_heatmap(invalid_data)
            
        with self.assertRaises(ValueError):
            self.visualizer.plot_interaction_network(invalid_data)
            
        with self.assertRaises(ValueError):
            metrics = ['Internal_cohesion']
            self.visualizer.plot_metrics_radar(invalid_data, metrics)
            
        with self.assertRaises(ValueError):
            self.visualizer.plot_temporal_metrics(invalid_data, 'metric_value')

if __name__ == '__main__':
    unittest.main()
