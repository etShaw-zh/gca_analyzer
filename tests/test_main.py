"""
Test module for the GCA Analyzer main script

This module contains unit tests for the command-line interface functionality.
"""

import os
import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, Mock
import tempfile
from gca_analyzer.__main__ import main
from gca_analyzer.llm_processor import LLMTextProcessor

@pytest.fixture
def sample_data():
    """Create sample conversation data for testing."""
    return pd.DataFrame({
        'conversation_id': ['test_conv'] * 6,
        'person_id': ['A', 'B', 'A', 'B', 'A', 'B'],
        'text': [
            'Hello there!',
            'Hi, how are you?',
            'I am doing well.',
            'That is good to hear.',
            'What are your plans?',
            'Just working today.'
        ],
        'time': pd.date_range(start='2024-01-01', periods=6, freq='h'),
        'seq': range(6)
    })

@pytest.fixture
def mock_llm():
    """Mock LLM processor for testing."""
    with patch('gca_analyzer.analyzer.LLMTextProcessor') as mock:
        processor = Mock(spec=LLMTextProcessor)
        # Create a numpy array of shape (6, 3) with constant values
        embeddings = np.array([[0.1, 0.2, 0.3]] * 6, dtype=np.float32)
        # Return list of numpy arrays
        processor.doc2vector.return_value = embeddings  # Return numpy array directly
        mock.return_value = processor
        yield processor

def test_main_with_minimal_args(tmp_path, sample_data, mock_llm):
    """Test main function with minimal arguments."""
    data_file = tmp_path / "test_data.csv"
    sample_data.to_csv(data_file, index=False)

    output_dir = tmp_path / "output"
    output_dir.mkdir()

    with patch('sys.argv', [
        'gca_analyzer',
        '--data', str(data_file),
        '--output', str(output_dir)
    ]):
        main()
        # Check output directory exists
        assert output_dir.exists()
        # Check required output files exist
        assert (output_dir / "metrics_test_conv.csv").exists()
        assert (output_dir / "descriptive_statistics_test_conv.csv").exists()

def test_main_with_custom_window_config(tmp_path, sample_data, mock_llm):
    """Test main function with custom window configuration."""
    data_file = tmp_path / "test_data.csv"
    sample_data.to_csv(data_file, index=False)

    output_dir = tmp_path / "output"
    output_dir.mkdir()

    with patch('sys.argv', [
        'gca_analyzer',
        '--data', str(data_file),
        '--output', str(output_dir),
        '--min-window-size', '2',
        '--max-window-size', '4',
        '--best-window-indices', '0.5'
    ]):
        main()
        # Check output directory exists
        assert output_dir.exists()
        # Check required output files exist
        assert (output_dir / "metrics_test_conv.csv").exists()
        assert (output_dir / "descriptive_statistics_test_conv.csv").exists()

def test_main_with_custom_model_config(tmp_path, sample_data, mock_llm):
    """Test main function with custom model configuration."""
    data_file = tmp_path / "test_data.csv"
    sample_data.to_csv(data_file, index=False)

    output_dir = tmp_path / "output"
    output_dir.mkdir()

    with patch('sys.argv', [
        'gca_analyzer',
        '--data', str(data_file),
        '--output', str(output_dir),
        '--model-name', 'sentence-transformers/all-MiniLM-L6-v2',
        '--model-mirror', 'https://huggingface.co/sentence-transformers'
    ]):
        main()
        assert output_dir.exists()
        assert (output_dir / "metrics_test_conv.csv").exists()

def test_main_with_custom_visualization_config(tmp_path, sample_data, mock_llm):
    """Test main function with custom visualization configuration."""
    data_file = tmp_path / "test_data.csv"
    sample_data.to_csv(data_file, index=False)

    output_dir = tmp_path / "output"
    output_dir.mkdir()

    with patch('sys.argv', [
        'gca_analyzer',
        '--data', str(data_file),
        '--output', str(output_dir),
        '--default-figsize', '12', '8',
        '--heatmap-figsize', '10', '6'
    ]):
        main()
        # Check output directory exists
        assert output_dir.exists()
        # Check required output files exist
        assert (output_dir / "metrics_test_conv.csv").exists()
        assert (output_dir / "descriptive_statistics_test_conv.csv").exists()

def test_main_with_logging_config(tmp_path, sample_data, mock_llm):
    """Test main function with custom logging configuration."""
    data_file = tmp_path / "test_data.csv"
    sample_data.to_csv(data_file, index=False)

    output_dir = tmp_path / "output"
    output_dir.mkdir()
    log_file = tmp_path / "test.log"

    with patch('sys.argv', [
        'gca_analyzer',
        '--data', str(data_file),
        '--output', str(output_dir),
        '--log-file', str(log_file),
        '--console-level', 'DEBUG',
        '--file-level', 'INFO'
    ]):
        main()
        # Check output directory exists
        assert output_dir.exists()
        # Check required output files exist
        assert (output_dir / "metrics_test_conv.csv").exists()
        assert (output_dir / "descriptive_statistics_test_conv.csv").exists()
        # Check log file exists
        assert log_file.exists()

def test_main_with_invalid_data_path(tmp_path, mock_llm):
    """Test main function with invalid data file path."""
    with pytest.raises(FileNotFoundError):
        with patch('sys.argv', [
            'gca_analyzer',
            '--data', str(tmp_path / "nonexistent.csv"),
            '--output', str(tmp_path)
        ]):
            main()

def test_main_with_invalid_output_dir(tmp_path, sample_data, mock_llm):
    """Test main function with invalid output directory."""
    data_file = tmp_path / "test_data.csv"
    sample_data.to_csv(data_file, index=False)
    invalid_dir = tmp_path / "nonexistent" / "directory"
    
    with pytest.raises(OSError):
        with patch('sys.argv', [
            'gca_analyzer',
            '--data', str(data_file),
            '--output', str(invalid_dir / "output")
        ]):
            main()

def test_main_with_invalid_window_size(tmp_path, sample_data, mock_llm):
    """Test main function with invalid window size configuration."""
    data_file = tmp_path / "test_data.csv"
    sample_data.to_csv(data_file, index=False)
    
    with pytest.raises(ValueError):
        with patch('sys.argv', [
            'gca_analyzer',
            '--data', str(data_file),
            '--output', str(tmp_path),
            '--min-window-size', '5',
            '--max-window-size', '3'
        ]):
            main()

def test_main_with_all_options(tmp_path, sample_data, mock_llm):
    """Test main function with all available options."""
    data_file = tmp_path / "test_data.csv"
    sample_data.to_csv(data_file, index=False)
    
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    log_file = tmp_path / "test.log"
    
    with patch('sys.argv', [
        'gca_analyzer',
        '--data', str(data_file),
        '--output', str(output_dir),
        '--min-window-size', '2',
        '--max-window-size', '4',
        '--best-window-indices', '0.5',
        '--model-name', 'sentence-transformers/all-MiniLM-L6-v2',
        '--model-mirror', 'https://huggingface.co/sentence-transformers',
        '--default-figsize', '12', '8',
        '--heatmap-figsize', '10', '6',
        '--log-file', str(log_file),
        '--console-level', 'DEBUG',
        '--file-level', 'INFO'
    ]):
        main()
        assert output_dir.exists()
        assert (output_dir / "metrics_test_conv.csv").exists()
        assert log_file.exists()

def test_main_with_invalid_log_level(tmp_path, sample_data, mock_llm):
    """Test main function with invalid log level."""
    data_file = tmp_path / "test_data.csv"
    sample_data.to_csv(data_file, index=False)
    
    with pytest.raises(SystemExit):
        with patch('sys.argv', [
            'gca_analyzer',
            '--data', str(data_file),
            '--output', str(tmp_path),
            '--console-level', 'INVALID'
        ]):
            main()

def test_main_with_invalid_window_threshold(tmp_path, sample_data, mock_llm):
    """Test main function with invalid window threshold."""
    data_file = tmp_path / "test_data.csv"
    sample_data.to_csv(data_file, index=False)
    
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    
    with patch('sys.argv', [
        'gca_analyzer',
        '--data', str(data_file),
        '--output', str(output_dir),
        '--best-window-indices', '2.0'  # Invalid threshold > 1
    ]), pytest.raises(ValueError):
        main()

def test_main_with_invalid_figsize(tmp_path, sample_data, mock_llm):
    """Test main function with invalid figsize."""
    data_file = tmp_path / "test_data.csv"
    sample_data.to_csv(data_file, index=False)
    
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    
    with patch('sys.argv', [
        'gca_analyzer',
        '--data', str(data_file),
        '--output', str(output_dir),
        '--default-figsize', '-1', '8'  # Invalid negative size
    ]), pytest.raises(ValueError):
        main()

def test_main_with_custom_umap_config(tmp_path, sample_data, mock_llm):
    """Test main function with custom UMAP configuration."""
    data_file = tmp_path / "test_data.csv"
    sample_data.to_csv(data_file, index=False)

    output_dir = tmp_path / "output"
    output_dir.mkdir()

    with patch('sys.argv', [
        'gca_analyzer',
        '--data', str(data_file),
        '--output', str(output_dir),
        '--umap-n-neighbors', '10',
        '--umap-min-dist', '0.2',
        '--umap-n-components', '3',
        '--umap-random-state', '100'
    ]):
        main()
