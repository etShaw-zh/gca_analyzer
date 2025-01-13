"""
Test module for the GCA Analyzer main script

This module contains unit tests for the command-line interface functionality.
"""

import os
import pytest
from unittest.mock import patch, MagicMock
import pandas as pd
from gca_analyzer.__main__ import main
from gca_analyzer.llm_processor import LLMTextProcessor

@pytest.fixture
def sample_csv(tmp_path):
    """Create a temporary CSV file for testing."""
    data = pd.DataFrame({
        'conversation_id': ['conv1'] * 3,
        'person_id': ['p1', 'p2', 'p1'],
        'text': ['Hello', 'Hi there', 'How are you?'],
        'time': ['2025-01-13 10:00:00', '2025-01-13 10:01:00', '2025-01-13 10:02:00']
    })
    csv_path = tmp_path / "test_data.csv"
    data.to_csv(csv_path, index=False)
    return str(csv_path)

@pytest.fixture
def output_dir(tmp_path):
    """Create a temporary output directory."""
    output_path = tmp_path / "test_output"
    output_path.mkdir()
    return str(output_path)

@pytest.fixture
def mock_llm():
    """Mock LLM processor to avoid network calls."""
    with patch('gca_analyzer.analyzer.LLMTextProcessor') as mock:
        processor = mock.return_value
        processor.doc2vector.return_value = pd.DataFrame(
            [[0.1, 0.2], [0.2, 0.3], [0.3, 0.4]],
            columns=['dim1', 'dim2']
        )
        yield processor

def test_main_with_minimal_args(sample_csv, output_dir, mock_llm):
    """Test main function with minimal required arguments."""
    with patch('sys.argv', ['gca_analyzer', 
                          '--data', sample_csv,
                          '--output', output_dir]):
        main()
        # Verify LLM processor was called
        assert mock_llm.doc2vector.called
        # Check if output directory exists
        assert os.path.exists(output_dir)
        # Check if results file was created
        assert os.path.exists(os.path.join(output_dir, 'descriptive_statistics_gca.csv'))

def test_main_with_custom_window_config(sample_csv, output_dir, mock_llm):
    """Test main function with custom window configuration."""
    with patch('sys.argv', ['gca_analyzer',
                          '--data', sample_csv,
                          '--output', output_dir,
                          '--best-window-indices', '0.4',
                          '--min-window-size', '2',
                          '--max-window-size', '5']):
        main()
        assert os.path.exists(output_dir)
        assert os.path.exists(os.path.join(output_dir, 'descriptive_statistics_gca.csv'))

def test_main_with_custom_model_config(sample_csv, output_dir, mock_llm):
    """Test main function with custom model configuration."""
    with patch('sys.argv', ['gca_analyzer',
                          '--data', sample_csv,
                          '--output', output_dir,
                          '--model-name', 'bert-base-uncased',
                          '--model-mirror', 'https://huggingface.co/models']):
        main()
        assert os.path.exists(output_dir)
        assert os.path.exists(os.path.join(output_dir, 'descriptive_statistics_gca.csv'))

def test_main_with_custom_visualization_config(sample_csv, output_dir, mock_llm):
    """Test main function with custom visualization configuration."""
    with patch('sys.argv', ['gca_analyzer',
                          '--data', sample_csv,
                          '--output', output_dir,
                          '--default-figsize', '12', '10',
                          '--heatmap-figsize', '8', '6']):
        main()
        assert os.path.exists(output_dir)
        assert os.path.exists(os.path.join(output_dir, 'descriptive_statistics_gca.csv'))

def test_main_with_logging_config(sample_csv, output_dir, tmp_path, mock_llm):
    """Test main function with custom logging configuration."""
    log_file = str(tmp_path / "test.log")
    with patch('sys.argv', ['gca_analyzer',
                          '--data', sample_csv,
                          '--output', output_dir,
                          '--log-file', log_file,
                          '--console-level', 'DEBUG']):
        main()
        assert os.path.exists(log_file)
        assert os.path.exists(os.path.join(output_dir, 'descriptive_statistics_gca.csv'))

def test_main_with_invalid_data_path(output_dir, mock_llm):
    """Test main function with invalid data file path."""
    with pytest.raises(FileNotFoundError):
        with patch('sys.argv', ['gca_analyzer',
                              '--data', 'nonexistent.csv',
                              '--output', output_dir]):
            main()

def test_main_with_invalid_output_dir(sample_csv, tmp_path, mock_llm):
    """Test main function with invalid output directory."""
    invalid_dir = tmp_path / "nonexistent" / "directory"
    with pytest.raises(OSError):
        with patch('sys.argv', ['gca_analyzer',
                              '--data', sample_csv,
                              '--output', str(invalid_dir)]):
            main()

def test_main_with_invalid_window_size(sample_csv, output_dir, mock_llm):
    """Test main function with invalid window size configuration."""
    with pytest.raises(ValueError, match="min_num cannot be greater than max_num"):
        with patch('sys.argv', ['gca_analyzer',
                              '--data', sample_csv,
                              '--output', output_dir,
                              '--min-window-size', '10',
                              '--max-window-size', '5']):
            main()
