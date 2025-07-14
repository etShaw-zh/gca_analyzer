"""
Test module for the GCA Analyzer main script

This module contains unit tests for the command-line interface functionality.
"""

import os
import pytest
from unittest.mock import patch, MagicMock
import pandas as pd
from gca_analyzer.llm_processor import LLMTextProcessor
from gca_analyzer.main import (
    main_cli,
    show_welcome,
    show_error,
    show_success,
    show_info,
    Prompt,
    Confirm
)

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
        main_cli()
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
        main_cli()
        assert os.path.exists(output_dir)
        assert os.path.exists(os.path.join(output_dir, 'descriptive_statistics_gca.csv'))

def test_main_with_custom_model_config(sample_csv, output_dir, mock_llm):
    """Test main function with custom model configuration."""
    with patch('sys.argv', ['gca_analyzer',
                          '--data', sample_csv,
                          '--output', output_dir,
                          '--model-name', 'bert-base-uncased',
                          '--model-mirror', 'https://huggingface.co/models']):
        main_cli()
        assert os.path.exists(output_dir)
        assert os.path.exists(os.path.join(output_dir, 'descriptive_statistics_gca.csv'))

def test_main_with_custom_visualization_config(sample_csv, output_dir, mock_llm):
    """Test main function with custom visualization configuration."""
    with patch('sys.argv', ['gca_analyzer',
                          '--data', sample_csv,
                          '--output', output_dir,
                          '--default-figsize', '12', '10',
                          '--heatmap-figsize', '8', '6']):
        main_cli()
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
        main_cli()
        assert os.path.exists(log_file)
        assert os.path.exists(os.path.join(output_dir, 'descriptive_statistics_gca.csv'))

def test_main_with_invalid_data_path(output_dir, mock_llm, capsys):
    """Test main function with invalid data file path."""
    with patch('sys.argv', ['gca_analyzer',
                          '--data', 'nonexistent.csv',
                          '--output', output_dir]):
        main_cli()
    captured = capsys.readouterr()
    assert "Input file not found: nonexistent.csv" in captured.out

def test_main_with_invalid_output_dir(sample_csv, tmp_path, mock_llm, capsys):
    """Test main function with invalid output directory."""
    invalid_dir = tmp_path / "nonexistent" / "directory"
    with patch('sys.argv', ['gca_analyzer',
                          '--data', sample_csv,
                          '--output', str(invalid_dir)]):
        main_cli()
    captured = capsys.readouterr()
    assert "Parent directory does not exist" in captured.out

def test_main_with_invalid_window_size(sample_csv, output_dir, mock_llm, capsys):
    """Test main function with invalid window size configuration."""
    with patch('sys.argv', ['gca_analyzer',
                          '--data', sample_csv,
                          '--output', output_dir,
                          '--min-window-size', '10',
                          '--max-window-size', '5']):
        main_cli()
    captured = capsys.readouterr()
    # Check for the error message in the rich-formatted output
    assert "min_num cannot be greater" in captured.out and "than max_num" in captured.out


def test_main_interactive_mode(sample_csv, output_dir, mock_llm, capsys):
    """Test main function with interactive mode."""
    # Mock rich Prompt and Confirm
    with patch('gca_analyzer.main.Prompt.ask') as mock_prompt, \
         patch('gca_analyzer.main.Confirm.ask') as mock_confirm:

        # Set up mock responses
        mock_prompt.side_effect = [sample_csv, output_dir]
        mock_confirm.return_value = False  # no advanced settings
        
        with patch('sys.argv', ['gca_analyzer', '--interactive']):
            main_cli()
    
    captured = capsys.readouterr()
    assert "Interactive Configuration Wizard" in captured.out
    assert "Analysis completed" in captured.out


def test_main_no_args_interactive(sample_csv, output_dir, mock_llm, capsys):
    """Test main function with no arguments (should trigger interactive mode)."""
    # Mock rich Prompt and Confirm
    with patch('gca_analyzer.main.Prompt.ask') as mock_prompt, \
         patch('gca_analyzer.main.Confirm.ask') as mock_confirm:

        # Set up mock responses
        mock_prompt.side_effect = [sample_csv, output_dir]
        mock_confirm.return_value = False  # no advanced settings
        
        with patch('sys.argv', ['gca_analyzer']):
            main_cli()
    
    captured = capsys.readouterr()
    assert "Interactive Configuration Wizard" in captured.out


def test_main_interactive_cancelled(capsys):
    """Test main function with interactive mode cancelled."""
    # Mock rich Prompt with invalid file path
    with patch('gca_analyzer.main.Prompt.ask') as mock_prompt:
        mock_prompt.return_value = 'nonexistent.csv'  # invalid data file path
        
        with patch('sys.argv', ['gca_analyzer', '--interactive']):
            main_cli()
    
    captured = capsys.readouterr()
    assert "Configuration cancelled" in captured.out


def test_main_interactive_advanced_settings(sample_csv, output_dir, mock_llm, capsys):
    """Test main function with interactive mode and advanced settings."""
    # Mock rich Prompt and Confirm for advanced settings
    with patch('gca_analyzer.main.Prompt.ask') as mock_prompt, \
         patch('gca_analyzer.main.Confirm.ask') as mock_confirm:

        # Set up mock responses for advanced settings
        mock_prompt.side_effect = [
            sample_csv,  # data file path
            output_dir,  # output directory
            '0.5',  # best window indices
            '3',  # active participant indices
            '3',  # min window size
            '',  # max window size (auto)
            '',  # model name (default)
            '',  # model mirror (default)
            'DEBUG',  # console level
            ''  # log file (skip)
        ]
        mock_confirm.return_value = True  # yes to advanced settings
        
        with patch('sys.argv', ['gca_analyzer', '--interactive']):
            main_cli()
    
    captured = capsys.readouterr()
    assert "Window Configuration" in captured.out
    assert "Model Configuration" in captured.out
    assert "Logging Configuration" in captured.out


def test_main_no_data_arg_non_interactive(capsys):
    """Test main function with no data argument in non-interactive mode."""
    with patch('sys.argv', ['gca_analyzer']):
        # Patch sys.argv to simulate no interactive flag
        with patch('sys.argv', ['gca_analyzer', '--output', 'test']):
            main_cli()
    
    captured = capsys.readouterr()
    assert "--data argument is required in non-interactive mode" in captured.out


def test_cli_helper_functions(capsys):
    """Test CLI helper functions for rich formatting."""
    show_welcome()
    show_info("Test info message")
    show_success("Test success message")
    show_error("Test error message")
    
    captured = capsys.readouterr()
    assert "Welcome" in captured.out
    assert "Test info message" in captured.out
    assert "Test success message" in captured.out
    assert "Test error message" in captured.out

def test_main_entry_point(sample_csv, output_dir, mock_llm):
    """Test __main__.py main entry point for coverage."""
    from gca_analyzer.__main__ import main_cli
    with patch('sys.argv', ['gca_analyzer', '--data', sample_csv, '--output', output_dir]):
        main_cli()
    assert os.path.exists(os.path.join(output_dir, 'descriptive_statistics_gca.csv'))
