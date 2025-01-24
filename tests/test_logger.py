import os
import sys
import pytest
from loguru import logger
from gca_analyzer.logger import setup_logger
from gca_analyzer.config import Config

@pytest.fixture
def temp_config(tmp_path):
    """Create a temporary config with custom logger settings."""
    class TestConfig:
        class LoggerConfig:
            log_file = str(tmp_path / "test.log")
            console_level = "DEBUG"
            file_level = "INFO"
            console_format = "<level>{level: <8}</level> <green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> - {message}"
            file_format = "{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {message}"
            rotation = "1 day"
            compression = "zip"
        logger = LoggerConfig()
    return TestConfig()

def test_setup_logger_with_config(temp_config):
    """Test setup_logger with custom config."""
    logger = setup_logger(temp_config)
    
    # Verify logger is configured
    assert logger is not None
    assert os.path.exists(temp_config.logger.log_file)
    
    # Test logging to file
    test_message = "Test info message"
    logger.info(test_message)
    
    with open(temp_config.logger.log_file) as f:
        log_content = f.read()
        assert test_message in log_content

def test_setup_logger_without_config():
    """Test setup_logger without config (using default config)."""
    from io import StringIO
    import sys
    
    # Create a string buffer to capture output
    string_io = StringIO()
    
    # Remove any existing handlers and add our own
    logger.remove()
    logger.add(string_io, format="{message}", level="DEBUG")
    
    # Test logging
    logger.debug("Test debug message")
    
    # Get output and verify
    output = string_io.getvalue()
    assert "Test debug message" in output
    
    # Clean up
    logger.remove()

def test_setup_logger_with_invalid_file(tmp_path):
    """Test setup_logger with invalid log file path."""
    class InvalidConfig:
        class LoggerConfig:
            log_file = str(tmp_path / "nonexistent" / "directory" / "test.log")
            console_level = "INFO"
            file_level = "INFO"
            console_format = "{message}"
            file_format = "{message}"
            rotation = None
            compression = None
        logger = LoggerConfig()
    
    # The logger will create parent directories, so we need to make the parent read-only
    nonexistent_dir = tmp_path / "nonexistent"
    nonexistent_dir.mkdir()
    os.chmod(str(nonexistent_dir), 0o444)  # Read-only
    
    try:
        with pytest.raises(OSError):
            setup_logger(InvalidConfig())
    finally:
        os.chmod(str(nonexistent_dir), 0o777)  # Restore permissions

def test_setup_logger_multiple_calls(temp_config):
    """Test setup_logger with multiple calls."""
    logger1 = setup_logger(temp_config)
    logger2 = setup_logger(temp_config)
    
    assert logger1 is logger2  # Should return same logger instance
    
    # Test both console and file logging
    test_message = "Test multiple loggers"
    logger1.info(test_message)
    
    with open(temp_config.logger.log_file) as f:
        log_content = f.read()
        assert test_message in log_content

def test_setup_logger_log_levels(temp_config):
    """Test setup_logger with different log levels."""
    logger = setup_logger(temp_config)
    
    # Debug message should appear in console but not in file
    debug_message = "Debug test message"
    logger.debug(debug_message)
    
    # Info message should appear in both console and file
    info_message = "Info test message"
    logger.info(info_message)
    
    with open(temp_config.logger.log_file) as f:
        log_content = f.read()
        assert info_message in log_content
        assert debug_message not in log_content  # Debug messages shouldn't be in file
