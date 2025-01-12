"""Logger Module

This module provides a configured logger for the GCA analyzer package,
with support for console and file output, colored formatting,
and different log levels.

Author: Jianjun Xiao
Email: et_shaw@126.com
Date: 2025-01-12
License: MIT
"""

import os
import sys
from typing import Optional

from loguru import logger


def setup_logger(log_file: Optional[str] = None) -> logger:
    """Setup loguru logger with a beautiful format and optional file output.

    This function configures a loguru logger with colored formatting and
    optional file output. The logger supports different log levels and
    includes timestamp, level, module info, and message in its output.

    Args:
        log_file: Optional path to log file. If None, only console output
            is enabled. The directory will be created if it doesn't exist.

    Returns:
        logger: Configured loguru logger instance

    Example:
        >>> logger = setup_logger("/path/to/logs/app.log")
        >>> logger.info("Application started")
        2025-01-13 02:10:24 | INFO     | __main__:main:1 | Application started
    """
    # Remove default handler
    logger.remove()
    
    # Format for both console and file
    log_format = (
        "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
        "<level>{level: <8}</level> | "
        "<cyan>{name}</cyan>:<cyan>{function}</cyan>:"
        "<cyan>{line}</cyan> | "
        "<level>{message}</level>"
    )
    
    # Add console handler with colors
    logger.add(
        sys.stderr,
        format=log_format,
        level="INFO",
        colorize=True,
        enqueue=True
    )
    
    # Add file handler if log_file is specified
    if log_file:
        os.makedirs(
            os.path.dirname(log_file),
            exist_ok=True
        )
        logger.add(
            log_file,
            format=log_format,
            level="DEBUG",
            rotation="10 MB",
            compression="zip",
            enqueue=True
        )
    
    return logger


# Create a global logger instance
logger = setup_logger()
