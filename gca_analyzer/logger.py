"""Logger Module

This module provides a configured logger for the GCA analyzer package,
with support for console output and optional file output.

Author: Jianjun Xiao
Email: et_shaw@126.com
Date: 2025-01-12
License: Apache 2.0
"""

import sys
from typing import Optional
from loguru import logger

from .config import Config, default_config


def setup_logger(config: Optional[Config] = None) -> logger:
    """Setup loguru logger with console output and optional file output.

    Args:
        config: Optional configuration instance. If None, uses default_config.

    Returns:
        logger: Configured loguru logger instance
    """
    config = config or default_config
    logger.remove()

    logger.add(
        sys.stdout,
        colorize=True,
        format=config.logger.console_format,
        level=config.logger.console_level
    )

    if config.logger.log_file:
        logger.add(
            config.logger.log_file,
            format=config.logger.file_format,
            level=config.logger.file_level,
            rotation=config.logger.rotation,
            compression=config.logger.compression
        )

    return logger


logger = setup_logger()
