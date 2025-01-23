"""Logger Module

This module provides a configured logger for the GCA analyzer package,
with support for console output and optional file output.

Author: Jianjun Xiao
Email: et_shaw@126.com
Date: 2025-01-12
License: Apache 2.0
"""

import sys
from typing import Any, Optional

from loguru import logger as _logger

from .config import Config, default_config


def setup_logger(config: Optional[Config] = None) -> Any:
    """Setup loguru logger with console output and optional file output.

    Args:
        config: Optional configuration instance. If None, uses default_config.

    Returns:
        Any: Configured loguru logger instance
    """
    config = config or default_config
    _logger.remove()

    _logger.add(
        sys.stdout,
        colorize=True,
        format=config.logger.console_format,
        level=config.logger.console_level,
    )

    if config.logger.log_file:
        _logger.add(
            config.logger.log_file,
            format=config.logger.file_format,
            level=config.logger.file_level,
            rotation=config.logger.rotation,
            compression=config.logger.compression,
        )

    return _logger


logger = _logger
