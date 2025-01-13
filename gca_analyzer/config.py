"""Configuration Module

This module provides configuration management for the GCA analyzer.

Author: Jianjun Xiao
Email: et_shaw@126.com
Date: 2025-01-12
License: Apache 2.0
"""

from dataclasses import dataclass, field
from typing import Tuple


@dataclass
class WindowConfig:
    """Configuration for conversation window analysis.

    Attributes:
        best_window_indices (float): Target participation threshold for window
            selection (0-1). Defaults to 0.3.
        min_window_size (int): Minimum size of sliding window. Defaults to 2.
        max_window_size (int): Maximum size of sliding window. Defaults to 10.
    """

    best_window_indices: float = 0.3
    min_window_size: int = 2
    max_window_size: int = 10


@dataclass
class ModelConfig:
    """Configuration for language model settings.

    Attributes:
        model_name (str): Name of the pretrained model to use. Defaults to
            'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'.
        embedding_dimension (int): Dimension of the embedding vectors.
            Defaults to 384.
        mirror_url (str): Mirror URL for model downloads. Defaults to
            'https://modelscope.cn/models'.
    """

    model_name: str = (
        'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'
    )
    embedding_dimension: int = 384
    mirror_url: str = 'https://modelscope.cn/models'


@dataclass
class VisualizationConfig:
    """Configuration for visualization settings.

    Attributes:
        default_figsize (Tuple[int, int]): Default figure size for plots.
            Defaults to (10, 6).
        heatmap_figsize (Tuple[int, int]): Figure size for heatmap plots.
            Defaults to (12, 8).
        network_figsize (Tuple[int, int]): Figure size for network plots.
            Defaults to (10, 10).
    """

    default_figsize: Tuple[int, int] = (10, 6)
    heatmap_figsize: Tuple[int, int] = (12, 8)
    network_figsize: Tuple[int, int] = (10, 10)


@dataclass
class LoggerConfig:
    """Configuration for logging settings.

    Attributes:
        console_level (str): Logging level for console output. Defaults to "INFO".
        file_level (str): Logging level for file output. Defaults to "DEBUG".
        log_file (str | None): Path to log file. If None, only console output is used.
            Defaults to None.
        rotation (str): Log file rotation setting. Defaults to "10 MB".
        compression (str): Log file compression format. Defaults to "zip".
        console_format (str): Format string for console output.
        file_format (str): Format string for file output.
    """

    console_level: str = "INFO"
    file_level: str = "DEBUG"
    log_file: str | None = None
    rotation: str = "10 MB"
    compression: str = "zip"
    console_format: str = "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}:{function}:{line}</cyan> | <level>{message}</level>"
    file_format: str = "{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} | {message}"


@dataclass
class Config:
    """Main configuration class for GCA analyzer.

    This class aggregates all configuration components including window
    analysis settings, model settings, and visualization parameters.

    Attributes:
        window (WindowConfig): Configuration for window analysis.
        model (ModelConfig): Configuration for language model.
        visualization (VisualizationConfig): Configuration for visualization.
        logger (LoggerConfig): Configuration for logging.
    """

    window: WindowConfig = field(default_factory=WindowConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    visualization: VisualizationConfig = field(
        default_factory=VisualizationConfig
    )
    logger: LoggerConfig = field(default_factory=LoggerConfig)


# Default configuration instance
default_config = Config()
