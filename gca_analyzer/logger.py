"""Logger Module

This module provides a configured logger for the GCA analyzer package,
with support for console output and optional file output using a tree structure.

Author: Jianjun Xiao
Email: et_shaw@126.com
Date: 2025-01-12
License: Apache 2.0
"""

import sys
from typing import Optional
from rich.console import Console
from rich.tree import Tree
from rich.text import Text
from rich import print as rprint
from loguru import logger

console = Console()

class TreeHandler:
    def __init__(self):
        self.tree = Tree("GCA Analyzer Log")

    def write(self, message):
        level, msg = message.record["level"].name, message.record["message"]
        node = self.tree.add(
            Text(f"[{level}] {msg}", style=f"bold {self._get_color(level)}")
        )
        node.add(Text(f"{message.record['name']}:{message.record['function']}:{message.record['line']}", style="cyan"))
        console.print(self.tree)
        self.tree = Tree("GCA Analyzer Log")

    def _get_color(self, level):
        colors = {
            "DEBUG": "blue",
            "INFO": "green",
            "WARNING": "yellow",
            "ERROR": "red",
            "CRITICAL": "red bold",
        }
        return colors.get(level, "white")

def setup_logger(log_file: Optional[str] = None) -> logger:
    """Setup loguru logger with tree-structured console output and optional file output.

    Args:
        log_file: Optional path to log file. If None, only console output is used.

    Returns:
        logger: Configured loguru logger instance
    """
    logger.remove()

    tree_handler = TreeHandler()
    logger.add(tree_handler.write, colorize=True, level="INFO")

    if log_file:
        logger.add(
            log_file,
            format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} | {message}",
            level="DEBUG",
            rotation="10 MB",
            compression="zip"
        )

    return logger

logger = setup_logger()
