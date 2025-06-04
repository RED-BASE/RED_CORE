"""
Centralized logging configuration for RED_CORE.
Provides structured logging for experiments, analysis, and debugging.
"""

import logging
import sys
from pathlib import Path
from typing import Optional


def setup_logger(
    name: str,
    level: str = "INFO",
    log_file: Optional[Path] = None,
    format_style: str = "structured"
) -> logging.Logger:
    """
    Setup a logger with consistent formatting across RED_CORE.
    
    Args:
        name: Logger name (typically __name__)
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional file path for logging output
        format_style: 'structured' for detailed logs, 'simple' for basic output
    
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper()))
    
    # Remove existing handlers to avoid duplicates
    logger.handlers.clear()
    
    # Choose format based on style
    if format_style == "structured":
        formatter = logging.Formatter(
            '%(asctime)s | %(name)s | %(levelname)s | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
    else:  # simple
        formatter = logging.Formatter('%(levelname)s: %(message)s')
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler if specified
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


def get_experiment_logger(experiment_code: str) -> logging.Logger:
    """Get a logger specifically configured for experiment runs."""
    return setup_logger(
        f"red_core.experiment.{experiment_code}",
        level="INFO",
        format_style="structured"
    )


def get_analysis_logger() -> logging.Logger:
    """Get a logger for analysis scripts."""
    return setup_logger(
        "red_core.analysis",
        level="INFO", 
        format_style="simple"
    )


def get_debug_logger(name: str) -> logging.Logger:
    """Get a debug logger for development."""
    return setup_logger(
        f"red_core.debug.{name}",
        level="DEBUG",
        format_style="structured"
    )