"""Structured logging infrastructure."""

import logging
import sys
from pathlib import Path
from typing import Optional
from .config import LoggingConfig


# Global logger registry
_loggers = {}


def setup_logger(
    name: str,
    config: Optional[LoggingConfig] = None,
    log_file: Optional[str] = None,
    level: Optional[str] = None
) -> logging.Logger:
    """
    Set up a logger with console and file handlers.
    
    Args:
        name: Logger name
        config: LoggingConfig object (optional)
        log_file: Path to log file (overrides config)
        level: Log level (overrides config)
        
    Returns:
        Configured logger instance
    """
    # Use config or defaults
    if config is None:
        config = LoggingConfig()
    
    log_level = level or config.level
    log_file_path = log_file or config.file
    log_format = config.format
    
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, log_level))
    
    # Remove existing handlers to avoid duplicates
    logger.handlers.clear()
    
    # Create formatter
    formatter = logging.Formatter(log_format)
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, log_level))
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler (if specified)
    if log_file_path:
        log_path = Path(log_file_path)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_file_path)
        file_handler.setLevel(getattr(logging, log_level))
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    # Store in registry
    _loggers[name] = logger
    
    return logger


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger by name. Creates a default logger if not found.
    
    Args:
        name: Logger name
        
    Returns:
        Logger instance
    """
    if name in _loggers:
        return _loggers[name]
    
    # Create default logger if not found
    return setup_logger(name)


class LoggerMixin:
    """Mixin class to add logging capability to any class."""
    
    @property
    def logger(self) -> logging.Logger:
        """Get logger for this class."""
        logger_name = f"{self.__class__.__module__}.{self.__class__.__name__}"
        return get_logger(logger_name)
