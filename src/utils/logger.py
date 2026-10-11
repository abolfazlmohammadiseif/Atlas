"""
Centralized logging configuration module for the application.
Provides a structured logger instance with environment-aware formatting.
"""

import logging
import sys
from typing import Final

from src.utils.config import settings

# Default logging format string
LOG_FORMAT: Final[str] = (
    "%(asctime)s | %(levelname)-8s | %(name)s:%(funcName)s:%(lineno)d - %(message)s"
)


def setup_logger(name: str = "ai_agent") -> logging.Logger:
    """Configures and returns a structured Python logger.

    Args:
        name: Name of the logger module.

    Returns:
        logging.Logger: Configured logger instance.
    """
    logger = logging.getLogger(name)

    # Prevent duplicate handlers if already configured
    if logger.hasHandlers():
        return logger

    # Set log level from application configuration
    log_level = getattr(logging, settings.log_level.upper(), logging.INFO)
    logger.setLevel(log_level)

    # Console Handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)

    # Formatter
    formatter = logging.Formatter(LOG_FORMAT, datefmt="%Y-%m-%d %H:%M:%S")
    console_handler.setFormatter(formatter)

    logger.addHandler(console_handler)
    return logger


# Global default logger instance
logger = setup_logger()


if __name__ == "__main__":
    logger.info("Starting Autonomous Research Agent Logger test...")
    logger.debug("Debug mode activated: Environment is %s", settings.environment)
    logger.warning("This is a warning log example.")
    logger.error("This is an error log example.")
