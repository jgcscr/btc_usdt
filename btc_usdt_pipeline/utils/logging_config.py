import logging
import os
from pathlib import Path
from typing import Optional

def setup_logging(
    log_filename: str = 'btc_usdt_pipeline.log',
    log_dir: Optional[str] = None,
    level: str = 'INFO',
    console: bool = True,
    file: bool = True,
    log_format: str = '%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    date_format: str = '%Y-%m-%d %H:%M:%S'
) -> logging.Logger:
    """
    Set up and return a logger with consistent formatting and configurable handlers.
    Args:
        log_filename (str): Name of the log file.
        log_dir (Optional[str]): Directory to store log files. Defaults to current directory.
        level (str): Logging level (e.g., 'INFO', 'DEBUG').
        console (bool): Whether to log to console.
        file (bool): Whether to log to a file.
        log_format (str): Log message format.
        date_format (str): Date format for log messages.
    Returns:
        logging.Logger: Configured logger instance.
    """
    logger = logging.getLogger(log_filename)
    logger.setLevel(level.upper())
    formatter = logging.Formatter(log_format, datefmt=date_format)

    # Remove existing handlers to avoid duplicate logs
    if logger.hasHandlers():
        logger.handlers.clear()

    if file:
        log_dir = log_dir or os.getcwd()
        Path(log_dir).mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(os.path.join(log_dir, log_filename))
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    if console:
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    return logger
