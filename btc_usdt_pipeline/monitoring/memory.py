import psutil
import time
import logging
from functools import wraps
from btc_usdt_pipeline.utils.logging_config import setup_logging

logger = setup_logging(log_filename='memory.log')

def monitor_memory(threshold_gb=11, check_interval=30):
    """
    Monitor RAM usage and warn if above threshold.
    Args:
        threshold_gb (float): RAM usage threshold in GB.
        check_interval (int): Seconds between checks.
    """
    try:
        import IPython.display as display
        while True:
            mem = psutil.virtual_memory()
            used_gb = mem.used / 1e9
            if used_gb > threshold_gb:
                logger.warning(f"[WARNING] High RAM usage: {used_gb:.2f} GB")
                display.clear_output(wait=True)
            time.sleep(check_interval)
    except ImportError:
        logger.error("psutil not installed. Run `!pip install psutil`.")

def check_memory_usage():
    """
    Logs and returns the current memory usage percentage.
    Returns:
        float: Memory usage percentage.
    """
    mem = psutil.virtual_memory()
    usage_percent = mem.percent
    logger.info(f"[Memory] Current memory usage: {usage_percent:.2f}%")
    return usage_percent

def memory_safe(min_free_percent=10):
    """
    Decorator to check memory usage before function execution.
    If free memory is below min_free_percent, aborts execution.
    Args:
        min_free_percent (float): Minimum free memory percent required to run.
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            mem = psutil.virtual_memory()
            free_percent = 100 - mem.percent
            if free_percent < min_free_percent:
                logger.warning(f"[Memory] ABORT: Not enough free memory ({free_percent:.2f}% < {min_free_percent}%)")
                return None
            return func(*args, **kwargs)
        return wrapper
    return decorator
