"""
colab_utils.py

Utilities for Google Colab and Google Drive integration, checkpointing, and memory monitoring.

Example usage:
    from btc_usdt_pipeline.utils.colab_utils import (
        is_colab, mount_gdrive, get_gdrive_path, save_checkpoint, load_checkpoint, monitor_memory
    )

    if is_colab():
        mount_gdrive()
        data_path = get_gdrive_path('MyDrive/datasets/btc_data.parquet')
    else:
        data_path = './data/btc_data.parquet'

    # Checkpointing
    save_checkpoint(obj, 'checkpoint.pkl')
    obj = load_checkpoint('checkpoint.pkl')

    # Memory monitoring (Colab only)
    monitor_memory(threshold_gb=10)
"""
import os
import pickle
import time
from pathlib import Path
from btc_usdt_pipeline.utils.logging_config import setup_logging
from btc_usdt_pipeline.monitoring.memory import monitor_memory

logger = setup_logging(log_filename='colab_utils.log')


def is_colab():
    """Detect if running in Google Colab environment."""
    try:
        import google.colab  # noqa: F401
        return True
    except ImportError:
        return False


def mount_gdrive():
    """Mount Google Drive in Colab if not already mounted."""
    if is_colab():
        from google.colab import drive
        if not Path('/content/drive').exists():
            drive.mount('/content/drive')


def get_gdrive_path(relative_path):
    """
    Get the full path to a file/folder in Google Drive.
    Args:
        relative_path (str): Path relative to 'MyDrive'.
    Returns:
        str: Full path in Colab environment.
    """
    if is_colab():
        return os.path.join('/content/drive', relative_path)
    return relative_path


def save_checkpoint(study, iteration, base_path):
    """
    Save an Optuna study or any object as a checkpoint.
    Args:
        study: The Optuna study or object to save.
        iteration (int): The current iteration or trial number.
        base_path (str or Path): Directory to save the checkpoint.
    Returns:
        str: Path to the saved checkpoint.
    """
    Path(base_path).mkdir(parents=True, exist_ok=True)
    checkpoint_path = Path(base_path) / f"checkpoint_iter_{iteration}.pkl"
    try:
        with open(checkpoint_path, 'wb') as f:
            pickle.dump(study, f)
        logger.info(f"Checkpoint saved: {checkpoint_path}")
        return str(checkpoint_path)
    except Exception as e:
        logger.error(f"Failed to save checkpoint: {checkpoint_path}: {e}")
        return None


def load_checkpoint(checkpoint_path):
    """
    Load a checkpointed Optuna study or object.
    Args:
        checkpoint_path (str or Path): Path to the checkpoint file.
    Returns:
        The loaded object, or None if loading fails.
    """
    try:
        with open(checkpoint_path, 'rb') as f:
            obj = pickle.load(f)
        logger.info(f"Checkpoint loaded: {checkpoint_path}")
        return obj
    except Exception as e:
        logger.error(f"Failed to load checkpoint: {checkpoint_path}: {e}")
        return None


def verify_checkpoint(checkpoint_path):
    """
    Verify that a checkpoint file is readable and valid.
    Returns True if valid, False otherwise.
    """
    obj = load_checkpoint(checkpoint_path)
    if obj is not None:
        logger.info(f"Checkpoint verified: {checkpoint_path}")
        return True
    logger.warning(f"Checkpoint verification failed: {checkpoint_path}")
    return False


def find_latest_checkpoint(base_path):
    """
    Find the latest valid checkpoint in a directory.
    Returns the path to the latest valid checkpoint, or None if none found.
    """
    base_path = Path(base_path)
    checkpoints = sorted(base_path.glob("checkpoint_iter_*.pkl"), key=lambda p: int(p.stem.split('_')[-1]), reverse=True)
    for ckpt in checkpoints:
        if verify_checkpoint(ckpt):
            logger.info(f"Latest valid checkpoint found: {ckpt}")
            return str(ckpt)
    logger.warning(f"No valid checkpoint found in {base_path}")
    return None


class PeriodicCheckpointer:
    """
    Utility for automatic periodic checkpointing in long-running loops.
    Usage:
        checkpointer = PeriodicCheckpointer(study, base_path, frequency)
        for i in range(...):
            ...
            checkpointer.maybe_checkpoint(i)
    """
    def __init__(self, obj, base_path, frequency=10):
        self.obj = obj
        self.base_path = base_path
        self.frequency = frequency

    def maybe_checkpoint(self, iteration):
        if self.frequency > 0 and iteration % self.frequency == 0:
            save_checkpoint(self.obj, iteration, self.base_path)
