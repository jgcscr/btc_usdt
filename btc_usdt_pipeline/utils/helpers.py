# utils/helpers.py
"""
Utility functions for logging, configuration, data splitting, metrics calculation,
and target variable creation.
"""
import logging
import logging.handlers
import os
import json
import numpy as np
import pandas as pd
from typing import Tuple, List, Dict, Any, Optional, Union
from pathlib import Path
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from btc_usdt_pipeline.utils.data_processing import calculate_metrics

# --- Logger Setup ---
def setup_logger(log_filename: str, level: str = None, logs_dir: 'Path' = None) -> logging.Logger:
    """
    Sets up a logger that writes to a file and console.
    Uses configuration for log level, format, and base directory.
    Adds rotating file handler.
    """
    from btc_usdt_pipeline import config
    level = level or config.LOG_LEVEL
    logs_dir = logs_dir or config.LOGS_DIR
    logs_dir.mkdir(parents=True, exist_ok=True)

    log_filepath = logs_dir / log_filename

    logger = logging.getLogger(log_filename.replace('.log', ''))  # Use filename base as logger name
    logger.setLevel(level)

    # Prevent adding multiple handlers if called repeatedly
    if logger.hasHandlers():
        logger.handlers.clear()

    formatter = logging.Formatter(config.LOG_FORMAT, datefmt=config.LOG_DATE_FORMAT)

    # Console Handler
    ch = logging.StreamHandler()
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    # Rotating File Handler (e.g., 5 files, 5MB each)
    fh = logging.handlers.RotatingFileHandler(log_filepath, maxBytes=5*1024*1024, backupCount=5)
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    return logger

# Setup a general logger for utils functions themselves using the name from config
utils_logger = setup_logger('utils.log')

# --- JSON Handling ---
def save_json(data: Dict, file_path: Path) -> None:
    """Saves data to a JSON file."""
    logger = setup_logger('utils.log')
    try:
        file_path.parent.mkdir(parents=True, exist_ok=True)
        with open(file_path, 'w') as f:
            json.dump(data, f, indent=4)
        logger.info(f"Saved JSON data to {file_path}")
    except Exception as e:
        logger.error(f"Error saving JSON file {file_path}: {e}")

def load_json(file_path: Path) -> Optional[Dict]:
    """Loads data from a JSON file."""
    logger = setup_logger('utils.log')
    if not file_path.exists():
        logger.error(f"JSON file not found: {file_path}")
        return None
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        logger.info(f"Loaded JSON data from {file_path}")
        return data
    except json.JSONDecodeError as e:
        logger.error(f"Error decoding JSON from {file_path}: {e}")
        return None
    except Exception as e:
        logger.error(f"Error reading JSON file {file_path}: {e}")
        return None

# --- Data Splitting ---
def split_data(df: pd.DataFrame,
               train_frac: float = None,
               val_frac: float = None) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split DataFrame into train, validation, and test sets by time order.
    Returns: train_df, val_df, test_df
    """
    from btc_usdt_pipeline import config
    train_frac = train_frac if train_frac is not None else config.TRAIN_FRAC
    val_frac = val_frac if val_frac is not None else config.VAL_FRAC

    n = len(df)
    if not (0 < train_frac < 1 and 0 < val_frac < 1 and train_frac + val_frac < 1):
        raise ValueError("train_frac and val_frac must be between 0 and 1, and their sum must be less than 1")

    train_end = int(n * train_frac)
    val_end = int(n * (train_frac + val_frac))

    train_df = df.iloc[:train_end]
    val_df = df.iloc[train_end:val_end]
    test_df = df.iloc[val_end:]

    utils_logger.info(f"Data split: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")
    return train_df, val_df, test_df

# --- Trade Log Summary ---
def print_trade_summary(trade_log: List[Dict[str, Any]], num_trades: int = 10) -> None:
    """Prints a summary of the first and last few trades."""
    if not trade_log:
        print("No trades executed.")
        return

    print(f"\n--- Trade Log Summary (First {num_trades} and Last {num_trades}) ---")
    header = f"{'Type':<5} {'Entry Idx':<20} {'Entry Pr':>10} {'Exit Idx':<20} {'Exit Pr':>10} {'Size':>10} {'PnL':>12} {'Reason':<8}"
    print(header)
    print("-" * len(header))

    trades_to_show = trade_log[:num_trades]
    if len(trade_log) > 2 * num_trades:
        trades_to_show.append({'Type': '...'})  # Separator
        trades_to_show.extend(trade_log[-num_trades:])
    elif len(trade_log) > num_trades:
        trades_to_show.extend(trade_log[num_trades:])

    for trade in trades_to_show:
        if trade.get('Type') == '...':
            print("...")
            continue

        entry_idx_str = str(trade.get('Entry_idx', 'N/A'))
        exit_idx_str = str(trade.get('Exit_idx', 'N/A'))
        if len(entry_idx_str) > 19: entry_idx_str = entry_idx_str[:19]
        if len(exit_idx_str) > 19: exit_idx_str = exit_idx_str[:19]

        print(f"{trade.get('Type', '?'):<5} "
              f"{entry_idx_str:<20} "
              f"{trade.get('Entry', 0.0):>10.2f} "
              f"{exit_idx_str:<20} "
              f"{trade.get('Exit', 0.0):>10.2f} "
              f"{trade.get('Size', 0.0):>10.4f} "
              f"{trade.get('PnL', 0.0):>12.2f} "
              f"{trade.get('Exit Reason', ''):<8}")
    print("-" * len(header))

# --- Target Variable Creation ---
def make_binary_target(df: pd.DataFrame, future_window: int = None, threshold_usd: float = None, target_col_name: str = None) -> pd.DataFrame:
    """
    Creates a binary target column based on future price movement.
    1 if the price moves up by at least threshold_usd within future_window bars.
    0 otherwise (includes moves down or insufficient moves up).
    Uses parameters from config by default.

    Args:
        df (pd.DataFrame): DataFrame with 'close' price column.
        future_window (int): How many bars into the future to look.
        threshold_usd (float): The minimum USD upward movement required for target = 1.
        target_col_name (str): The name for the new target column.

    Returns:
        pd.DataFrame: DataFrame with the added binary target column.
    """
    from btc_usdt_pipeline import config
    future_window = future_window if future_window is not None else config.TARGET_FUTURE_WINDOW
    threshold_usd = threshold_usd if threshold_usd is not None else config.TARGET_THRESHOLD_USD
    target_col_name = target_col_name if target_col_name is not None else config.TARGET_COLUMN_NAME

    if 'close' not in df.columns:
        raise ValueError("DataFrame must contain a 'close' column.")

    # Calculate future max price within the window
    df['future_max_close'] = df['close'].rolling(window=future_window, closed='left').max().shift(-future_window)

    # Calculate the price change
    df['future_price_change'] = df['future_max_close'] - df['close']

    # Create binary target
    df[target_col_name] = (df['future_price_change'] >= threshold_usd).astype(int)

    # Clean up intermediate columns
    df = df.drop(columns=['future_max_close', 'future_price_change'])

    # Log distribution
    target_counts = df[target_col_name].value_counts(normalize=True)
    logger = setup_logger('utils.log')
    logger.info(f"Created target '{target_col_name}' with window={future_window}, threshold=${threshold_usd:.2f}. Distribution:\n{target_counts}")

    return df

# --- Sequence Creation for RNNs ---
def create_sequences(data: np.ndarray, targets: np.ndarray, timesteps: int = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Creates sequences for LSTM/GRU models using TimeseriesGenerator.

    Args:
        data (np.ndarray): Feature data (rows=samples, cols=features).
        targets (np.ndarray): Target data (1D array).
        timesteps (int): Number of timesteps in each sequence.

    Returns:
        Tuple[np.ndarray, np.ndarray]: X_seq (samples, timesteps, features), y_seq (samples,)
    """
    from btc_usdt_pipeline import config
    timesteps = timesteps if timesteps is not None else config.SEQUENCE_TIMESTEPS

    if len(data) <= timesteps:
        # Not enough data to create sequences
        return np.empty((0, timesteps, data.shape[1])), np.empty((0,))

    generator = TimeseriesGenerator(data, targets, length=timesteps, batch_size=len(data))
    # The generator yields batches, here we take the only batch
    X_seq, y_seq = generator[0]
    return X_seq, y_seq

# --- Plot Equity Curve ---
def plot_equity_curve(equity_curve: List[float], index: pd.DatetimeIndex, save_path: Optional[Path] = None) -> None:
    """Plots the equity curve over time."""
    logger = setup_logger('utils.log')
    if not equity_curve or len(equity_curve) - 1 != len(index):
        logger.error(f"Equity curve length ({len(equity_curve)}) must be one greater than index length ({len(index)}) for plotting.")
        print("Error: Cannot plot equity curve due to length mismatch.")
        return

    plt.style.use('seaborn-v0_8-darkgrid')
    fig, ax = plt.subplots(figsize=(12, 6))

    # Plot equity curve (use index for x-axis, skip initial equity point for alignment)
    ax.plot(index, equity_curve[1:], label='Equity Curve', color='blue')

    ax.set_title('Backtest Equity Curve')
    ax.set_xlabel('Time')
    ax.set_ylabel('Equity ($)')
    ax.legend()
    ax.grid(True)

    # Format x-axis dates
    fig.autofmt_xdate()
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))

    plt.tight_layout()

    if save_path:
        try:
            save_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path)
            logger.info(f"Equity curve plot saved to {save_path}")
        except Exception as e:
            logger.error(f"Error saving equity curve plot to {save_path}: {e}")
    else:
        plt.show()

    plt.close(fig)  # Close the figure to free memory
