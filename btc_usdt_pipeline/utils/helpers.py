# utils/helpers.py
"""
Utility functions for logging, configuration, data splitting, metrics calculation,
and target variable creation.
"""
import logging
import os
import numpy as np
import pandas as pd
from typing import Tuple, List, Dict, Any, Optional
from pathlib import Path
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from btc_usdt_pipeline.utils.data_processing import calculate_metrics
from btc_usdt_pipeline.exceptions import BacktestError, DataAlignmentError, ParameterValidationError, OptimizationError
from btc_usdt_pipeline.utils.logging_config import setup_logging
from btc_usdt_pipeline.io.serialization import to_json, save_json, load_json
from btc_usdt_pipeline.types import TradeLogType, MetricsDict

# --- Logger Setup ---
utils_logger = setup_logging(log_filename='utils.log')

# --- Data Splitting ---
def split_data(df: pd.DataFrame,
               train_frac: Optional[float] = None,
               val_frac: Optional[float] = None) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
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
def print_trade_summary(trade_log: TradeLogType, num_trades: int = 10) -> None:
    """Logs a summary of the first and last few trades."""
    if not trade_log:
        utils_logger.info("No trades executed.")
        return

    utils_logger.info(f"--- Trade Log Summary (First {num_trades} and Last {num_trades}) ---")
    header = f"{'Type':<5} {'Entry Idx':<20} {'Entry Pr':>10} {'Exit Idx':<20} {'Exit Pr':>10} {'Size':>10} {'PnL':>12} {'Reason':<8}"
    utils_logger.info(header)
    utils_logger.info("-" * len(header))

    trades_to_show = trade_log[:num_trades]
    if len(trade_log) > 2 * num_trades:
        trades_to_show.append({'Type': '...'})  # Separator
        trades_to_show.extend(trade_log[-num_trades:])
    elif len(trade_log) > num_trades:
        trades_to_show.extend(trade_log[num_trades:])

    for trade in trades_to_show:
        if trade.get('Type') == '...':
            utils_logger.info("...")
            continue

        entry_idx_str = str(trade.get('Entry_idx', 'N/A'))
        exit_idx_str = str(trade.get('Exit_idx', 'N/A'))
        if len(entry_idx_str) > 19: entry_idx_str = entry_idx_str[:19]
        if len(exit_idx_str) > 19: exit_idx_str = exit_idx_str[:19]

        utils_logger.info(f"{trade.get('Type', '?'):<5} "
              f"{entry_idx_str:<20} "
              f"{trade.get('Entry', 0.0):>10.2f} "
              f"{exit_idx_str:<20} "
              f"{trade.get('Exit', 0.0):>10.2f} "
              f"{trade.get('Size', 0.0):>10.4f} "
              f"{trade.get('PnL', 0.0):>12.2f} "
              f"{trade.get('Exit Reason', ''):<8}")
    utils_logger.info("-" * len(header))

# --- Target Variable Creation ---
def make_binary_target(df: pd.DataFrame,
                       future_window: Optional[int] = None,
                       threshold_usd: Optional[float] = None,
                       target_col_name: Optional[str] = None) -> pd.DataFrame:
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
    logger = setup_logging('utils.log')
    logger.info(f"Created target '{target_col_name}' with window={future_window}, threshold=${threshold_usd:.2f}. Distribution:\n{target_counts}")

    return df

# --- Sequence Creation for RNNs ---
def create_sequences(data: np.ndarray,
                     targets: np.ndarray,
                     timesteps: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Creates sequences for LSTM/GRU models using TimeseriesGenerator.
    Uses lazy loading for TensorFlow to avoid compatibility issues.

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
    
    # Lazy loading for TensorFlow
    try:
        from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator
        generator = TimeseriesGenerator(data, targets, length=timesteps, batch_size=len(data))
        # The generator yields batches, here we take the only batch
        X_seq, y_seq = generator[0]
        return X_seq, y_seq
    except ImportError as e:
        utils_logger.warning(f"Failed to import TensorFlow: {e}. Using manual sequence creation.")
        # Fallback implementation without TensorFlow
        samples = len(data) - timesteps + 1
        X_seq = np.zeros((samples, timesteps, data.shape[1]))
        y_seq = np.zeros(samples)
        
        for i in range(samples):
            X_seq[i] = data[i:i+timesteps]
            y_seq[i] = targets[i+timesteps-1]
            
        return X_seq, y_seq

# --- Plot Equity Curve ---
def plot_equity_curve(equity_curve: List[float],
                      index: pd.DatetimeIndex,
                      save_path: Optional[Path] = None) -> None:
    """Plots the equity curve over time."""
    logger = setup_logging('utils.log')
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
