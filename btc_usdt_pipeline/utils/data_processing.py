"""
data_processing.py

Utility functions for DataFrame optimization, preprocessing, and metrics calculation.
"""
import pandas as pd
import numpy as np
from typing import Optional
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, log_loss
from btc_usdt_pipeline.utils.helpers import DataAlignmentError, setup_logger

def optimize_dataframe_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """
    Downcast numeric columns to save memory.
    Args:
        df (pd.DataFrame): Input DataFrame.
    Returns:
        pd.DataFrame: DataFrame with optimized dtypes.
    Example:
        df = optimize_dataframe_dtypes(df)
    """
    for col in df.select_dtypes(include=["float64", "float32"]).columns:
        df[col] = pd.to_numeric(df[col], downcast="float")
    for col in df.select_dtypes(include=["int64", "int32"]).columns:
        df[col] = pd.to_numeric(df[col], downcast="integer")
    return df

def optimize_memory_usage(df: pd.DataFrame, logger=None) -> pd.DataFrame:
    """
    Optimize DataFrame memory usage by downcasting numerics, converting categoricals, and handling datetimes.
    Logs memory usage before and after optimization.
    Args:
        df (pd.DataFrame): Input DataFrame.
        logger (logging.Logger, optional): Logger for memory usage info.
    Returns:
        pd.DataFrame: Optimized DataFrame.
    """
    import gc
    start_mem = df.memory_usage(deep=True).sum() / 1024 ** 2
    if logger:
        logger.info(f"Memory usage before optimization: {start_mem:.2f} MB")
    else:
        print(f"Memory usage before optimization: {start_mem:.2f} MB")

    # Downcast numeric columns
    for col in df.select_dtypes(include=["float64", "float32"]).columns:
        df[col] = pd.to_numeric(df[col], downcast="float")
    for col in df.select_dtypes(include=["int64", "int32"]).columns:
        df[col] = pd.to_numeric(df[col], downcast="integer")

    # Convert object columns to category if low cardinality
    for col in df.select_dtypes(include=["object"]).columns:
        num_unique = df[col].nunique(dropna=False)
        num_total = len(df[col])
        if num_total > 0 and num_unique / num_total < 0.5:
            try:
                df[col] = df[col].astype("category")
            except Exception:
                pass

    # Convert columns that look like datetimes
    for col in df.columns:
        if df[col].dtype == "object":
            try:
                parsed = pd.to_datetime(df[col], errors="raise")
                if not parsed.isnull().all():
                    df[col] = parsed
            except Exception:
                pass

    gc.collect()
    end_mem = df.memory_usage(deep=True).sum() / 1024 ** 2
    if logger:
        logger.info(f"Memory usage after optimization: {end_mem:.2f} MB (reduced by {start_mem - end_mem:.2f} MB)")
    else:
        print(f"Memory usage after optimization: {end_mem:.2f} MB (reduced by {start_mem - end_mem:.2f} MB)")
    return df

def preprocess_data(df: pd.DataFrame, sort_by: Optional[str] = None) -> pd.DataFrame:
    """
    Common preprocessing: handle missing values, sort, etc.
    Args:
        df (pd.DataFrame): Input DataFrame.
        sort_by (str, optional): Column to sort by.
    Returns:
        pd.DataFrame: Preprocessed DataFrame.
    Example:
        df = preprocess_data(df, sort_by='open_time')
    """
    df = df.copy()
    # Drop duplicate rows
    df = df.drop_duplicates()
    # Optionally sort
    if sort_by and sort_by in df.columns:
        df = df.sort_values(by=sort_by)
    # Reset index
    df = df.reset_index(drop=True)
    # Fill missing values (customize as needed)
    df = df.fillna(method='ffill').fillna(method='bfill')
    return df

def calculate_metrics(y_true, y_pred) -> dict:
    """
    Calculate common classification metrics.
    Args:
        y_true: Ground truth labels.
        y_pred: Predicted labels or probabilities.
    Returns:
        dict: Dictionary of metrics.
    Example:
        metrics = calculate_metrics(y_true, y_pred)
    """
    metrics = {}
    try:
        metrics['accuracy'] = accuracy_score(y_true, y_pred)
        metrics['precision'] = precision_score(y_true, y_pred, zero_division=0)
        metrics['recall'] = recall_score(y_true, y_pred, zero_division=0)
        metrics['f1'] = f1_score(y_true, y_pred, zero_division=0)
        if np.issubdtype(np.array(y_pred).dtype, np.floating):
            metrics['log_loss'] = log_loss(y_true, y_pred)
    except Exception as e:
        metrics['error'] = str(e)
    return metrics

def align_and_validate_data(df: pd.DataFrame, arr, arr_name="signals", index_col=None, logger=None):
    """
    Validates and aligns a DataFrame and a signals/features array.
    Handles datetime and integer indices. Returns aligned (df, arr) or raises DataAlignmentError.
    Logs warnings for dropped data points.
    """
    logger = logger or setup_logger('utils.log')
    orig_len = len(df)
    arr_len = len(arr)
    # If index_col is provided, ensure it's the index
    if index_col and index_col in df.columns:
        df = df.set_index(index_col)
    # If lengths match and index is monotonic, return as is
    if orig_len == arr_len:
        if isinstance(df.index, (pd.DatetimeIndex, pd.Int64Index, pd.RangeIndex)):
            if not df.index.is_monotonic_increasing:
                logger.warning("DataFrame index is not monotonic increasing. Sorting.")
                df = df.sort_index()
        return df, arr
    # If lengths mismatch, try to align by index if arr is a pandas Series with index
    if hasattr(arr, 'index') and isinstance(arr.index, (pd.DatetimeIndex, pd.Int64Index, pd.RangeIndex)):
        common_idx = df.index.intersection(arr.index)
        if len(common_idx) == 0:
            logger.error(f"No overlapping indices between DataFrame and {arr_name}.")
            raise DataAlignmentError(f"No overlapping indices between DataFrame and {arr_name}.")
        aligned_df = df.loc[common_idx]
        aligned_arr = arr.loc[common_idx]
        dropped_df = orig_len - len(aligned_df)
        dropped_arr = arr_len - len(aligned_arr)
        logger.warning(f"Alignment dropped {dropped_df} rows from DataFrame and {dropped_arr} from {arr_name}.")
        logger.info(f"Alignment report: matched {len(aligned_df)} rows.")
        return aligned_df, aligned_arr
    logger.error(f"Cannot align DataFrame and {arr_name}: length mismatch and no index to align.")
    raise DataAlignmentError(f"Cannot align DataFrame and {arr_name}: length mismatch and no index to align.")
