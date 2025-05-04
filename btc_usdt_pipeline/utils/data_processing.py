"""
data_processing.py

Utility functions for DataFrame optimization, preprocessing, and metrics calculation.
"""
import pandas as pd
import numpy as np
from typing import Optional
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, log_loss

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
