import json
import pickle
import pandas as pd
import numpy as np
import datetime
import time
from typing import Any, Optional

class EnhancedJSONEncoder(json.JSONEncoder):
    """
    Custom JSON encoder for pandas, numpy, and datetime types, including NaN/Inf handling.
    """
    def default(self, obj):
        if isinstance(obj, (np.integer, np.int64, np.int32, np.int16, np.int8)):
            return int(obj)
        if isinstance(obj, (np.floating, np.float64, np.float32, np.float16)):
            if np.isnan(obj) or np.isinf(obj):
                return None
            return float(obj)
        if isinstance(obj, (np.ndarray,)):
            return obj.tolist()
        if isinstance(obj, (pd.Timestamp, datetime.datetime, datetime.date)):
            return obj.isoformat()
        if isinstance(obj, (pd.Series,)):
            return obj.tolist()
        if isinstance(obj, (pd.DataFrame,)):
            return obj.to_dict(orient="records")
        return super().default(obj)

def retry_file_operation(func):
    """
    Decorator for retrying file operations with exponential backoff.
    """
    def wrapper(*args, **kwargs):
        max_attempts = 5
        delay = 0.5
        for attempt in range(max_attempts):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                if attempt == max_attempts - 1:
                    raise
                time.sleep(delay)
                delay *= 2
    return wrapper

@retry_file_operation
def to_json(obj: Any, path: Optional[str] = None, **kwargs) -> Optional[str]:
    """
    Serialize an object (DataFrame, numpy array, dict, etc.) to JSON, handling datetimes and NaNs.
    If path is given, writes to file, else returns JSON string.
    """
    json_str = json.dumps(obj, cls=EnhancedJSONEncoder, allow_nan=True, **kwargs)
    if path:
        with open(path, "w") as f:
            f.write(json_str)
        return None
    return json_str

@retry_file_operation
def to_pickle(obj: Any, path: str) -> None:
    """
    Serialize an object to a pickle file with error handling.
    """
    with open(path, "wb") as f:
        pickle.dump(obj, f)

@retry_file_operation
def from_pickle(path: str) -> Any:
    """
    Load an object from a pickle file with error handling.
    """
    with open(path, "rb") as f:
        return pickle.load(f)

@retry_file_operation
def to_csv(df: pd.DataFrame, path: str, **kwargs) -> None:
    """
    Save a DataFrame to CSV with retry logic.
    """
    df.to_csv(path, **kwargs)

@retry_file_operation
def from_csv(path: str, **kwargs) -> pd.DataFrame:
    """
    Load a DataFrame from CSV with type inference and retry logic.
    """
    return pd.read_csv(path, infer_datetime_format=True, **kwargs)

def results_to_json(equity_curve, trade_log, path: Optional[str] = None) -> Optional[str]:
    """
    Convert backtest results (equity curve, trade log) to JSON.
    """
    result = {
        "equity_curve": equity_curve,
        "trade_log": trade_log
    }
    return to_json(result, path=path)
