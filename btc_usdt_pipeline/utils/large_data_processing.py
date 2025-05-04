"""
large_data_processing.py

Utilities for processing datasets larger than memory using Dask.
"""
import dask.dataframe as dd
import psutil
from functools import wraps
from btc_usdt_pipeline.utils.colab_utils import check_memory_usage

def memory_safe_dask(min_free_percent=10):
    """
    Decorator for Dask operations to check memory before execution.
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            free_percent = 100 - psutil.virtual_memory().percent
            if free_percent < min_free_percent:
                print(f"[LargeDataProcessing] ABORT: Not enough free memory ({free_percent:.2f}% < {min_free_percent}%)")
                return None
            return func(*args, **kwargs)
        return wrapper
    return decorator

@memory_safe_dask(min_free_percent=10)
def process_large_dataframe(file_path, process_func, file_type='parquet', **kwargs):
    """
    Loads a large dataset using Dask and applies a processing function chunk-wise.
    Args:
        file_path (str or Path): Path to the large data file.
        process_func (callable): Function to apply to each partition (should accept a DataFrame and return a DataFrame).
        file_type (str): 'parquet' or 'csv'.
        **kwargs: Additional arguments for Dask read functions.
    Returns:
        dask.dataframe.DataFrame: The processed Dask DataFrame (not computed).
    Example:
        def my_processing(df):
            # Feature computation or cleaning
            return df
        result = process_large_dataframe('data/large.parquet', my_processing)
    """
    try:
        if file_type == 'parquet':
            ddf = dd.read_parquet(file_path, **kwargs)
        elif file_type == 'csv':
            ddf = dd.read_csv(file_path, **kwargs)
        else:
            raise ValueError(f"Unsupported file_type: {file_type}")
        print(f"[LargeDataProcessing] Loaded Dask DataFrame with {ddf.npartitions} partitions.")
        # Apply the processing function to each partition
        processed_ddf = ddf.map_partitions(process_func)
        check_memory_usage()
        return processed_ddf
    except Exception as e:
        print(f"[LargeDataProcessing] Error processing large dataframe: {e}")
        return None

# Example chunk-wise feature computation for large historical datasets
def compute_features_on_large_data(file_path, feature_func, output_path, file_type='parquet', **kwargs):
    """
    Loads a large dataset, applies a feature computation function chunk-wise, and saves the result.
    Args:
        file_path (str or Path): Input data file.
        feature_func (callable): Function to compute features on each partition.
        output_path (str or Path): Where to save the processed data.
        file_type (str): 'parquet' or 'csv'.
        **kwargs: Additional arguments for Dask read/write functions.
    """
    ddf = process_large_dataframe(file_path, feature_func, file_type=file_type, **kwargs)
    if ddf is not None:
        try:
            ddf.to_parquet(output_path) if file_type == 'parquet' else ddf.to_csv(output_path, single_file=True)
            print(f"[LargeDataProcessing] Processed data saved to {output_path}")
        except Exception as e:
            print(f"[LargeDataProcessing] Error saving processed data: {e}")
