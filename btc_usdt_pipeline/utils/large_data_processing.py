"""
large_data_processing.py

Utilities for processing datasets larger than memory using Dask.
"""
import dask.dataframe as dd
import psutil
from btc_usdt_pipeline.monitoring.memory import monitor_memory
from btc_usdt_pipeline.monitoring.memory import memory_safe
from btc_usdt_pipeline.utils.logging_config import setup_logging

logger = setup_logging(log_filename='large_data_processing.log')

@memory_safe(min_free_percent=10)
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
        logger.info(f"Loaded Dask DataFrame with {ddf.npartitions} partitions.")
        # Apply the processing function to each partition
        processed_ddf = ddf.map_partitions(process_func)
        check_memory_usage()
        return processed_ddf
    except Exception as e:
        logger.error(f"Error processing large dataframe: {e}")
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
            if file_type == 'parquet':
                ddf.to_parquet(output_path)
            else:
                ddf.to_csv(output_path, single_file=True)
            logger.info(f"Processed data saved to {output_path}")
        except Exception as e:
            logger.error(f"Error saving processed data: {e}")
