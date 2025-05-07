"""
parallel_processing.py

Unified utilities for parallel processing of DataFrames and general tasks using multiprocessing.
Safeguards are included for Colab environments.
"""
import multiprocessing as mp
import numpy as np
import pandas as pd
import os
from typing import Callable, List, Any, Optional, Tuple, Dict
from btc_usdt_pipeline.utils.colab_utils import is_colab

def parallel_process(
    tasks: List[Tuple[Callable, tuple, dict]],
    n_workers: Optional[int] = None
) -> List[Any]:
    """
    Run a list of (function, args, kwargs) in parallel and return results.
    Args:
        tasks: List of (func, args, kwargs) tuples.
        n_workers: Number of worker processes to use.
    Returns:
        List of results from each task.
    """
    if n_workers is None:
        n_workers = 2 if is_colab() else max(1, (os.cpu_count() or 2) - 1)
    ctx = mp.get_context('fork')
    with ctx.Pool(n_workers) as pool:
        results = [pool.apply_async(func, args, kwargs) for func, args, kwargs in tasks]
        return [r.get() for r in results]

def parallelize_dataframe(
    df: pd.DataFrame,
    func: Callable[[pd.DataFrame], pd.DataFrame],
    n_cores: Optional[int] = None
) -> pd.DataFrame:
    """
    Split a DataFrame into chunks and process them in parallel using multiprocessing.
    Args:
        df: The DataFrame to process.
        func: Function to apply to each chunk.
        n_cores: Number of processes to use. Defaults to os.cpu_count()-1 or 2 in Colab.
    Returns:
        pd.DataFrame: Concatenated result after processing.
    """
    if n_cores is None:
        n_cores = 2 if is_colab() else max(1, (os.cpu_count() or 2) - 1)
    df_split = np.array_split(df, n_cores)
    ctx = mp.get_context('fork')
    with ctx.Pool(n_cores) as pool:
        results = pool.map(func, df_split)
    return pd.concat(results)

# Example usage for DataFrame:
# from btc_usdt_pipeline.utils.parallel_processing import parallelize_dataframe
# def calc_features_chunk(chunk):
#     # ... feature calculation logic ...
#     return chunk
# df = parallelize_dataframe(df, calc_features_chunk)

# Example usage for general tasks:
# from btc_usdt_pipeline.utils.parallel_processing import parallel_process
# tasks = [(func1, (arg1,), {}), (func2, (arg2, arg3), {'kwarg': val})]
# results = parallel_process(tasks)
