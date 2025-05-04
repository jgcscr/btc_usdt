"""
parallel_processing.py

Utilities for parallel processing of DataFrames and tasks using multiprocessing.
Safeguards are included for Colab environments.
"""
import multiprocessing as mp
import numpy as np
import pandas as pd
import os
from btc_usdt_pipeline.utils.colab_utils import is_colab


def parallelize_dataframe(df, func, n_cores=None):
    """
    Split a DataFrame into chunks and process them in parallel using multiprocessing.
    Args:
        df (pd.DataFrame): The DataFrame to process.
        func (callable): Function to apply to each chunk.
        n_cores (int, optional): Number of processes to use. Defaults to os.cpu_count()-1 or 2 in Colab.
    Returns:
        pd.DataFrame: Concatenated result after processing.
    """
    if is_colab():
        # Colab is not always multiprocessing-safe; use 2 cores max
        n_cores = 2 if n_cores is None else min(n_cores, 2)
    else:
        n_cores = n_cores or max(1, os.cpu_count() - 1)
    df_split = np.array_split(df, n_cores)
    with mp.get_context('fork').Pool(n_cores) as pool:
        results = pool.map(func, df_split)
    return pd.concat(results)


class ParallelTaskExecutor:
    """
    Manages a queue of tasks to be executed in parallel with resource management.
    """
    def __init__(self, n_workers=None):
        if is_colab():
            self.n_workers = 2 if n_workers is None else min(n_workers, 2)
        else:
            self.n_workers = n_workers or max(1, os.cpu_count() - 1)
        self.pool = mp.get_context('fork').Pool(self.n_workers)
        self.tasks = []

    def add_task(self, func, *args, **kwargs):
        self.tasks.append((func, args, kwargs))

    def run_all(self):
        results = []
        for func, args, kwargs in self.tasks:
            results.append(self.pool.apply_async(func, args, kwargs))
        self.pool.close()
        self.pool.join()
        return [r.get() for r in results]

    def __del__(self):
        self.pool.terminate()

# Example usage for feature calculation:
# from btc_usdt_pipeline.utils.parallel_processing import parallelize_dataframe
# def calc_features_chunk(chunk):
#     # ... feature calculation logic ...
#     return chunk
# df = parallelize_dataframe(df, calc_features_chunk)
