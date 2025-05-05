import os
import pandas as pd
from threading import Lock

class DataManager:
    _instance = None
    _lock = Lock()

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            with cls._lock:
                if not cls._instance:
                    cls._instance = super(DataManager, cls).__new__(cls)
                    cls._instance._cache = {}
        return cls._instance

    def _ensure_str_path(self, path):
        """Convert Path objects to strings to avoid compatibility issues."""
        if path is not None:
            return str(path)
        return path

    def load_data(self, path, file_type='parquet', use_cache=True, **kwargs):
        """
        Load data from a file path (local or Google Drive). Supports caching.
        Args:
            path (str): File path or Google Drive path.
            file_type (str): 'parquet', 'csv', or 'json'.
            use_cache (bool): Whether to use cached data.
            **kwargs: Additional arguments for pandas read functions.
        Returns:
            pd.DataFrame: Loaded data.
        """
        path = self._ensure_str_path(path)
        cache_key = (path, file_type)
        if use_cache and cache_key in self._cache:
            return self._cache[cache_key]
        try:
            if path.startswith('gdrive://'):
                local_path = self._download_from_gdrive(path)
            else:
                local_path = path
            if file_type == 'parquet':
                df = pd.read_parquet(local_path, **kwargs)
            elif file_type == 'csv':
                df = pd.read_csv(local_path, **kwargs)
            elif file_type == 'json':
                df = pd.read_json(local_path, **kwargs)
            else:
                raise ValueError(f"Unsupported file_type: {file_type}")
            if use_cache:
                self._cache[cache_key] = df
            # Optimize memory usage after loading
            from btc_usdt_pipeline.utils.data_processing import optimize_memory_usage
            df = optimize_memory_usage(df)
            return df
        except Exception as e:
            print(f"[DataManager] Error loading {path}: {e}")
            return None

    def save_data(self, df, path, file_type='parquet', **kwargs):
        """
        Save DataFrame to a file path (local or Google Drive).
        Args:
            df (pd.DataFrame): Data to save.
            path (str): File path or Google Drive path.
            file_type (str): 'parquet', 'csv', or 'json'.
            **kwargs: Additional arguments for pandas to_* functions.
        """
        path = self._ensure_str_path(path)
        try:
            if path.startswith('gdrive://'):
                local_path = self._get_gdrive_upload_path(path)
            else:
                local_path = path
            if file_type == 'parquet':
                df.to_parquet(local_path, **kwargs)
            elif file_type == 'csv':
                df.to_csv(local_path, **kwargs)
            elif file_type == 'json':
                df.to_json(local_path, **kwargs)
            else:
                raise ValueError(f"Unsupported file_type: {file_type}")
        except Exception as e:
            print(f"[DataManager] Error saving to {path}: {e}")

    def _download_from_gdrive(self, gdrive_path):
        # Placeholder for Google Drive download logic
        # Should return a local file path after downloading
        raise NotImplementedError("Google Drive download not implemented.")

    def _get_gdrive_upload_path(self, gdrive_path):
        # Placeholder for Google Drive upload logic
        # Should return a local file path for saving before upload
        raise NotImplementedError("Google Drive upload not implemented.")
