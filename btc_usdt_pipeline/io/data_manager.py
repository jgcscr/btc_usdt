import os
import pandas as pd
from threading import Lock
from btc_usdt_pipeline.utils.logging_config import setup_logging

logger = setup_logging(log_filename='data_manager.log')

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
        if path is not None:
            return str(path)
        return path

    def load_data(self, path, file_type='parquet', use_cache=True, **kwargs):
        path = self._ensure_str_path(path)
        if use_cache and path in self._cache:
            return self._cache[path]
        if file_type == 'parquet':
            df = pd.read_parquet(path, **kwargs)
        elif file_type == 'csv':
            df = pd.read_csv(path, **kwargs)
        else:
            raise ValueError(f"Unsupported file_type: {file_type}")
        if use_cache:
            self._cache[path] = df
        return df

    def save_data(self, df, path, file_type='parquet', **kwargs):
        path = self._ensure_str_path(path)
        if file_type == 'parquet':
            df.to_parquet(path, **kwargs)
        elif file_type == 'csv':
            df.to_csv(path, **kwargs)
        else:
            raise ValueError(f"Unsupported file_type: {file_type}")

    def _download_from_gdrive(self, gdrive_path):
        pass

    def _get_gdrive_upload_path(self, gdrive_path):
        pass
