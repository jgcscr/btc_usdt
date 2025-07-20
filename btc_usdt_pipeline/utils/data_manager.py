import pandas as pd
import os

class DataManager:
    """
    Minimal DataManager for loading parquet files.
    """
    def load_data(self, file_path, file_type='parquet', use_cache=True):
        """
        Loads data from a parquet file.
        Args:
            file_path (str): Path to the file.
            file_type (str): Only 'parquet' is supported.
            use_cache (bool): Ignored for now.
        Returns:
            pd.DataFrame or None
        """
        if file_type != 'parquet':
            raise ValueError("Only 'parquet' file_type is supported.")
        if not os.path.exists(file_path):
            return None
        try:
            return pd.read_parquet(file_path)
        except Exception as e:
            print(f"Error loading parquet file: {e}")
            return None
