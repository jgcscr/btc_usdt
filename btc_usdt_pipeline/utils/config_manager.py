# config_manager.py
"""
ConfigManager class for managing configuration objects and updating paths for Colab/Drive compatibility.
"""

class ConfigManager:
    """
    Manages configuration objects or dictionaries, updating all file and directory paths
    to be compatible with Google Colab and Google Drive if needed.
    """
    def __init__(self, config_obj):
        self.config = config_obj

    def update_paths_for_colab(self):
        """
        Update all Path or str file/directory attributes in the config to be Colab/Drive compatible.
        Uses get_gdrive_path from colab_utils if running in Colab.
        """
        from btc_usdt_pipeline.utils.colab_utils import is_colab, get_gdrive_path
        if not is_colab():
            return
        self._update_paths_recursive(self.config)

    def _update_paths_recursive(self, obj, parent_key=None):
        # Recursively update all Path or str attributes that look like paths
        from pathlib import Path
        from btc_usdt_pipeline.utils.colab_utils import get_gdrive_path
        if isinstance(obj, dict):
            for k, v in obj.items():
                obj[k] = self._update_paths_recursive(v, parent_key=k)
            return obj
        elif hasattr(obj, '__dict__'):
            for k, v in vars(obj).items():
                setattr(obj, k, self._update_paths_recursive(v, parent_key=k))
            return obj
        elif isinstance(obj, Path):
            if not str(obj).startswith('/content/drive'):
                return Path(get_gdrive_path(str(obj)))
            return obj
        elif isinstance(obj, str):
            if (('/' in obj or '\\' in obj) and not obj.startswith('/content/drive')):
                return get_gdrive_path(obj)
            return obj
        else:
            return obj
