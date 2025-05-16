# config_manager.py
"""
ConfigManager class for managing configuration objects and updating paths for Colab/Drive compatibility.
"""

import os
import yaml
from typing import Dict, Any, Optional
from pathlib import Path
import logging
import copy

CONFIG_DIR = Path(__file__).parent.parent.parent / 'configs'

class ConfigManager:
    def __init__(self, env: Optional[str] = None, gcs_bucket_name: Optional[str] = None):
        self.env = env or os.environ.get('BTCUSDT_ENV', 'dev')
        self.gcs_bucket_name = gcs_bucket_name or os.environ.get('GCS_BUCKET_NAME')
        self.configs: Dict[str, Dict[str, Any]] = {}
        self._load_all_configs()
        self._apply_env_overrides()
        self._apply_env_var_overrides()
        if self.gcs_bucket_name:
            self._convert_paths_to_gcs()

    def _load_yaml(self, path: Path) -> Dict[str, Any]:
        if not path.exists():
            return {}
        with open(path, 'r') as f:
            return yaml.safe_load(f) or {}

    def _load_all_configs(self):
        for fname in os.listdir(CONFIG_DIR):
            if fname.endswith('.yaml'):
                key = fname.replace('_config.yaml', '').replace('.yaml', '')
                self.configs[key] = self._load_yaml(CONFIG_DIR / fname)

    def _apply_env_overrides(self):
        # Support configs/app_config.dev.yaml, etc.
        env_file = CONFIG_DIR / f'app_config.{self.env}.yaml'
        if env_file.exists():
            env_config = self._load_yaml(env_file)
            self.configs['app'].update(env_config)

    def _apply_env_var_overrides(self):
        # Allow overrides like BTCUSDT__MODEL__N_ESTIMATORS=200
        for k, v in os.environ.items():
            if k.startswith('BTCUSDT__'):
                parts = k[len('BTCUSDT__'):].lower().split('__')
                if len(parts) >= 2:
                    section, key = parts[0], '_'.join(parts[1:])
                    if section in self.configs:
                        # Try to infer type for env vars
                        original_value = self.configs[section].get(key)
                        if isinstance(original_value, bool):
                            v = v.lower() in ['true', '1', 'yes']
                        elif isinstance(original_value, int):
                            try:
                                v = int(v)
                            except ValueError:
                                logging.warning(f"Could not convert env var {k}={v} to int for {section}.{key}")
                        elif isinstance(original_value, float):
                            try:
                                v = float(v)
                            except ValueError:
                                logging.warning(f"Could not convert env var {k}={v} to float for {section}.{key}")
                        self.configs[section][key] = v

    def _convert_paths_to_gcs(self):
        if not self.gcs_bucket_name:
            return

        path_keys = ['data_dir', 'models_dir', 'results_dir', 'logs_dir',
                       'raw_data_path', 'enriched_data_path',
                       'auto_optimize_results_path', 'indicator_optimize_results_path',
                       'model_predictions_path']

        for section_name, section_config in self.configs.items():
            for key, value in section_config.items():
                if key in path_keys and isinstance(value, str):
                    # Ensure we don't add multiple gs:// prefixes if already a GCS path
                    if value.startswith('gs://'):
                        continue
                    # Make sure it's a relative path before prepending bucket
                    # Absolute local paths should not be converted directly
                    if not Path(value).is_absolute():
                        self.configs[section_name][key] = f"gs://{self.gcs_bucket_name}/{value.lstrip('/')}"
                    else:
                        logging.warning(f"Absolute path {value} for {key} in section {section_name} will not be converted to GCS path.")
                # Handle lists of paths if necessary (e.g. model_files in a model config)
                # This is an example, adjust if you have such structures
                if isinstance(value, list) and key.endswith('_files'):
                    new_list = []
                    for item in value:
                        if isinstance(item, str) and not Path(item).is_absolute() and not item.startswith('gs://'):
                            new_list.append(f"gs://{self.gcs_bucket_name}/{item.lstrip('/')}")
                        else:
                            new_list.append(item)
                    self.configs[section_name][key] = new_list

    def get_gcs_path(self, relative_path: str) -> str:
        if self.gcs_bucket_name and not relative_path.startswith('gs://'):
            return f"gs://{self.gcs_bucket_name}/{relative_path.lstrip('/')}"
        return relative_path

    def get(self, key: str, default: Any = None) -> Any:
        # Support dot notation: model.random_forest.n_estimators
        parts = key.split('.')
        cfg = self.configs
        for part in parts:
            if isinstance(cfg, dict) and part in cfg:
                cfg = cfg[part]
            else:
                return default
        return cfg

    def as_dict(self) -> Dict[str, Any]:
        return copy.deepcopy(self.configs)

# Singleton instance
config_manager = ConfigManager()
