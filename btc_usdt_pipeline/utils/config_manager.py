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
    def __init__(self, env: Optional[str] = None):
        self.env = env or os.environ.get('BTCUSDT_ENV', 'dev')
        self.configs: Dict[str, Dict[str, Any]] = {}
        self._load_all_configs()
        self._apply_env_overrides()
        self._apply_env_var_overrides()

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
                        self.configs[section][key] = v

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
