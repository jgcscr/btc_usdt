# config_manager.py
"""
ConfigManager class for managing configuration objects and updating paths for Colab/Drive compatibility.
"""

import json
import yaml
import os
import copy
from typing import Dict, Any, Optional, List
from datetime import datetime

class ConfigManager:
    """
    Manages backtest and optimization configurations with validation, persistence, and versioning.
    """
    CONFIG_DIR = os.path.join(os.path.dirname(__file__), '../../configs')
    HISTORY_FILE = os.path.join(CONFIG_DIR, 'config_history.json')
    DEFAULT_CONFIG_NAME = 'default'
    ALLOWED_RANGES = {
        'sl_multiplier': (0.01, 10),
        'tp_multiplier': (0.01, 20),
        'risk_fraction': (0.0, 1.0),
        'commission_rate': (0.0, 0.1),
        'slippage_points': (0.0, 1000),
        'n_trials': (1, 10000),
        'n_jobs': (1, 128),
    }

    def __init__(self):
        os.makedirs(self.CONFIG_DIR, exist_ok=True)
        self.configs: Dict[str, Dict[str, Any]] = {}
        self.history: List[Dict[str, Any]] = self._load_history()
        self.default_config = self._get_builtin_default()
        self.configs[self.DEFAULT_CONFIG_NAME] = copy.deepcopy(self.default_config)

    def _get_builtin_default(self) -> Dict[str, Any]:
        return {
            'sl_multiplier': 1.5,
            'tp_multiplier': 3.0,
            'risk_fraction': 0.01,
            'commission_rate': 0.00075,
            'slippage_points': 2.0,
            'n_trials': 100,
            'n_jobs': 4,
        }

    def validate(self, config: Dict[str, Any]) -> None:
        for k, v in config.items():
            if k in self.ALLOWED_RANGES:
                minv, maxv = self.ALLOWED_RANGES[k]
                if not (minv <= v <= maxv):
                    raise ValueError(f"Parameter '{k}'={v} out of allowed range {minv}-{maxv}")

    def create_config(self, name: str, **params) -> Dict[str, Any]:
        config = copy.deepcopy(self.default_config)
        config.update(params)
        self.validate(config)
        self.configs[name] = config
        self._save_config_file(name, config)
        self._add_to_history(name, config)
        return config

    def _save_config_file(self, name: str, config: Dict[str, Any]) -> None:
        path = os.path.join(self.CONFIG_DIR, f"{name}.json")
        with open(path, 'w') as f:
            json.dump(config, f, indent=2)

    def load_config(self, name_or_path: str) -> Dict[str, Any]:
        if os.path.isfile(name_or_path):
            with open(name_or_path, 'r') as f:
                config = json.load(f)
        else:
            path = os.path.join(self.CONFIG_DIR, f"{name_or_path}.json")
            with open(path, 'r') as f:
                config = json.load(f)
        self.validate(config)
        return config

    def compare_configs(self, config1: Dict[str, Any], config2: Dict[str, Any]) -> Dict[str, Any]:
        diff = {}
        keys = set(config1.keys()).union(config2.keys())
        for k in keys:
            v1 = config1.get(k)
            v2 = config2.get(k)
            if v1 != v2:
                diff[k] = {'config1': v1, 'config2': v2}
        return diff

    def get_default_config(self) -> Dict[str, Any]:
        return copy.deepcopy(self.default_config)

    def list_configs(self) -> List[str]:
        return [f[:-5] for f in os.listdir(self.CONFIG_DIR) if f.endswith('.json')]

    def _add_to_history(self, name: str, config: Dict[str, Any]) -> None:
        entry = {
            'timestamp': datetime.utcnow().isoformat(),
            'name': name,
            'config': copy.deepcopy(config)
        }
        self.history.append(entry)
        self._save_history()

    def _save_history(self):
        with open(self.HISTORY_FILE, 'w') as f:
            json.dump(self.history, f, indent=2)

    def _load_history(self) -> List[Dict[str, Any]]:
        if os.path.exists(self.HISTORY_FILE):
            with open(self.HISTORY_FILE, 'r') as f:
                return json.load(f)
        return []

    def get_history(self) -> List[Dict[str, Any]]:
        return self.history

# --- Simple CLI ---
if __name__ == '__main__':
    import argparse
    cm = ConfigManager()
    parser = argparse.ArgumentParser(description="ConfigManager CLI")
    parser.add_argument('--list', action='store_true', help='List available configurations')
    parser.add_argument('--create', nargs='+', help='Create a new config: name key1=val1 key2=val2 ...')
    parser.add_argument('--load', type=str, help='Load a config by name or path')
    parser.add_argument('--compare', nargs=2, help='Compare two configs by name or path')
    parser.add_argument('--default', action='store_true', help='Show default config')
    parser.add_argument('--history', action='store_true', help='Show config history')
    args = parser.parse_args()

    if args.list:
        print("Available configs:", cm.list_configs())
    if args.create:
        name = args.create[0]
        param_pairs = args.create[1:]
        params = {}
        for pair in param_pairs:
            k, v = pair.split('=')
            try:
                v = float(v)
            except Exception:
                pass
            params[k] = v
        config = cm.create_config(name, **params)
        print(f"Created config '{name}':", config)
    if args.load:
        config = cm.load_config(args.load)
        print(f"Loaded config from '{args.load}':", config)
    if args.compare:
        c1 = cm.load_config(args.compare[0])
        c2 = cm.load_config(args.compare[1])
        diff = cm.compare_configs(c1, c2)
        print("Config differences:", diff)
    if args.default:
        print("Default config:", cm.get_default_config())
    if args.history:
        print("Config history:")
        for entry in cm.get_history():
            print(entry)
