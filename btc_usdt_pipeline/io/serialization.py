
"""Serialization utilities for the BTC/USDT pipeline."""
import json
from typing import Any, Optional

def to_json(obj: Any, path: str) -> None:
    """Save an object as JSON to the specified path."""
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(obj, f, indent=2)

def save_json(obj: Any, path: str) -> None:
    """Alias for to_json."""
    to_json(obj, path)

def load_json(path: str) -> Optional[Any]:
    """Load an object from a JSON file at the specified path."""
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception:
        return None
