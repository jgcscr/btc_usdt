import json
from typing import Any, Optional
from pathlib import Path

def to_json(obj: Any, path: str) -> None:
    with open(path, 'w') as f:
        json.dump(obj, f, indent=2)

def save_json(obj: Any, path: str) -> None:
    to_json(obj, path)

def load_json(path: str) -> Optional[Any]:
    try:
        with open(path, 'r') as f:
            return json.load(f)
    except Exception:
        return None
