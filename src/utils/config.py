from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Iterable, List

import yaml


def load_config(path: str | Path) -> Dict[str, Any]:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def save_config(cfg: Dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)


def _coerce_value(value: str) -> Any:
    lowered = value.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    try:
        if "." in value:
            return float(value)
        return int(value)
    except ValueError:
        return value


def _set_nested(cfg: Dict[str, Any], keys: List[str], value: Any) -> None:
    cur = cfg
    for key in keys[:-1]:
        if key not in cur or not isinstance(cur[key], dict):
            cur[key] = {}
        cur = cur[key]
    cur[keys[-1]] = value


def merge_overrides(cfg: Dict[str, Any], overrides: Iterable[str] | None) -> Dict[str, Any]:
    merged = deepcopy(cfg)
    if not overrides:
        return merged
    for item in overrides:
        if "=" not in item:
            raise ValueError(f"Invalid override {item!r}; expected key=value")
        key, value = item.split("=", 1)
        _set_nested(merged, key.split("."), _coerce_value(value))
    return merged
