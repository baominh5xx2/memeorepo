from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping


try:
    import yaml
except ImportError as exc:  # pragma: no cover - environment dependent
    raise ImportError(
        "PyYAML is required for loading config files. Install it with: pip install pyyaml"
    ) from exc


ConfigDict = Dict[str, Any]


@dataclass
class ParsedOverride:
    key: str
    value: Any


def load_yaml_config(path: str | Path) -> ConfigDict:
    cfg_path = Path(path)
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config file not found: {cfg_path}")

    with cfg_path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError(f"Config root must be a mapping: {cfg_path}")
    return data


def parse_overrides(raw_overrides: Iterable[str]) -> List[ParsedOverride]:
    parsed: List[ParsedOverride] = []
    for item in raw_overrides:
        if "=" not in item:
            raise ValueError(f"Override must be key=value, got: {item}")

        key, raw_value = item.split("=", 1)
        key = key.strip()
        if not key:
            raise ValueError(f"Override key cannot be empty: {item}")

        parsed.append(ParsedOverride(key=key, value=_coerce_value(raw_value.strip())))
    return parsed


def apply_overrides(cfg: ConfigDict, overrides: Iterable[ParsedOverride]) -> ConfigDict:
    out = _deep_copy(cfg)
    for ov in overrides:
        _set_by_dotted_key(out, ov.key, ov.value)
    return out


def _coerce_value(value: str) -> Any:
    lowered = value.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    if lowered in {"none", "null"}:
        return None

    try:
        if "." in value:
            return float(value)
        return int(value)
    except ValueError:
        return value


def _set_by_dotted_key(target: MutableMapping[str, Any], dotted_key: str, value: Any) -> None:
    keys = dotted_key.split(".")
    cursor: MutableMapping[str, Any] = target
    for key in keys[:-1]:
        if key not in cursor or not isinstance(cursor[key], dict):
            cursor[key] = {}
        cursor = cursor[key]
    cursor[keys[-1]] = value


def _deep_copy(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _deep_copy(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_deep_copy(v) for v in obj]
    return obj
