"""Shared utilities for DAMP-ES.

This module keeps imports lightweight so helpers that do not require heavy
dependencies (for example torch) can still be imported in minimal environments.
"""

from .config import apply_overrides, load_yaml_config, parse_overrides
from .io import ensure_dir, list_images, read_lines, write_lines

__all__ = [
    "apply_overrides",
    "load_yaml_config",
    "parse_overrides",
    "ensure_dir",
    "list_images",
    "read_lines",
    "write_lines",
    "set_global_seed",
]


def __getattr__(name: str):
    if name == "set_global_seed":
        from .seed import set_global_seed

        return set_global_seed
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
