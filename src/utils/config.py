"""Config helpers for YAML files with environment-variable expansion."""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any, Dict

import yaml


_ENV_VAR_PATTERN = re.compile(r"\$(?:\{(?P<braced>[A-Za-z_][A-Za-z0-9_]*)\}|(?P<bare>[A-Za-z_][A-Za-z0-9_]*))")


def _expand_env_markers(text: str) -> str:
    def replacer(match: re.Match[str]) -> str:
        var_name = match.group("braced") or match.group("bare")
        if not var_name:
            return match.group(0)
        return os.environ.get(var_name, match.group(0))

    return _ENV_VAR_PATTERN.sub(replacer, text)


def _expand_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _expand_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_expand_value(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_expand_value(item) for item in value)
    if isinstance(value, str):
        expanded = _expand_env_markers(value)
        return os.path.expandvars(os.path.expanduser(expanded))
    return value


def load_yaml_config(path: str | Path) -> Dict[str, Any]:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise RuntimeError(f"Expected YAML mapping at {path}, got {type(raw).__name__}.")
    return _expand_value(raw)
