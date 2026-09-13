from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict) or config.get("schema_version") != 1:
        raise ValueError("configuration must be a mapping with schema_version: 1")
    for key in ("robot", "camera", "target", "collection", "solver", "validation"):
        if key not in config:
            raise ValueError(f"configuration is missing '{key}'")
    return config


def construct(spec: dict[str, Any]) -> Any:
    factory = spec.get("factory")
    if not isinstance(factory, str) or ":" not in factory:
        raise ValueError("factory must use 'module:ClassName' syntax")
    module_name, attribute = factory.split(":", 1)
    try:
        module = importlib.import_module(module_name)
        cls = getattr(module, attribute)
    except (ImportError, AttributeError) as exc:
        raise RuntimeError(f"cannot load factory {factory}: {exc}") from exc
    options = dict(spec.get("options", {}))
    if not isinstance(options, dict):
        raise ValueError("factory options must be a mapping")
    if "streams" in spec:
        if not isinstance(spec["streams"], dict):
            raise ValueError("streams must be a mapping")
        options["streams"] = spec["streams"]
    return cls(**options)
