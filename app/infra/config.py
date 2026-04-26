"""Configuration loading for BadDragon (JSON-only)."""

from __future__ import annotations

import os
import json
from functools import lru_cache
from pathlib import Path
from typing import Any


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def config_path() -> Path:
    env_path = os.getenv("BADDRAGON_CONFIG_PATH", "").strip()
    if env_path:
        p = Path(env_path).expanduser()
        if p.exists():
            return p

    candidates = [
        Path.cwd() / "config.json",
        Path.home() / ".baddragon" / "config.json",
        project_root() / "config.json",
    ]
    for path in candidates:
        if path.exists():
            return path
    return project_root() / "config.json"


def _deep_merge(base: dict[str, Any], patch: dict[str, Any]) -> dict[str, Any]:
    result = dict(base)
    for k, v in patch.items():
        if isinstance(v, dict) and isinstance(result.get(k), dict):
            result[k] = _deep_merge(result[k], v)
        else:
            result[k] = v
    return result


def default_config() -> dict[str, Any]:
    return {
        "model": {
            "default": "GLM-5",
            "provider": "custom",
            "base_url": "https://modelservice.jdcloud.com/coding/openai/v1",
            "api_key": "",
            "api_mode": "chat_completions",
            "temperature": 0.2,
            "max_tokens": 2048,
        },
        "fallback_providers": [],
    }


def _load_raw_config(path: Path) -> dict[str, Any]:
    suffix = path.suffix.lower()
    text = path.read_text(encoding="utf-8")

    if suffix == ".json":
        raw = json.loads(text or "{}")
        if not isinstance(raw, dict):
            raise ValueError("config.json must be a JSON object")
        return raw

    raise ValueError(
        f"Unsupported config format: {path.name}. "
        "Only config.json is supported."
    )


@lru_cache(maxsize=1)
def load_config() -> dict[str, Any]:
    cfg = default_config()
    path = config_path()
    if path.exists():
        raw = _load_raw_config(path)
        cfg = _deep_merge(cfg, raw)
    return cfg
