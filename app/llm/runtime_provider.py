"""Runtime provider resolution (Hermes-style, simplified)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from app.infra.config import load_config


@dataclass(frozen=True)
class RuntimeProvider:
    provider: str
    api_mode: str
    model: str
    base_url: str
    api_key: str
    temperature: float
    max_tokens: int
    source: str


def _model_cfg() -> dict[str, Any]:
    cfg = load_config()
    model_cfg = cfg.get("model", {})
    if not isinstance(model_cfg, dict):
        return {}
    return dict(model_cfg)


def _normalize_provider(raw: str) -> str:
    p = (raw or "").strip().lower()
    if p in ("", "auto"):
        return "auto"
    if p in ("openai", "openai-compatible", "custom"):
        return "custom"
    if p in ("zai", "glm", "zhipu"):
        return "zai"
    if p in ("anthropic",):
        return "anthropic"
    return p


def _detect_api_mode(base_url: str, requested: str, configured: str) -> str:
    if requested in ("anthropic",) or configured in ("anthropic",):
        return "anthropic_messages"
    lower = (base_url or "").strip().lower()
    if lower.endswith("/anthropic"):
        return "anthropic_messages"
    return "chat_completions"


def resolve_runtime_provider(
    *,
    requested: str | None = None,
    explicit_api_key: str | None = None,
    explicit_base_url: str | None = None,
) -> RuntimeProvider:
    cfg = _model_cfg()

    requested_provider = _normalize_provider(requested or str(cfg.get("provider", "auto")))
    configured_provider = _normalize_provider(str(cfg.get("provider", "auto")))

    provider = requested_provider if requested_provider != "auto" else configured_provider
    if provider == "auto":
        provider = "custom"

    base_url = (
        (explicit_base_url or "").strip()
        or str(cfg.get("base_url", "")).strip()
        or "https://modelservice.jdcloud.com/coding/openai/v1"
    ).rstrip("/")
    api_key = (explicit_api_key or "").strip() or str(cfg.get("api_key", "")).strip()

    api_mode = (
        str(cfg.get("api_mode", "")).strip().lower()
        or _detect_api_mode(base_url, requested_provider, configured_provider)
    )
    if api_mode not in ("chat_completions", "anthropic_messages", "codex_responses"):
        api_mode = "chat_completions"

    model = str(cfg.get("default", "GLM-5")).strip()
    temperature = float(cfg.get("temperature", 0.2))
    max_tokens = int(cfg.get("max_tokens", 2048))

    if not api_key:
        raise ValueError("No API key found. Set model.api_key in config.json")
    if not base_url:
        raise ValueError("No base_url found. Set model.base_url in config.json")

    return RuntimeProvider(
        provider=provider,
        api_mode=api_mode,
        model=model,
        base_url=base_url,
        api_key=api_key,
        temperature=temperature,
        max_tokens=max_tokens,
        source="config+env",
    )
