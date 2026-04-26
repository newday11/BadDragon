"""ask_user tool."""

from __future__ import annotations

from typing import Any


def run(arguments: dict[str, Any]) -> dict[str, Any]:
    question = str(arguments.get("question", "") or "").strip()
    if not question:
        return _err("`question` is required.")

    candidates_raw = arguments.get("candidates", [])
    candidates: list[str] = []
    if isinstance(candidates_raw, list):
        candidates = [str(x).strip() for x in candidates_raw if str(x).strip()]

    return {
        "status": "needs_user",
        "output": {
            "question": question,
            "candidates": candidates,
            "note": "User decision required.",
        },
        "error": "",
        "artifacts": [],
        "meta": {"tool": "ask_user"},
    }


def _err(message: str) -> dict[str, Any]:
    return {
        "status": "error",
        "output": {},
        "error": str(message),
        "artifacts": [],
        "meta": {"tool": "ask_user"},
    }
