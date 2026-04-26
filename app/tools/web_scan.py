"""web_scan tool."""

from __future__ import annotations

from typing import Any

from .web_state import SESSION


def run(arguments: dict[str, Any]) -> dict[str, Any]:
    tabs_only = bool(arguments.get("tabs_only", False))
    text_only = bool(arguments.get("text_only", False))
    switch_tab_id = str(arguments.get("switch_tab_id", "") or "").strip() or None
    try:
        out = SESSION.scan(
            tabs_only=tabs_only,
            switch_tab_id=switch_tab_id,
            text_only=text_only,
        )
        return {
            "status": "ok",
            "output": out,
            "error": "",
            "artifacts": [],
            "meta": {"tool": "web_scan"},
        }
    except Exception as exc:
        return {
            "status": "error",
            "output": {},
            "error": f"web_scan failed: {exc}",
            "artifacts": [],
            "meta": {"tool": "web_scan"},
        }

