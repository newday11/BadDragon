"""file_write tool."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def run(arguments: dict[str, Any]) -> dict[str, Any]:
    path_raw = str(arguments.get("path", "") or "").strip()
    if not path_raw:
        return _err("`path` is required.")

    mode = str(arguments.get("mode", "overwrite") or "overwrite").strip().lower()
    if mode not in {"overwrite", "append", "prepend"}:
        return _err("`mode` must be overwrite|append|prepend.")

    content = str(arguments.get("content", "") or "")

    path = Path(path_raw)
    if not path.is_absolute():
        path = Path.cwd() / path
    path = path.resolve()

    try:
        path.parent.mkdir(parents=True, exist_ok=True)
    except Exception as exc:
        return _err(f"Cannot create parent directory: {exc}")

    try:
        if mode == "overwrite":
            path.write_text(content, encoding="utf-8")
        elif mode == "append":
            old = path.read_text(encoding="utf-8", errors="replace") if path.exists() else ""
            path.write_text(old + content, encoding="utf-8")
        else:  # prepend
            old = path.read_text(encoding="utf-8", errors="replace") if path.exists() else ""
            path.write_text(content + old, encoding="utf-8")
    except Exception as exc:
        return _err(f"Write failed: {exc}")

    return {
        "status": "ok",
        "output": {
            "path": str(path),
            "mode": mode,
            "written_chars": len(content),
        },
        "error": "",
        "artifacts": [{"type": "file", "path": str(path)}],
        "meta": {"tool": "file_write"},
    }


def _err(message: str) -> dict[str, Any]:
    return {
        "status": "error",
        "output": {},
        "error": str(message),
        "artifacts": [],
        "meta": {"tool": "file_write"},
    }

