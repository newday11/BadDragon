"""file_patch tool."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def run(arguments: dict[str, Any]) -> dict[str, Any]:
    path_raw = str(arguments.get("path", "") or "").strip()
    old_content = str(arguments.get("old_content", "") or "")
    new_content = str(arguments.get("new_content", "") or "")

    if not path_raw:
        return _err("`path` is required.")
    if old_content == "":
        return _err("`old_content` is required and cannot be empty.")

    path = Path(path_raw)
    if not path.is_absolute():
        path = Path.cwd() / path
    path = path.resolve()

    if not path.exists() or not path.is_file():
        return _err(f"File not found: {path}")

    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except Exception as exc:
        return _err(f"Read failed: {exc}")

    count = text.count(old_content)
    if count == 0:
        return _err("`old_content` not found in file.")
    if count > 1:
        return _err("`old_content` is not unique in file.")

    patched = text.replace(old_content, new_content, 1)
    try:
        path.write_text(patched, encoding="utf-8")
    except Exception as exc:
        return _err(f"Write failed: {exc}")

    return {
        "status": "ok",
        "output": {
            "path": str(path),
            "replacements": 1,
        },
        "error": "",
        "artifacts": [{"type": "file", "path": str(path)}],
        "meta": {"tool": "file_patch"},
    }


def _err(message: str) -> dict[str, Any]:
    return {
        "status": "error",
        "output": {},
        "error": str(message),
        "artifacts": [],
        "meta": {"tool": "file_patch"},
    }

