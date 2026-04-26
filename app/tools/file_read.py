"""file_read tool."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def run(arguments: dict[str, Any]) -> dict[str, Any]:
    path_raw = str(arguments.get("path", "") or "").strip()
    if not path_raw:
        return _err("`path` is required.")

    start = _to_int(arguments.get("start", 1), 1)
    count = _to_int(arguments.get("count", 200), 200)
    if start < 1:
        start = 1
    if count < 1:
        count = 1
    if count > 1000:
        count = 1000

    path = Path(path_raw)
    if not path.is_absolute():
        path = Path.cwd() / path
    path = path.resolve()

    if not path.exists():
        return _err(f"File not found: {path}")
    if not path.is_file():
        return _err(f"Not a file: {path}")

    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except Exception as exc:
        return _err(f"Read failed: {exc}")

    idx0 = start - 1
    selected = lines[idx0 : idx0 + count]
    numbered = [f"{i}: {line}" for i, line in enumerate(selected, start=start)]
    content = "\n".join(numbered)

    return {
        "status": "ok",
        "output": {
            "path": str(path),
            "start": start,
            "count": len(selected),
            "content": content,
        },
        "error": "",
        "artifacts": [],
        "meta": {"tool": "file_read"},
    }


def _to_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except Exception:
        return default


def _err(message: str) -> dict[str, Any]:
    return {
        "status": "error",
        "output": {},
        "error": str(message),
        "artifacts": [],
        "meta": {"tool": "file_read"},
    }
