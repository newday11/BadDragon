"""code_run tool."""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any


def run(arguments: dict[str, Any]) -> dict[str, Any]:
    code_type = str(arguments.get("type", "python") or "python").strip().lower()
    if code_type not in {"python", "bash"}:
        return _err("`type` must be `python` or `bash`.")

    script = str(arguments.get("script", "") or "")
    if not script.strip():
        return _err("`script` is required.")

    timeout = _to_int(arguments.get("timeout", 60), 60)
    if timeout < 1:
        timeout = 1
    if timeout > 300:
        timeout = 300

    cwd_arg = str(arguments.get("cwd", "") or "").strip()
    cwd: str | None = None
    if cwd_arg:
        p = Path(cwd_arg)
        if not p.is_absolute():
            p = Path.cwd() / p
        p = p.resolve()
        if not p.exists() or not p.is_dir():
            return _err(f"`cwd` is invalid: {p}")
        cwd = str(p)

    if code_type == "python":
        cmd = ["python3", "-c", script]
    else:
        cmd = ["bash", "-lc", script]

    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=cwd,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        return {
            "status": "error",
            "output": {
                "stdout": (exc.stdout or "")[:8000],
                "stderr": (exc.stderr or "")[:8000],
                "returncode": None,
            },
            "error": f"timeout after {timeout}s",
            "artifacts": [],
            "meta": {"tool": "code_run", "type": code_type},
        }
    except Exception as exc:
        return _err(f"execution failed: {exc}")

    status = "ok" if proc.returncode == 0 else "error"
    return {
        "status": status,
        "output": {
            "stdout": (proc.stdout or "")[:12000],
            "stderr": (proc.stderr or "")[:12000],
            "returncode": proc.returncode,
        },
        "error": "" if status == "ok" else f"non-zero exit code: {proc.returncode}",
        "artifacts": [],
        "meta": {"tool": "code_run", "type": code_type},
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
        "meta": {"tool": "code_run"},
    }
