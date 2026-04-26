"""Task runtime persistence for crash/restart recovery."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class TaskRuntimeStore:
    """Persist active task execution state on disk."""

    def __init__(self, root_dir: Path | None = None) -> None:
        base = (root_dir or Path.cwd()) / "data" / "runtime"
        base.mkdir(parents=True, exist_ok=True)
        self.active_path = base / "active_task.json"
        self.last_path = base / "last_task_result.json"

    def load_active(self) -> dict[str, Any] | None:
        if not self.active_path.exists():
            return None
        try:
            data = json.loads(self.active_path.read_text(encoding="utf-8"))
        except Exception:
            return None
        return data if isinstance(data, dict) else None

    def save_active(self, payload: dict[str, Any]) -> None:
        data = dict(payload)
        data["updated_at_utc"] = datetime.now(timezone.utc).isoformat()
        self.active_path.write_text(
            json.dumps(data, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    def begin_task(self, *, user_goal: str, plan: dict[str, Any]) -> None:
        payload = {
            "status": "running",
            "user_goal": str(user_goal or ""),
            "plan": plan,
            "step_index": 0,
            "execution_log": [],
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
        }
        self.save_active(payload)

    def update_progress(
        self,
        *,
        step_index: int,
        execution_log: list[dict[str, Any]],
        last_step_trace: dict[str, Any] | None = None,
    ) -> None:
        current = self.load_active() or {}
        current["status"] = "running"
        current["step_index"] = max(0, int(step_index))
        current["execution_log"] = execution_log
        if isinstance(last_step_trace, dict):
            current["last_step_trace"] = last_step_trace
        self.save_active(current)

    def finalize(self, *, status: str, summary: dict[str, Any]) -> None:
        result = {
            "status": str(status or "unknown"),
            "summary": summary,
            "finished_at_utc": datetime.now(timezone.utc).isoformat(),
        }
        self.last_path.write_text(
            json.dumps(result, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        self.clear_active()

    def clear_active(self) -> None:
        if self.active_path.exists():
            self.active_path.unlink()
