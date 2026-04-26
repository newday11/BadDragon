"""Memory store for BadDragon.

Storage layout:
- Global owner memory folder (cross-project, persistent):
  ~/.baddragon/owner_memory/owner_profile_memory.json
  ~/.baddragon/owner_memory/owner_global_constraints.json
- Project memory folder (per project, reset by project):
  <project>/data/project_memory/project_long_term_memory.json
  <project>/data/project_memory/project_short_term_memory.json
  <project>/data/project_memory/context_memory.sqlite3

Rules implemented:
- Context memory is stored in SQLite and only the most recent 4 entries
  are injected per turn.
"""

from __future__ import annotations

import json
import sqlite3
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


BLOCK_KEYS = (
    "owner_profile_memory",
    "owner_global_constraints",
    "project_long_term_memory",
    "project_short_term_memory",
)

OWNER_MEMORY_FAST_INTERVAL_THRESHOLD_BYTES = 3072
OWNER_MEMORY_BLOCK_LIMIT_BYTES = 4096
OWNER_MEMORY_COMPACT_TARGET_BYTES = 2048
PROJECT_MEMORY_BLOCK_LIMIT_BYTES = 4096
PROJECT_MEMORY_COMPACT_TARGET_BYTES = 2048
PROJECT_MEMORY_INTERVAL = 2


@dataclass
class MemoryPaths:
    """Disk paths for memory persistence."""

    owner_profile_json: Path
    owner_constraints_json: Path
    owner_legacy_json: Path
    project_long_json: Path
    project_short_json: Path
    project_legacy_json: Path
    context_sqlite: Path


def _utf8_size(text: str) -> int:
    return len(text.encode("utf-8"))


def _shorten_text(text: str, max_chars: int = 140) -> str:
    compact = " ".join(str(text or "").split())
    if len(compact) <= max_chars:
        return compact
    return compact[: max_chars - 1] + "…"


class MemoryStore:
    """Persistent memory manager."""

    def __init__(self, root_dir: Path | None = None) -> None:
        project_base = (root_dir or Path.cwd()) / "data" / "project_memory"
        project_base.mkdir(parents=True, exist_ok=True)
        owner_base = Path.home() / ".baddragon" / "owner_memory"
        owner_base.mkdir(parents=True, exist_ok=True)
        self.paths = MemoryPaths(
            owner_profile_json=owner_base / "owner_profile_memory.json",
            owner_constraints_json=owner_base / "owner_global_constraints.json",
            owner_legacy_json=owner_base / "owner_memory.json",
            project_long_json=project_base / "project_long_term_memory.json",
            project_short_json=project_base / "project_short_term_memory.json",
            project_legacy_json=project_base / "project_memory.json",
            context_sqlite=project_base / "context_memory.sqlite3",
        )
        # In-memory counters only (reset on process restart).
        self.user_input_count = 0
        self.last_owner_memory_refresh_at = 0
        self.last_project_memory_refresh_at = 0
        self.last_project_short_memory_refresh_at = 0
        self.last_project_long_memory_refresh_at = 0
        self.last_owner_refresh_triggered = False
        self.last_project_refresh_triggered = False
        self.last_owner_sent_payload: dict[str, Any] | None = None
        self.last_owner_returned_payload: dict[str, Any] | None = None
        self.last_project_sent_payload: dict[str, Any] | None = None
        self.last_project_returned_payload: dict[str, Any] | None = None
        self._ensure_owner_files()
        self._ensure_project_files()
        self._ensure_context_db()

    def _ensure_owner_files(self) -> None:
        # Migrate from old single-file layout when present.
        if self.paths.owner_legacy_json.exists():
            try:
                legacy = json.loads(self.paths.owner_legacy_json.read_text(encoding="utf-8"))
            except Exception:
                legacy = {}
            if not self.paths.owner_profile_json.exists():
                self.paths.owner_profile_json.write_text(
                    json.dumps(
                        {"owner_profile_memory": str(legacy.get("owner_profile_memory", "") or "")},
                        ensure_ascii=False,
                        indent=2,
                    )
                    + "\n",
                    encoding="utf-8",
                )
            if not self.paths.owner_constraints_json.exists():
                self.paths.owner_constraints_json.write_text(
                    json.dumps(
                        {
                            "owner_global_constraints": str(
                                legacy.get("owner_global_constraints", "") or ""
                            )
                        },
                        ensure_ascii=False,
                        indent=2,
                    )
                    + "\n",
                    encoding="utf-8",
                )

        if not self.paths.owner_profile_json.exists():
            self.paths.owner_profile_json.write_text(
                json.dumps({"owner_profile_memory": ""}, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
        if not self.paths.owner_constraints_json.exists():
            self.paths.owner_constraints_json.write_text(
                json.dumps({"owner_global_constraints": ""}, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )

    def _ensure_project_files(self) -> None:
        # Migrate from old combined project_memory.json when present.
        if self.paths.project_legacy_json.exists():
            try:
                legacy = json.loads(self.paths.project_legacy_json.read_text(encoding="utf-8"))
            except Exception:
                legacy = {}
            if not self.paths.project_long_json.exists():
                self.paths.project_long_json.write_text(
                    json.dumps(
                        {
                            "project_long_term_memory": str(
                                legacy.get("project_long_term_memory", "") or ""
                            )
                        },
                        ensure_ascii=False,
                        indent=2,
                    )
                    + "\n",
                    encoding="utf-8",
                )
            if not self.paths.project_short_json.exists():
                self.paths.project_short_json.write_text(
                    json.dumps(
                        {
                            "project_short_term_memory": str(
                                legacy.get("project_short_term_memory", "") or ""
                            )
                        },
                        ensure_ascii=False,
                        indent=2,
                    )
                    + "\n",
                    encoding="utf-8",
                )

        if not self.paths.project_long_json.exists():
            self.paths.project_long_json.write_text(
                json.dumps({"project_long_term_memory": ""}, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
        if not self.paths.project_short_json.exists():
            self.paths.project_short_json.write_text(
                json.dumps({"project_short_term_memory": ""}, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )

    def _read_owner(self) -> dict[str, str]:
        try:
            profile_payload = json.loads(self.paths.owner_profile_json.read_text(encoding="utf-8"))
        except Exception:
            profile_payload = {}
        try:
            constraints_payload = json.loads(
                self.paths.owner_constraints_json.read_text(encoding="utf-8")
            )
        except Exception:
            constraints_payload = {}
        return {
            "owner_profile_memory": str(profile_payload.get("owner_profile_memory", "") or ""),
            "owner_global_constraints": str(
                constraints_payload.get("owner_global_constraints", "") or ""
            ),
        }

    def _read_project(self) -> dict[str, str]:
        try:
            long_payload = json.loads(self.paths.project_long_json.read_text(encoding="utf-8"))
        except Exception:
            long_payload = {}
        try:
            short_payload = json.loads(self.paths.project_short_json.read_text(encoding="utf-8"))
        except Exception:
            short_payload = {}
        return {
            "project_long_term_memory": str(long_payload.get("project_long_term_memory", "") or ""),
            "project_short_term_memory": str(short_payload.get("project_short_term_memory", "") or ""),
        }

    def _read_blocks(self) -> dict[str, str]:
        payload = {}
        payload.update(self._read_owner())
        payload.update(self._read_project())
        out: dict[str, str] = {}
        for key in BLOCK_KEYS:
            out[key] = str(payload.get(key, "") or "")
        return out

    def _write_blocks(self, blocks: dict[str, str]) -> None:
        owner_profile_payload = {
            "owner_profile_memory": str(blocks.get("owner_profile_memory", "") or ""),
        }
        owner_constraints_payload = {
            "owner_global_constraints": str(blocks.get("owner_global_constraints", "") or ""),
        }
        project_long_payload = {
            "project_long_term_memory": str(blocks.get("project_long_term_memory", "") or ""),
        }
        project_short_payload = {
            "project_short_term_memory": str(blocks.get("project_short_term_memory", "") or ""),
        }
        self.paths.owner_profile_json.write_text(
            json.dumps(owner_profile_payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        self.paths.owner_constraints_json.write_text(
            json.dumps(owner_constraints_payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        self.paths.project_long_json.write_text(
            json.dumps(project_long_payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        self.paths.project_short_json.write_text(
            json.dumps(project_short_payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    def _ensure_context_db(self) -> None:
        with sqlite3.connect(self.paths.context_sqlite) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS context_memory (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ts_utc TEXT NOT NULL,
                    user_text TEXT NOT NULL,
                    assistant_text TEXT NOT NULL
                )
                """
            )
            conn.commit()

    def get_blocks(self) -> dict[str, str]:
        return self._read_blocks()

    def set_block(self, key: str, value: str) -> None:
        if key not in BLOCK_KEYS:
            raise ValueError(f"Unknown memory block: {key}")
        blocks = self._read_blocks()
        blocks[key] = str(value or "")
        self._write_blocks(blocks)

    def add_context_turn(self, user_text: str, assistant_text: str) -> None:
        ts = datetime.now(timezone.utc).isoformat()
        with sqlite3.connect(self.paths.context_sqlite) as conn:
            conn.execute(
                "INSERT INTO context_memory (ts_utc, user_text, assistant_text) VALUES (?, ?, ?)",
                (ts, str(user_text or ""), str(assistant_text or "")),
            )
            conn.commit()

    def get_recent_context(self, limit: int = 4) -> list[dict[str, str]]:
        n = max(1, int(limit))
        with sqlite3.connect(self.paths.context_sqlite) as conn:
            rows = conn.execute(
                """
                SELECT ts_utc, user_text, assistant_text
                FROM context_memory
                ORDER BY id DESC
                LIMIT ?
                """,
                (n,),
            ).fetchall()
        rows.reverse()
        return [
            {
                "ts_utc": str(r[0]),
                "user_text": str(r[1]),
                "assistant_text": str(r[2]),
            }
            for r in rows
        ]

    def record_turn(self, user_text: str, assistant_text: str, llm_client: Any) -> bool:
        """Persist context and run owner-memory refresh logic.

        Returns True if any memory refresh was triggered.
        """
        return self.record_turn_with_mode(
            user_text=user_text,
            assistant_text=assistant_text,
            llm_client=llm_client,
            async_mode=True,
        )

    def record_turn_with_mode(
        self,
        user_text: str,
        assistant_text: str,
        llm_client: Any,
        async_mode: bool,
    ) -> bool:
        """Persist context and update memories (sync or async)."""
        self.add_context_turn(user_text=user_text, assistant_text=assistant_text)
        self.last_owner_refresh_triggered = False
        self.last_project_refresh_triggered = False
        self.last_owner_sent_payload = None
        self.last_owner_returned_payload = None
        self.last_project_sent_payload = None
        self.last_project_returned_payload = None
        self.user_input_count += 1
        current_count = self.user_input_count

        owner_due = self._is_owner_due(current_count=current_count)
        project_due = self._is_project_due(current_count=current_count)
        self.last_owner_refresh_triggered = owner_due
        self.last_project_refresh_triggered = project_due

        if not owner_due and not project_due:
            return False

        if async_mode:
            t = threading.Thread(
                target=self._background_refresh_worker,
                args=(current_count, owner_due, project_due),
                daemon=True,
            )
            t.start()
            return True

        owner_refreshed = False
        project_refreshed = False
        if owner_due:
            owner_refreshed = self._update_owner_memories_if_due(
                llm_client=llm_client,
                current_count=current_count,
            )
        if project_due:
            project_refreshed = self._update_project_memories_if_due(
                llm_client=llm_client,
                current_count=current_count,
            )
        return bool(owner_refreshed or project_refreshed)

    def _background_refresh_worker(
        self,
        current_count: int,
        owner_due: bool,
        project_due: bool,
    ) -> None:
        try:
            from app.llm.client import LLMClient
        except Exception:
            return
        client = LLMClient()
        if owner_due:
            self._update_owner_memories_if_due(llm_client=client, current_count=current_count)
        if project_due:
            self._update_project_memories_if_due(llm_client=client, current_count=current_count)

    def _is_owner_due(self, current_count: int) -> bool:
        owner = self._read_owner()
        owner_total_bytes = _utf8_size(owner["owner_profile_memory"]) + _utf8_size(
            owner["owner_global_constraints"]
        )
        refresh_interval = 5 if owner_total_bytes < OWNER_MEMORY_FAST_INTERVAL_THRESHOLD_BYTES else 10
        return (
            current_count > 0
            and current_count % refresh_interval == 0
            and self.last_owner_memory_refresh_at != current_count
        )

    def _is_project_due(self, current_count: int) -> bool:
        if current_count <= 0:
            return False
        if current_count % PROJECT_MEMORY_INTERVAL != 0:
            return False
        if self.last_project_memory_refresh_at == current_count:
            return False
        return len(self.get_recent_context(limit=5)) >= 2

    def _update_owner_memories_if_due(self, llm_client: Any, current_count: int) -> bool:
        owner = self._read_owner()

        owner_total_bytes = _utf8_size(owner["owner_profile_memory"]) + _utf8_size(
            owner["owner_global_constraints"]
        )
        refresh_interval = (
            5
            if owner_total_bytes < OWNER_MEMORY_FAST_INTERVAL_THRESHOLD_BYTES
            else 10
        )

        should_refresh = (
            current_count > 0
            and current_count % refresh_interval == 0
            and self.last_owner_memory_refresh_at != current_count
        )
        if not should_refresh:
            return False
        self.last_owner_refresh_triggered = True

        recent_5 = self.get_recent_context(limit=5)
        task_payload = {
            "task": "owner_memory_update",
            "trigger": f"every_{refresh_interval}_user_inputs_per_project_dynamic",
            "goal": "Update only the two owner memories based on the latest 5 dialogues and existing owner memory content. If any output block would exceed 4096 UTF-8 bytes, compact that block to <= 2048 bytes in this same response.",
            "rules": [
                "owner_profile_memory stores fine-grained user habits and preferences.",
                "owner_global_constraints stores universal cross-project constraints/rules from the user.",
                "If no new valid habit/constraint is found, keep existing content unchanged.",
                "Do not add project-specific temporary details into owner memories.",
                "Do update+compaction in one shot. Do not require a second pass.",
                "Write memory content in English only.",
                "Return valid JSON only.",
            ],
            "owner_memory_total_bytes": owner_total_bytes,
            "refresh_interval": refresh_interval,
            "limits": {
                "max_bytes_per_block": OWNER_MEMORY_BLOCK_LIMIT_BYTES,
                "target_bytes_if_over_limit": OWNER_MEMORY_COMPACT_TARGET_BYTES,
            },
            "owner_profile_memory_current": owner["owner_profile_memory"],
            "owner_global_constraints_current": owner["owner_global_constraints"],
            "recent_5_dialogues": recent_5,
            "output_schema": {
                "owner_profile_memory": "string",
                "owner_global_constraints": "string",
            },
        }

        resp = llm_client.chat(
            messages=[
                {
                    "role": "system",
                    "content": self._memory_worker_system_prompt(
                        "owner_profile_memory and owner_global_constraints"
                    ),
                },
                {
                    "role": "user",
                    "content": json.dumps(task_payload, ensure_ascii=False),
                }
            ],
            tools=None,
            tool_choice=None,
        )
        self.last_owner_sent_payload = llm_client.last_sent_payload
        self.last_owner_returned_payload = llm_client.last_returned_payload
        parsed = self._parse_json_object(self._extract_text(resp))
        if isinstance(parsed, dict):
            new_profile = parsed.get("owner_profile_memory")
            new_constraints = parsed.get("owner_global_constraints")
            updated = False
            if isinstance(new_profile, str) and new_profile.strip():
                owner["owner_profile_memory"] = new_profile.strip()
                updated = True
            if isinstance(new_constraints, str) and new_constraints.strip():
                owner["owner_global_constraints"] = new_constraints.strip()
                updated = True
            if updated:
                # Safety net: enforce hard bounds locally if model output is still too large.
                if _utf8_size(owner["owner_profile_memory"]) > OWNER_MEMORY_BLOCK_LIMIT_BYTES:
                    owner["owner_profile_memory"] = owner["owner_profile_memory"].encode("utf-8")[
                        :OWNER_MEMORY_COMPACT_TARGET_BYTES
                    ].decode("utf-8", errors="ignore")
                if _utf8_size(owner["owner_global_constraints"]) > OWNER_MEMORY_BLOCK_LIMIT_BYTES:
                    owner["owner_global_constraints"] = owner["owner_global_constraints"].encode("utf-8")[
                        :OWNER_MEMORY_COMPACT_TARGET_BYTES
                    ].decode("utf-8", errors="ignore")
                blocks = self._read_blocks()
                blocks["owner_profile_memory"] = owner["owner_profile_memory"]
                blocks["owner_global_constraints"] = owner["owner_global_constraints"]
                self._write_blocks(blocks)

        self.last_owner_memory_refresh_at = current_count
        return True

    def _update_project_memories_if_due(self, llm_client: Any, current_count: int) -> bool:
        if current_count <= 0:
            return False
        if current_count % PROJECT_MEMORY_INTERVAL != 0:
            return False
        if self.last_project_memory_refresh_at == current_count:
            return False

        blocks = self._read_blocks()
        recent_5 = self.get_recent_context(limit=5)
        # Project memories should be distilled from recent 2~5 dialogue turns.
        if len(recent_5) < 2:
            return False

        self.last_project_refresh_triggered = True
        task_payload = {
            "task": "project_memories_update",
            "target_memories": [
                "project_short_term_memory",
                "project_long_term_memory",
            ],
            "project_short_term_memory_current": blocks["project_short_term_memory"],
            "project_long_term_memory_current": blocks["project_long_term_memory"],
            "recent_dialogues_2_to_5": recent_5,
            "limits": {
                "max_bytes_per_block": PROJECT_MEMORY_BLOCK_LIMIT_BYTES,
                "target_bytes_if_over_limit": PROJECT_MEMORY_COMPACT_TARGET_BYTES,
            },
            "output_schema": {
                "project_short_term_memory": "string",
                "project_long_term_memory": "string",
            },
        }
        resp = llm_client.chat(
            messages=[
                {
                    "role": "system",
                    "content": self._memory_worker_system_prompt(
                        "project_short_term_memory and project_long_term_memory"
                    ),
                },
                {
                    "role": "user",
                    "content": json.dumps(task_payload, ensure_ascii=False),
                },
            ],
            tools=None,
            tool_choice=None,
        )
        self.last_project_sent_payload = llm_client.last_sent_payload
        self.last_project_returned_payload = llm_client.last_returned_payload
        parsed = self._parse_json_object(self._extract_text(resp))
        updated = False
        if isinstance(parsed, dict):
            new_short = parsed.get("project_short_term_memory")
            if isinstance(new_short, str) and new_short.strip():
                short_value = new_short.strip()
                if _utf8_size(short_value) > PROJECT_MEMORY_BLOCK_LIMIT_BYTES:
                    short_value = short_value.encode("utf-8")[
                        :PROJECT_MEMORY_COMPACT_TARGET_BYTES
                    ].decode("utf-8", errors="ignore")
                blocks["project_short_term_memory"] = short_value
                updated = True
            new_long = parsed.get("project_long_term_memory")
            if isinstance(new_long, str) and new_long.strip():
                long_value = new_long.strip()
                if _utf8_size(long_value) > PROJECT_MEMORY_BLOCK_LIMIT_BYTES:
                    long_value = long_value.encode("utf-8")[
                        :PROJECT_MEMORY_COMPACT_TARGET_BYTES
                    ].decode("utf-8", errors="ignore")
                blocks["project_long_term_memory"] = long_value
                updated = True
        if updated:
            self._write_blocks(blocks)

        self.last_project_short_memory_refresh_at = current_count
        self.last_project_long_memory_refresh_at = current_count
        self.last_project_memory_refresh_at = current_count
        return True

    def _memory_worker_system_prompt(self, target: str) -> str:
        return (
            "You are a memory-maintenance worker. "
            f"Your only task is to update {target} from the provided payload. "
            "Do not chat, do not add unrelated content, and return valid JSON only. "
            "If no meaningful new signal exists, keep current memory unchanged. "
            "Write memory content in English only."
        )

    def _extract_text(self, resp: Any) -> str:
        if isinstance(resp, dict):
            return str(
                (
                    resp.get("choices", [{}])[0]
                    .get("message", {})
                    .get("content", "")
                )
                or ""
            )
        try:
            return str(resp.choices[0].message.content or "")
        except Exception:
            return ""

    def _parse_json_object(self, text: str) -> dict[str, Any] | None:
        raw = text.strip()
        if not raw:
            return None

        # Direct parse first.
        try:
            obj = json.loads(raw)
            return obj if isinstance(obj, dict) else None
        except Exception:
            pass

        # Try markdown fenced JSON.
        if "```" in raw:
            start = raw.find("{")
            end = raw.rfind("}")
            if start != -1 and end != -1 and end > start:
                try:
                    obj = json.loads(raw[start : end + 1])
                    return obj if isinstance(obj, dict) else None
                except Exception:
                    return None

        # Last attempt: first/last brace slice.
        start = raw.find("{")
        end = raw.rfind("}")
        if start == -1 or end == -1 or end <= start:
            return None
        try:
            obj = json.loads(raw[start : end + 1])
            return obj if isinstance(obj, dict) else None
        except Exception:
            return None

    def build_memory_system_sections(self) -> list[str]:
        blocks = self._read_blocks()
        recent = self.get_recent_context(limit=4)

        sections: list[str] = [
            "[OWNER_PROFILE_MEMORY]\n"
            + (blocks["owner_profile_memory"] or "(empty)"),
            "[OWNER_GLOBAL_CONSTRAINTS]\n"
            + (blocks["owner_global_constraints"] or "(empty)"),
            "[PROJECT_LONG_TERM_MEMORY]\n"
            + (blocks["project_long_term_memory"] or "(empty)"),
            "[PROJECT_SHORT_TERM_MEMORY]\n"
            + (blocks["project_short_term_memory"] or "(empty)"),
        ]

        context_lines: list[str] = ["[CONTEXT_MEMORY_RECENT_4]"]
        if not recent:
            context_lines.append("(empty)")
        else:
            for idx, item in enumerate(recent, 1):
                context_lines.extend(
                    [
                        f"- Turn {idx} | ts: {item.get('ts_utc', '')}",
                        f"  User: {_shorten_text(item.get('user_text', ''), 120)}",
                        f"  Assistant: {_shorten_text(item.get('assistant_text', ''), 180)}",
                    ]
                )
        sections.append("\n".join(context_lines))
        return sections
