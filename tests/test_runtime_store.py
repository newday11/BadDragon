from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from app.orchestrator.runtime_store import TaskRuntimeStore


class RuntimeStoreTests(unittest.TestCase):
    def test_begin_update_finalize(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            store = TaskRuntimeStore(root_dir=root)
            store.begin_task(
                user_goal="打开百度",
                plan={
                    "mode": "task",
                    "task_name": "打开百度",
                    "steps": [{"step_id": 1, "title": "Open browser", "expected_output": "opened"}],
                    "done_criteria": ["opened"],
                    "fail_policy": {"max_retry_per_step": 2},
                },
            )
            self.assertIsNotNone(store.load_active())

            store.update_progress(
                step_index=1,
                execution_log=[{"step_id": 1, "status": "success"}],
            )
            active = store.load_active() or {}
            self.assertEqual(int(active.get("step_index", 0)), 1)

            store.finalize(status="done", summary={"task_name": "打开百度"})
            self.assertIsNone(store.load_active())
            self.assertTrue(store.last_path.exists())


if __name__ == "__main__":
    unittest.main()
