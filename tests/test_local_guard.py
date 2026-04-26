from __future__ import annotations

import unittest

from app.orchestrator.agent_loop import SimpleAgentLoop


class LocalGuardTests(unittest.TestCase):
    def setUp(self) -> None:
        self.loop = SimpleAgentLoop.__new__(SimpleAgentLoop)

    def test_web_intent_requires_web_action_or_open(self) -> None:
        tool_events = [
            {
                "name": "web_scan",
                "result": {
                    "status": "ok",
                    "output": {"tabs": []},
                    "error": "",
                },
            }
        ]
        guard = self.loop._build_local_guard(
            tool_events=tool_events,
            user_text="打开百度",
            current_step={"title": "Open browser"},
        )
        self.assertFalse(bool(guard.get("requirement_met")))

    def test_web_action_ok_can_pass(self) -> None:
        tool_events = [
            {
                "name": "web_execute_js",
                "result": {
                    "status": "ok",
                    "output": {"action": "navigate", "browser_opened": False},
                    "error": "",
                },
            }
        ]
        guard = self.loop._build_local_guard(
            tool_events=tool_events,
            user_text="打开百度",
            current_step={"title": "Navigate to target website"},
        )
        self.assertTrue(bool(guard.get("requirement_met")))


if __name__ == "__main__":
    unittest.main()
