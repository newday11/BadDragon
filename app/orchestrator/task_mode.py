"""Task-mode classification and planning helpers.

This module intentionally uses minimal prompts (no main system/memory blocks)
for routing and planning decisions.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import re
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from app.llm.client import LLMClient


MODE_SIMPLE = "simple"
MODE_TASK = "task"


@dataclass
class ModeDecision:
    mode: str
    source: str
    reason: str


@dataclass
class TaskPlan:
    mode: str
    task_name: str
    steps: list[dict[str, Any]]
    done_criteria: list[str]
    fail_policy: dict[str, Any]


class TaskModeRouter:
    """LLM-first router/planner with strict minimal prompts."""

    def route_via_llm(
        self,
        *,
        user_text: str,
        last_assistant_answer: str,
        llm_client: "LLMClient",
    ) -> tuple[ModeDecision, TaskPlan | None, dict[str, Any], dict[str, Any]]:
        system_prompt = (
            "You are a task router and planner."
            "First decide whether the request is simple or task mode."
            "Output must be JSON and JSON only."
            "Format:"
            "{"
            "\"mode\":\"simple|task\","
            "\"reason\":\"string\","
            "\"answer\":\"string, required in simple mode and optional in task mode\","
            "\"task_plan\":{"
            "\"task_name\":\"string\","
            "\"steps\":[{\"step_id\":1,\"title\":\"string\",\"expected_output\":\"string\"}],"
            "\"done_criteria\":[\"string\"],"
            "\"fail_policy\":{\"max_retry_per_step\":2}"
            "} | null"
            "}."
            "Requirements:"
            "1) In simple mode: task_plan must be null."
            "2) In task mode: answer may be empty, task_plan must be complete, steps count must be 2 to 8, and step_id must increase from 1."
            "3) In task mode each step must contain at most one major action."
            "Do not output any text outside JSON."
        )
        user_prompt = (
            "Route and plan the following input when needed:\n"
            f"last_assistant_answer: {last_assistant_answer or '(empty)'}\n"
            f"current_user_message: {user_text or ''}"
        )
        resp = llm_client.chat(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0,
            max_tokens=800,
        )
        sent = llm_client.last_sent_payload or {}
        returned = llm_client.last_returned_payload or {}

        text = llm_client.extract_text(resp)
        parsed = self._parse_json_object(text)
        mode = MODE_SIMPLE
        reason = "llm_default_simple"
        plan: TaskPlan | None = None
        if isinstance(parsed, dict):
            raw_mode = str(parsed.get("mode", "")).strip().lower()
            if raw_mode in {"task", "complex"}:
                mode = MODE_TASK
            elif raw_mode == "simple":
                mode = MODE_SIMPLE
            else:
                mode = MODE_TASK if "task" in raw_mode or "complex" in raw_mode else MODE_SIMPLE
            reason = str(parsed.get("reason", "llm_json")) or "llm_json"
            if mode == MODE_TASK:
                raw_plan = parsed.get("task_plan")
                if isinstance(raw_plan, dict):
                    plan = self._normalize_plan(raw_plan, fallback_user_text=user_text)
                else:
                    # tolerate models that put plan fields at top-level
                    plan = self._normalize_plan(parsed, fallback_user_text=user_text)
        else:
            low = (text or "").strip().lower()
            if "task" in low or "complex" in low:
                mode = MODE_TASK
                reason = "llm_text_task"
            else:
                mode = MODE_SIMPLE
                reason = "llm_text_simple"
        if mode == MODE_TASK and plan is None:
            # safe fallback plan
            plan = self._normalize_plan({}, fallback_user_text=user_text)
        # Local deterministic override:
        # if the user message clearly asks for executable actions, force task mode.
        if mode == MODE_SIMPLE and self._looks_task_like(user_text):
            mode = MODE_TASK
            reason = f"local_rule_override:{reason}"
            if plan is None:
                plan = self._normalize_plan({}, fallback_user_text=user_text)
        return ModeDecision(mode=mode, source="llm", reason=reason), plan, sent, returned

    def classify_via_llm(
        self,
        *,
        user_text: str,
        last_assistant_answer: str,
        llm_client: "LLMClient",
    ) -> tuple[ModeDecision, dict[str, Any], dict[str, Any]]:
        """Backward-compatible wrapper."""
        decision, _plan, sent, returned = self.route_via_llm(
            user_text=user_text,
            last_assistant_answer=last_assistant_answer,
            llm_client=llm_client,
        )
        return decision, sent, returned

    def fallback_plan(self, *, user_text: str) -> TaskPlan:
        """Return a safe default plan when routing output is malformed."""
        return self._normalize_plan({}, fallback_user_text=user_text)

    def plan_task_via_llm(
        self,
        *,
        user_text: str,
        last_assistant_answer: str,
        llm_client: "LLMClient",
    ) -> tuple[TaskPlan, dict[str, Any], dict[str, Any]]:
        system_prompt = (
            "You are a task planner. Break the user request into executable steps."
            "Output must be JSON and JSON only."
            "JSON schema:"
            "{"
            "\"mode\":\"task\","
            "\"task_name\":\"string\","
            "\"steps\":[{\"step_id\":1,\"title\":\"string\",\"expected_output\":\"string\"}],"
            "\"done_criteria\":[\"string\"],"
            "\"fail_policy\":{\"max_retry_per_step\":2}"
            "}"
            "Requirements: steps count must be 2 to 8; step_id must increase from 1; each step can contain at most one major action."
        )
        user_prompt = (
            "Generate a task plan from the following input:\n"
            f"last_assistant_answer: {last_assistant_answer or '(empty)'}\n"
            f"current_user_message: {user_text or ''}"
        )
        resp = llm_client.chat(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0,
            max_tokens=800,
        )
        sent = llm_client.last_sent_payload or {}
        returned = llm_client.last_returned_payload or {}

        text = llm_client.extract_text(resp)
        parsed = self._parse_json_object(text) or {}
        plan = self._normalize_plan(parsed, fallback_user_text=user_text)
        return plan, sent, returned

    def judge_step_via_llm(
        self,
        *,
        current_step: dict[str, Any],
        tool_events: list[dict[str, Any]],
        assistant_response_text: str,
        local_guard: dict[str, Any],
        llm_client: "LLMClient",
    ) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
        system_prompt = (
            "You are a step-result judge."
            "Output JSON only: {\"status\":\"success|retry|replan|fail\",\"reason\":\"...\"}."
            "Do not output text outside JSON."
        )
        compact_events: list[dict[str, Any]] = []
        for item in tool_events[-8:]:
            result = item.get("result", {}) if isinstance(item, dict) else {}
            output = result.get("output", {}) if isinstance(result, dict) else {}
            compact_events.append(
                {
                    "name": item.get("name"),
                    "status": result.get("status"),
                    "error": result.get("error", ""),
                    "output_preview": str(output)[:500],
                }
            )
        user_prompt = (
            "Judge the current step execution result:\n"
            f"current_step: {json.dumps(current_step, ensure_ascii=False)}\n"
            f"tool_events: {json.dumps(compact_events, ensure_ascii=False)}\n"
            f"assistant_response_text: {assistant_response_text or ''}\n"
            f"local_guard: {json.dumps(local_guard, ensure_ascii=False)}\n"
            "Rule: if local_guard.requirement_met is false, success is forbidden."
        )
        resp = llm_client.chat(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0,
            max_tokens=64,
        )
        sent = llm_client.last_sent_payload or {}
        returned = llm_client.last_returned_payload or {}
        text = llm_client.extract_text(resp)
        parsed = self._parse_json_object(text) or {}
        status = str(parsed.get("status", "")).strip().lower()
        if status not in {"success", "retry", "replan", "fail"}:
            # Safe default: retry once rather than claiming success.
            status = "retry"
        reason = str(parsed.get("reason", "llm_judge")) or "llm_judge"
        return {"status": status, "reason": reason}, sent, returned

    def verify_done_via_llm(
        self,
        *,
        user_text: str,
        done_criteria: list[str],
        execution_log: list[dict[str, Any]],
        llm_client: "LLMClient",
    ) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
        system_prompt = (
            "You are the final task verifier."
            "You must strictly evaluate completion against done_criteria."
            "Output JSON only: {\"status\":\"pass|fail\",\"reason\":\"...\"}."
            "Do not output text outside JSON."
        )
        user_prompt = (
            "Run final verification:\n"
            f"user_goal: {user_text or ''}\n"
            f"done_criteria: {json.dumps(done_criteria, ensure_ascii=False)}\n"
            f"execution_log: {json.dumps(execution_log, ensure_ascii=False)}"
        )
        resp = llm_client.chat(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0,
            max_tokens=96,
        )
        sent = llm_client.last_sent_payload or {}
        returned = llm_client.last_returned_payload or {}
        text = llm_client.extract_text(resp)
        parsed = self._parse_json_object(text) or {}
        status = str(parsed.get("status", "")).strip().lower()
        if status not in {"pass", "fail"}:
            status = "fail"
        reason = str(parsed.get("reason", "final_verify_failed")).strip() or "final_verify_failed"
        return {"status": status, "reason": reason}, sent, returned

    def replan_remaining_via_llm(
        self,
        *,
        user_text: str,
        original_plan: TaskPlan,
        current_step: dict[str, Any],
        failure_reason: str,
        llm_client: "LLMClient",
    ) -> tuple[TaskPlan, dict[str, Any], dict[str, Any]]:
        system_prompt = (
            "You are a task replanner."
            "Replan remaining steps based on the failure reason."
            "Output JSON only, using the same schema as the task planner."
            "Do not output text outside JSON."
        )
        user_prompt = (
            "Replan the remaining steps for this task:\n"
            f"user_goal: {user_text}\n"
            f"original_plan: {json.dumps(original_plan.__dict__, ensure_ascii=False)}\n"
            f"failed_step: {json.dumps(current_step, ensure_ascii=False)}\n"
            f"failure_reason: {failure_reason}"
        )
        resp = llm_client.chat(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0,
            max_tokens=900,
        )
        sent = llm_client.last_sent_payload or {}
        returned = llm_client.last_returned_payload or {}
        text = llm_client.extract_text(resp)
        parsed = self._parse_json_object(text) or {}
        plan = self._normalize_plan(parsed, fallback_user_text=user_text)
        return plan, sent, returned

    def _normalize_plan(self, data: dict[str, Any], *, fallback_user_text: str) -> TaskPlan:
        fallback_goal = str(fallback_user_text or "").strip()
        task_name = str(data.get("task_name", "") or "").strip()
        if not task_name:
            task_name = fallback_goal or "Execute user request"
        raw_steps = data.get("steps")
        steps: list[dict[str, Any]] = []
        if isinstance(raw_steps, list):
            for i, item in enumerate(raw_steps, start=1):
                if not isinstance(item, dict):
                    continue
                title = str(item.get("title", "")).strip() or f"Step {i}"
                expected = str(item.get("expected_output", "")).strip() or "Step output is completed."
                sid = item.get("step_id", i)
                try:
                    sid = int(sid)
                except Exception:
                    sid = i
                steps.append({"step_id": sid, "title": title, "expected_output": expected})
        if not steps:
            steps = [
                {
                    "step_id": 1,
                    "title": self._fallback_step_title(fallback_goal),
                    "expected_output": self._fallback_expected_output(fallback_goal),
                }
            ]
        # Enforce minimum step count for task mode.
        # Models may ignore prompt constraints and return a single step.
        if len(steps) < 2:
            steps = self._expand_to_min_two_steps(steps=steps, goal=fallback_goal)

        done = data.get("done_criteria")
        if not isinstance(done, list) or not done:
            done_criteria = ["All planned steps are completed."]
        else:
            done_criteria = [str(x) for x in done if str(x).strip()] or ["All planned steps are completed."]

        fail_policy = data.get("fail_policy") if isinstance(data.get("fail_policy"), dict) else {}
        max_retry = fail_policy.get("max_retry_per_step", 2)
        try:
            max_retry = int(max_retry)
        except Exception:
            max_retry = 2
        if max_retry < 1:
            max_retry = 1
        if max_retry > 5:
            max_retry = 5

        return TaskPlan(
            mode=MODE_TASK,
            task_name=task_name,
            steps=steps,
            done_criteria=done_criteria,
            fail_policy={"max_retry_per_step": max_retry},
        )

    @staticmethod
    def _expand_to_min_two_steps(*, steps: list[dict[str, Any]], goal: str) -> list[dict[str, Any]]:
        g = str(goal or "").strip()
        low = g.lower()
        is_web_goal = any(k in low for k in ["http://", "https://", "www.", "browser", "website", "web", "url"])
        if is_web_goal:
            return [
                {
                    "step_id": 1,
                    "title": "Open browser",
                    "expected_output": "A browser window is opened.",
                },
                {
                    "step_id": 2,
                    "title": "Navigate to target website",
                    "expected_output": "Target website is loaded and visible.",
                },
            ]
        first = steps[0] if steps else {
            "step_id": 1,
            "title": "Execute user request",
            "expected_output": "Primary action is completed.",
        }
        return [
            {
                "step_id": 1,
                "title": str(first.get("title", "Execute user request")),
                "expected_output": str(first.get("expected_output", "Primary action is completed.")),
            },
            {
                "step_id": 2,
                "title": "Verify result",
                "expected_output": "Result is verified with observable evidence.",
            },
        ]

    @staticmethod
    def _fallback_step_title(goal: str) -> str:
        g = str(goal or "").strip()
        if not g:
            return "Execute user request"
        # keep it short and readable in CLI
        if len(g) <= 36:
            return g
        return g[:35] + "…"

    @staticmethod
    def _fallback_expected_output(goal: str) -> str:
        g = str(goal or "").strip()
        if not g:
            return "Step completed."
        return f"Completed: {g}"

    @staticmethod
    def _parse_json_object(text: str) -> dict[str, Any] | None:
        if not text:
            return None
        text = text.strip()
        # direct parse
        try:
            obj = json.loads(text)
            if isinstance(obj, dict):
                return obj
        except Exception:
            pass

        # fenced json block
        m = re.search(r"```json\s*(\{[\s\S]*?\})\s*```", text, re.IGNORECASE)
        if m:
            try:
                obj = json.loads(m.group(1))
                if isinstance(obj, dict):
                    return obj
            except Exception:
                pass

        # first object substring
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            frag = text[start : end + 1]
            try:
                obj = json.loads(frag)
                if isinstance(obj, dict):
                    return obj
            except Exception:
                return None
        return None

    @staticmethod
    def _looks_task_like(user_text: str) -> bool:
        text = (user_text or "").strip().lower()
        if not text:
            return False
        action_tokens = [
            "open ",
            "create ",
            "build ",
            "develop ",
            "run ",
            "install ",
            "deploy ",
            "fix ",
            "modify ",
            "refactor ",
            "generate ",
            "query file",
        ]
        object_tokens = [
            "website",
            "webpage",
            "code",
            "script",
            "file",
            "project",
            "api",
            "database",
            "browser",
            "python",
            "bash",
            "git",
            "npm",
            "pip",
            "/",
            ".py",
            ".js",
            ".ts",
            ".json",
        ]
        has_action = any(tok in text for tok in action_tokens)
        has_object = any(tok in text for tok in object_tokens)
        return has_action and has_object
