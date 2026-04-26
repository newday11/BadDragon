"""Task-mode classification and planning helpers.

This module intentionally uses minimal prompts (no main system/memory blocks)
for routing and planning decisions.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import re
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from app.llm.client import LLMClient


MODE_SIMPLE = "simple"
MODE_TASK = "task"
PROMPT_DIR = Path(__file__).resolve().parents[1] / "prompts" / "task_mode"


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

    @staticmethod
    def _load_prompt_template(filename: str, fallback: str) -> str:
        path = PROMPT_DIR / filename
        try:
            if path.exists():
                content = path.read_text(encoding="utf-8")
                if content.strip():
                    return content
        except Exception:
            pass
        return fallback

    @staticmethod
    def _render_template(template: str, values: dict[str, Any]) -> str:
        text = str(template or "")
        for key, value in values.items():
            token = "{{" + str(key) + "}}"
            text = text.replace(token, str(value))
        return text

    def route_via_llm(
        self,
        *,
        user_text: str,
        last_assistant_answer: str,
        llm_client: "LLMClient",
    ) -> tuple[ModeDecision, TaskPlan | None, dict[str, Any], dict[str, Any]]:
        # Stage-1: deterministic basic rules.
        basic_mode, basic_reason, basic_hard, basic_confidence, basic_rules = self._basic_mode_judge(
            user_text=user_text
        )

        default_system_prompt = (
            "You are a task router and planner."
            "You are in stage-2 (LLM judge) after a stage-1 basic rule check."
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
        default_user_prompt = (
            "Route and plan the following input when needed:\n"
            "stage1_basic_mode: {{stage1_basic_mode}}\n"
            "stage1_reason: {{stage1_reason}}\n"
            "stage1_hard: {{stage1_hard}}\n"
            "stage1_confidence: {{stage1_confidence}}\n"
            "stage1_matched_rules: {{stage1_matched_rules}}\n"
            "last_assistant_answer: {{last_assistant_answer}}\n"
            "current_user_message: {{current_user_message}}"
        )
        system_prompt = self._load_prompt_template("route_system.txt", default_system_prompt)
        user_template = self._load_prompt_template("route_user.txt", default_user_prompt)
        user_prompt = self._render_template(
            user_template,
            {
                "stage1_basic_mode": basic_mode or "undecided",
                "stage1_reason": basic_reason,
                "stage1_hard": basic_hard,
                "stage1_confidence": f"{basic_confidence:.2f}",
                "stage1_matched_rules": json.dumps(basic_rules, ensure_ascii=False),
                "last_assistant_answer": last_assistant_answer or "(empty)",
                "current_user_message": user_text or "",
            },
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
        llm_mode = MODE_SIMPLE
        llm_reason = reason
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
            llm_mode = mode
            llm_reason = reason
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
            llm_mode = mode
            llm_reason = reason
        if mode == MODE_TASK and plan is None:
            # safe fallback plan
            plan = self._normalize_plan({}, fallback_user_text=user_text)

        # Merge stage-1 and stage-2:
        # - hard basic rules have priority;
        # - task-conflict prefers task when stage-1 has enough confidence.
        # - non-hard basic rules are used when LLM output is weak/invalid.
        source = "llm"
        conflict_policy_note = ""
        if basic_mode in {MODE_SIMPLE, MODE_TASK} and basic_hard:
            mode = basic_mode
            source = "basic+llm"
            reason = f"basic_hard:{basic_reason}; llm:{reason}"
            if mode == MODE_TASK and plan is None:
                plan = self._normalize_plan({}, fallback_user_text=user_text)
        elif basic_mode == MODE_TASK and llm_mode == MODE_SIMPLE and basic_confidence >= 0.70:
            mode = MODE_TASK
            source = "basic_conflict_override+llm"
            conflict_policy_note = "task_conflict_override"
            reason = (
                f"basic_conflict_override:{basic_reason}@{basic_confidence:.2f}; "
                f"llm:{llm_reason}"
            )
            if plan is None:
                plan = self._normalize_plan({}, fallback_user_text=user_text)
        elif basic_mode == MODE_SIMPLE and llm_mode == MODE_TASK and basic_confidence >= 0.90:
            mode = MODE_SIMPLE
            source = "basic_conflict_override+llm"
            conflict_policy_note = "simple_conflict_override"
            reason = (
                f"basic_conflict_override:{basic_reason}@{basic_confidence:.2f}; "
                f"llm:{llm_reason}"
            )
        elif basic_mode in {MODE_SIMPLE, MODE_TASK} and reason in {"llm_default_simple", "llm_text_simple", "llm_text_task"}:
            mode = basic_mode
            source = "basic_fallback+llm"
            reason = f"basic_fallback:{basic_reason}; llm:{reason}"
            if mode == MODE_TASK and plan is None:
                plan = self._normalize_plan({}, fallback_user_text=user_text)

        # If mode is task, we must have a concrete and valid task plan from LLM.
        need_explicit_plan = False
        if mode == MODE_TASK:
            if plan is None:
                need_explicit_plan = True
            elif not self._is_plan_quality_ok(plan):
                need_explicit_plan = True
        if need_explicit_plan:
            planned, plan_sent, plan_returned = self.plan_task_via_llm(
                user_text=user_text,
                last_assistant_answer=last_assistant_answer,
                llm_client=llm_client,
            )
            plan = planned
            # Retry planning once when plan quality is still weak.
            if not self._is_plan_quality_ok(plan):
                planned2, plan_sent2, plan_returned2 = self.plan_task_via_llm(
                    user_text=user_text,
                    last_assistant_answer=last_assistant_answer,
                    llm_client=llm_client,
                )
                plan = planned2
                plan_sent = {"attempt1": plan_sent, "attempt2": plan_sent2}
                plan_returned = {"attempt1": plan_returned, "attempt2": plan_returned2}
            source = f"{source}+planner"
            reason = f"{reason}; explicit_planner_used"
            if conflict_policy_note:
                reason = f"{reason}; {conflict_policy_note}"
            sent = {
                "route_sent": sent,
                "plan_sent": plan_sent,
            }
            returned = {
                "route_returned": returned,
                "plan_returned": plan_returned,
            }
        elif conflict_policy_note:
            reason = f"{reason}; {conflict_policy_note}"

        return ModeDecision(mode=mode, source=source, reason=reason), plan, sent, returned

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
        default_system_prompt = (
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
        default_user_prompt = (
            "Generate a task plan from the following input:\n"
            "last_assistant_answer: {{last_assistant_answer}}\n"
            "current_user_message: {{current_user_message}}"
        )
        system_prompt = self._load_prompt_template("plan_system.txt", default_system_prompt)
        user_template = self._load_prompt_template("plan_user.txt", default_user_prompt)
        user_prompt = self._render_template(
            user_template,
            {
                "last_assistant_answer": last_assistant_answer or "(empty)",
                "current_user_message": user_text or "",
            },
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
            # Deterministic fallback: if local guard is satisfied and there is
            # concrete tool evidence, treat as success; otherwise retry.
            has_ok_tool = any(
                str((item.get("result", {}) if isinstance(item, dict) else {}).get("status", "")).lower() == "ok"
                for item in tool_events
            )
            if bool(local_guard.get("requirement_met", False)) and has_ok_tool:
                status = "success"
            else:
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
                # Normalize step_id to sequential order for deterministic execution.
                steps.append({"step_id": i, "title": title, "expected_output": expected})
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

    @staticmethod
    def _basic_mode_judge(user_text: str) -> tuple[str | None, str, bool, float, list[str]]:
        """Stage-1 deterministic mode judgment.

        Returns:
          mode: simple/task/None
          reason: short reason
          hard: whether this rule should override stage-2 LLM output
          confidence: 0~1 confidence score
          matched_rules: matched rule ids/names
        """
        text = (user_text or "").strip().lower()
        if not text:
            return None, "empty_input", False, 0.0, []

        web_task_keys = [
            "http://", "https://", "www.", "网站", "网页", "url", "browser",
            "打开", "访问", "navigate", "crawl", "抓取",
        ]
        exec_task_keys = [
            "run ", "execute ", "install ", "deploy ", "fix ", "modify ",
            "写代码", "改代码", "文件", ".py", ".js", ".ts", "git", "bash", "python",
        ]
        simple_keys = [
            "解释", "是什么", "为什么", "how", "what", "why", "概念", "原理", "difference",
        ]

        if any(k in text for k in web_task_keys):
            return MODE_TASK, "web_intent", True, 0.95, ["web_intent"]
        if any(k in text for k in exec_task_keys):
            return MODE_TASK, "execution_intent", True, 0.90, ["execution_intent"]
        if any(k in text for k in simple_keys) and not TaskModeRouter._looks_task_like(text):
            return MODE_SIMPLE, "explanatory_intent", False, 0.72, ["explanatory_intent"]
        if TaskModeRouter._looks_task_like(text):
            return MODE_TASK, "action_object_pattern", False, 0.68, ["action_object_pattern"]
        return None, "no_rule_hit", False, 0.0, []

    @staticmethod
    def _is_plan_quality_ok(plan: TaskPlan | None) -> bool:
        if plan is None:
            return False
        steps = plan.steps if isinstance(plan.steps, list) else []
        if len(steps) < 2 or len(steps) > 8:
            return False
        expected_sid = 1
        for item in steps:
            if not isinstance(item, dict):
                return False
            sid = item.get("step_id")
            try:
                sid_int = int(sid)
            except Exception:
                return False
            if sid_int != expected_sid:
                return False
            title = str(item.get("title", "") or "").strip()
            expected = str(item.get("expected_output", "") or "").strip()
            if not title or not expected:
                return False
            if not TaskModeRouter._is_single_major_action_title(title):
                return False
            expected_sid += 1
        return True

    @staticmethod
    def _is_single_major_action_title(title: str) -> bool:
        t = str(title or "").strip().lower()
        if not t:
            return False
        # Heuristic guard: avoid compound titles that bundle multiple major actions.
        multi_action_markers = [" and ", " then ", "并且", "然后", "以及", "同时", "；", ";"]
        return not any(m in t for m in multi_action_markers)
