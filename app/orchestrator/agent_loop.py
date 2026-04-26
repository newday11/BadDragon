"""Orchestrator loop with mode routing and task-step execution."""

from __future__ import annotations

import json
from typing import Any, Callable

from app.context.system_prompt import build_system_prompt_protocol
from app.llm.client import LLMClient
from app.memory.store import MemoryStore
from app.orchestrator.runtime_store import TaskRuntimeStore
from app.orchestrator.state import StepState, StepStateMachine
from app.orchestrator.task_mode import MODE_TASK, ModeDecision, TaskModeRouter, TaskPlan
from app.tools.registry import ToolRegistry


class SimpleAgentLoop:
    """Single-agent orchestrator with simple/task mode routing."""

    def __init__(self, async_memory_updates: bool = True) -> None:
        self.client = LLMClient()
        self.memory = MemoryStore()
        self.mode_router = TaskModeRouter()
        self.tool_registry = ToolRegistry()
        self.runtime_store = TaskRuntimeStore()
        self.async_memory_updates = async_memory_updates
        self.system_message = {"role": "system", "content": build_system_prompt_protocol()}

        self.last_assistant_text = ""
        self.last_mode_decision = ModeDecision(mode="simple", source="init", reason="init")
        self.last_task_plan: dict[str, Any] | None = None

        self.last_mode_sent_payload: dict[str, Any] | None = None
        self.last_mode_returned_payload: dict[str, Any] | None = None
        self.last_plan_sent_payload: dict[str, Any] | None = None
        self.last_plan_returned_payload: dict[str, Any] | None = None
        self.last_step_traces: list[dict[str, Any]] = []

        self.last_owner_memory_refresh_success = False
        self.last_project_memory_refresh_success = False
        self.last_chat_sent_payload: dict[str, Any] | None = None
        self.last_chat_returned_payload: dict[str, Any] | None = None
        self._progress_callback: Callable[[str, dict[str, Any]], None] | None = None
        self.last_turn_llm_stats: dict[str, int] = {
            "total": 0,
            "route": 0,
            "simple_answer": 0,
            "task_execute": 0,
            "task_judge": 0,
            "task_final_verify": 0,
            "task_replan": 0,
            "planner": 0,
            "owner_memory": 0,
            "project_memory": 0,
            "other": 0,
        }
        self.last_resume_used = False

    def ask(
        self,
        user_text: str,
        progress_callback: Callable[[str, dict[str, Any]], None] | None = None,
    ) -> str:
        self._progress_callback = progress_callback
        call_start = int(getattr(self.client, "total_chat_calls", 0))
        self.last_step_traces = []
        self.last_task_plan = None
        self.last_plan_sent_payload = None
        self.last_plan_returned_payload = None
        self.last_resume_used = False

        active = self.runtime_store.load_active()
        if active and self._is_resume_command(user_text):
            self.last_mode_decision = ModeDecision(
                mode=MODE_TASK,
                source="runtime_resume",
                reason="resume_active_task",
            )
            self.last_mode_sent_payload = {}
            self.last_mode_returned_payload = {}
            plan_data = active.get("plan", {}) if isinstance(active, dict) else {}
            plan = TaskPlan(
                mode=str(plan_data.get("mode", MODE_TASK) or MODE_TASK),
                task_name=str(plan_data.get("task_name", "Resumed task")),
                steps=list(plan_data.get("steps", []) or []),
                done_criteria=list(plan_data.get("done_criteria", []) or ["All planned steps are completed."]),
                fail_policy=dict(plan_data.get("fail_policy", {}) or {"max_retry_per_step": 2}),
            )
            start_step_index = int(active.get("step_index", 0) or 0)
            execution_log = list(active.get("execution_log", []) or [])
            user_goal = str(active.get("user_goal", "") or user_text)
            self.last_resume_used = True
            content = self._run_task_plan(
                user_text=user_goal,
                plan=plan,
                start_step_index=start_step_index,
                execution_log=execution_log,
            )
        else:
            mode_decision, task_plan, mode_sent, mode_returned = self.mode_router.route_via_llm(
                user_text=user_text,
                last_assistant_answer=self.last_assistant_text,
                llm_client=self.client,
            )
            self.last_mode_decision = mode_decision
            self.last_mode_sent_payload = mode_sent
            self.last_mode_returned_payload = mode_returned

            if mode_decision.mode == MODE_TASK:
                content = self._ask_task_mode(user_text, routed_plan=task_plan)
            else:
                content = self._ask_simple_mode(user_text)

        content = str(content).strip() or "I received your message, but no valid response was generated this turn."
        self.last_assistant_text = content

        self.memory.record_turn_with_mode(
            user_text=user_text,
            assistant_text=content,
            llm_client=self.client,
            async_mode=self.async_memory_updates,
        )
        self.last_owner_memory_refresh_success = self.memory.last_owner_refresh_triggered
        self.last_project_memory_refresh_success = self.memory.last_project_refresh_triggered

        call_end = int(getattr(self.client, "total_chat_calls", 0))
        total_calls = max(0, call_end - call_start)
        route_calls = 0 if self.last_resume_used else 1
        planner_calls = 1 if self.last_plan_sent_payload is not None else 0
        step_exec_calls = 0
        judge_calls = 0
        replan_calls = 0
        final_verify_calls = 0
        for item in self.last_step_traces:
            if isinstance(item, dict) and item.get("type") == "replan":
                replan_calls += 1
                continue
            if isinstance(item, dict) and item.get("type") == "final_verify":
                final_verify_calls += 1
                continue
            if isinstance(item, dict) and "step" in item:
                step_exec_calls += int(item.get("llm_rounds", 1) or 1)
                judge_calls += 1
        simple_calls = 1 if self.last_mode_decision.mode != MODE_TASK else 0
        main_calls = (
            route_calls
            + planner_calls
            + simple_calls
            + step_exec_calls
            + judge_calls
            + final_verify_calls
            + replan_calls
        )
        memory_calls = max(0, total_calls - main_calls)
        owner_memory_calls = 1 if (self.memory.last_owner_sent_payload is not None) else 0
        project_memory_calls = 1 if (self.memory.last_project_sent_payload is not None) else 0
        other_calls = max(0, memory_calls - owner_memory_calls - project_memory_calls)
        self.last_turn_llm_stats = {
            "total": total_calls,
            "route": route_calls,
            "simple_answer": simple_calls,
            "task_execute": step_exec_calls,
            "task_judge": judge_calls,
            "task_final_verify": final_verify_calls,
            "task_replan": replan_calls,
            "planner": planner_calls,
            "owner_memory": owner_memory_calls,
            "project_memory": project_memory_calls,
            "other": other_calls,
            "resumed": 1 if self.last_resume_used else 0,
        }

        self._progress_callback = None
        return content

    @staticmethod
    def _is_resume_command(text: str) -> bool:
        t = str(text or "").strip().lower()
        return t in {"resume", "continue", "continue task", "resume task", "resume execution"}

    def _emit_progress(self, event: str, payload: dict[str, Any]) -> None:
        if self._progress_callback is None:
            return
        try:
            self._progress_callback(event, payload)
        except Exception:
            # Progress output should never break the main execution loop.
            return

    def _ask_simple_mode(self, user_text: str) -> str:
        prompt_messages: list[dict[str, str]] = [
            self.system_message,
            {
                "role": "system",
                "content": (
                    "[TASK_MODE]\n"
                    "mode: simple\n"
                    "For explanatory requests, answer directly and concisely without unnecessary execution."
                ),
            },
        ]
        for section in self._memory_sections_for_mode(mode="simple"):
            prompt_messages.append({"role": "system", "content": section})
        prompt_messages.append({"role": "user", "content": user_text})

        resp = self.client.chat(messages=prompt_messages)
        self.last_chat_sent_payload = self.client.last_sent_payload
        self.last_chat_returned_payload = self.client.last_returned_payload
        return self.client.extract_text(resp)

    def _ask_task_mode(self, user_text: str, *, routed_plan: TaskPlan | None) -> str:
        plan = routed_plan or self.mode_router.fallback_plan(user_text=user_text)
        # Planning is now merged into mode routing. Keep these fields as None so
        # debug output clearly indicates there was no second planning request.
        self.last_plan_sent_payload = None
        self.last_plan_returned_payload = None
        self.last_task_plan = {
            "mode": plan.mode,
            "task_name": plan.task_name,
            "steps": plan.steps,
            "done_criteria": plan.done_criteria,
            "fail_policy": plan.fail_policy,
        }
        self._emit_progress(
            "task_plan",
            {
                "task_name": plan.task_name,
                "steps": plan.steps,
                "done_criteria": plan.done_criteria,
            },
        )
        self.runtime_store.begin_task(
            user_goal=user_text,
            plan={
                "mode": plan.mode,
                "task_name": plan.task_name,
                "steps": plan.steps,
                "done_criteria": plan.done_criteria,
                "fail_policy": plan.fail_policy,
            },
        )
        return self._run_task_plan(
            user_text=user_text,
            plan=plan,
            start_step_index=0,
            execution_log=[],
        )

    def _run_task_plan(
        self,
        *,
        user_text: str,
        plan: TaskPlan,
        start_step_index: int,
        execution_log: list[dict[str, Any]],
    ) -> str:
        max_retry = int(plan.fail_policy.get("max_retry_per_step", 2) or 2)
        if max_retry < 1:
            max_retry = 1
        if max_retry > 5:
            max_retry = 5

        steps: list[dict[str, Any]] = list(plan.steps)
        step_index = max(0, int(start_step_index))
        if step_index > len(steps):
            step_index = len(steps)

        while step_index < len(steps):
            step = steps[step_index]
            retry_count = 0
            failure_reason = ""
            failure_stage = "read_error"
            last_observation_fingerprint = ""
            step_sm = StepStateMachine(step_id=int(step.get("step_id", step_index + 1) or (step_index + 1)))

            while True:
                if step_sm.state != StepState.ACT:
                    ok, trans_msg = step_sm.transition(StepState.ACT, "start_step_attempt")
                    if not ok:
                        return f"Task failed: {trans_msg}"

                (
                    step_response,
                    step_sent,
                    step_returned,
                    step_rounds,
                    tool_events,
                    step_meta,
                ) = self._execute_single_step(
                    user_text=user_text,
                    plan=plan,
                    current_step=step,
                    failure_reason=failure_reason,
                    failure_stage=failure_stage,
                )
                ok, trans_msg = step_sm.transition(StepState.OBSERVE, "observe_step_result")
                if not ok:
                    return f"Task failed: {trans_msg}"

                local_guard = self._build_local_guard(
                    tool_events=tool_events,
                    user_text=user_text,
                    current_step=step,
                )
                observation_fp = self._observation_fingerprint(
                    tool_events=tool_events,
                    step_response=step_response,
                )

                judge, judge_sent, judge_returned = self.mode_router.judge_step_via_llm(
                    current_step=step,
                    tool_events=tool_events,
                    assistant_response_text=step_response,
                    local_guard=local_guard,
                    llm_client=self.client,
                )
                ok, trans_msg = step_sm.transition(StepState.DECIDE, "judge_step_result")
                if not ok:
                    return f"Task failed: {trans_msg}"

                # Hard constraint: without valid tool evidence, success is forbidden.
                if not local_guard.get("requirement_met", False) and judge.get("status") == "success":
                    judge = {
                        "status": "retry",
                        "reason": "local_guard_blocked_success:no_valid_tool_evidence",
                    }
                # No-new-information retry is forbidden.
                if (
                    judge.get("status") == "retry"
                    and last_observation_fingerprint
                    and observation_fp == last_observation_fingerprint
                ):
                    judge = {
                        "status": "replan",
                        "reason": "no_new_information_on_retry",
                    }

                trace = {
                    "step": step,
                    "retry_count": retry_count,
                    "failure_stage": failure_stage,
                    "step_sent": step_sent,
                    "step_returned": step_returned,
                    "step_rounds": step_rounds,
                    "llm_rounds": len(step_rounds),
                    "step_meta": step_meta,
                    "tool_events": tool_events,
                    "local_guard": local_guard,
                    "step_response_text": step_response,
                    "judge": judge,
                    "judge_sent": judge_sent,
                    "judge_returned": judge_returned,
                    "state_history": list(step_sm.history),
                }
                self.last_step_traces.append(trace)
                self.runtime_store.update_progress(
                    step_index=step_index,
                    execution_log=execution_log,
                    last_step_trace=trace,
                )

                status = judge.get("status")
                reason = str(judge.get("reason", "")).strip()
                last_observation_fingerprint = observation_fp
                trace["decision"] = {"status": status, "reason": reason or "(empty)"}

                if status == "success":
                    ok, trans_msg = step_sm.transition(StepState.DONE, "step_success")
                    if not ok:
                        return f"Task failed: {trans_msg}"
                    execution_log.append(
                        {
                            "step_id": step.get("step_id"),
                            "title": step.get("title"),
                            "status": "success",
                            "response": step_response,
                        }
                    )
                    self._emit_progress(
                        "step_done",
                        {
                            "step_id": step.get("step_id"),
                            "title": step.get("title"),
                        },
                    )
                    step_index += 1
                    self.runtime_store.update_progress(
                        step_index=step_index,
                        execution_log=execution_log,
                    )
                    break

                if status == "retry" and retry_count < max_retry:
                    retry_count += 1
                    # Fixed escalation:
                    # 1st fail -> read_error, 2nd fail -> check_env, 3rd fail -> replan.
                    if retry_count == 1:
                        failure_stage = "read_error"
                    elif retry_count == 2:
                        failure_stage = "check_env"
                    else:
                        status = "replan"
                        reason = "third_failure_switch_strategy"
                    failure_reason = reason or "judge_requested_retry"
                    self._emit_progress(
                        "step_retry",
                        {
                            "step_id": step.get("step_id"),
                            "title": step.get("title"),
                            "reason": failure_reason,
                            "retry_count": retry_count,
                        },
                    )
                    if status == "retry":
                        continue

                if status == "replan":
                    ok, trans_msg = step_sm.transition(StepState.FAIL, reason or "judge_requested_replan")
                    if not ok:
                        return f"Task failed: {trans_msg}"
                    replanned, replan_sent, replan_returned = self.mode_router.replan_remaining_via_llm(
                        user_text=user_text,
                        original_plan=plan,
                        current_step=step,
                        failure_reason=reason or "judge_requested_replan",
                        llm_client=self.client,
                    )
                    self.last_step_traces.append(
                        {
                            "type": "replan",
                            "replan_sent": replan_sent,
                            "replan_returned": replan_returned,
                            "new_plan": {
                                "task_name": replanned.task_name,
                                "steps": replanned.steps,
                                "done_criteria": replanned.done_criteria,
                                "fail_policy": replanned.fail_policy,
                            },
                        }
                    )
                    # Keep completed prefix; replace current+remaining with replanned steps.
                    steps = steps[:step_index] + replanned.steps
                    plan = TaskPlan(
                        mode=replanned.mode,
                        task_name=replanned.task_name,
                        steps=steps,
                        done_criteria=replanned.done_criteria,
                        fail_policy=replanned.fail_policy,
                    )
                    self.last_task_plan = {
                        "mode": plan.mode,
                        "task_name": plan.task_name,
                        "steps": plan.steps,
                        "done_criteria": plan.done_criteria,
                        "fail_policy": plan.fail_policy,
                    }
                    max_retry = int(plan.fail_policy.get("max_retry_per_step", max_retry) or max_retry)
                    if max_retry < 1:
                        max_retry = 1
                    if max_retry > 5:
                        max_retry = 5
                    self.runtime_store.save_active(
                        {
                            "status": "running",
                            "user_goal": user_text,
                            "plan": self.last_task_plan,
                            "step_index": step_index,
                            "execution_log": execution_log,
                        }
                    )
                    break

                execution_log.append(
                    {
                        "step_id": step.get("step_id"),
                        "title": step.get("title"),
                        "status": "fail",
                        "reason": reason or "step_failed",
                        "response": step_response,
                    }
                )
                ok, trans_msg = step_sm.transition(StepState.FAIL, reason or "step_failed")
                if not ok:
                    return f"Task failed: {trans_msg}"
                self._emit_progress(
                    "step_failed",
                    {
                        "step_id": step.get("step_id"),
                        "title": step.get("title"),
                        "reason": reason or "step_failed",
                    },
                )
                self.runtime_store.finalize(
                    status="fail",
                    summary={
                        "task_name": plan.task_name,
                        "reason": reason or "step_failed",
                        "execution_log": execution_log,
                    },
                )
                return self._render_task_result(
                    plan=plan,
                    execution_log=execution_log,
                    final_status="fail",
                )

        verify, verify_sent, verify_returned = self.mode_router.verify_done_via_llm(
            user_text=user_text,
            done_criteria=plan.done_criteria,
            execution_log=execution_log,
            llm_client=self.client,
        )
        self.last_step_traces.append(
            {
                "type": "final_verify",
                "verify": verify,
                "verify_sent": verify_sent,
                "verify_returned": verify_returned,
            }
        )
        if verify.get("status") != "pass":
            reason = str(verify.get("reason", "final_verify_failed")).strip() or "final_verify_failed"
            self._emit_progress("task_final_verify_failed", {"reason": reason})
            self.runtime_store.finalize(
                status="fail",
                summary={
                    "task_name": plan.task_name,
                    "reason": reason,
                    "execution_log": execution_log,
                },
            )
            return f"Task failed: final verification failed ({reason})"

        self._emit_progress("task_all_done", {"task_name": plan.task_name})
        self.runtime_store.finalize(
            status="done",
            summary={
                "task_name": plan.task_name,
                "done_criteria": plan.done_criteria,
                "execution_log": execution_log,
            },
        )
        return self._render_task_result(plan=plan, execution_log=execution_log, final_status="done")

    def _execute_single_step(
        self,
        *,
        user_text: str,
        plan: TaskPlan,
        current_step: dict[str, Any],
        failure_reason: str,
        failure_stage: str,
    ) -> tuple[str, dict[str, Any], dict[str, Any], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
        prompt_messages: list[dict[str, str]] = [
            self.system_message,
            {
                "role": "system",
                "content": (
                    "[TASK_MODE]\n"
                    "mode: task\n"
                    "You are executing ONE current step of a larger plan.\n"
                    "Focus on current step, produce concrete progress, and include evidence in the response.\n\n"
                    "[PROCESS_CONTRACT]\n"
                    "- For task mode, do not output only a long plan.\n"
                    "- First perform the current step using available tools.\n"
                    "- Decide next action only after tool results are observed.\n"
                    "- Do not claim DONE unless done criteria can be verified.\n\n"
                    "[EXECUTION_GRANULARITY_CONTRACT]\n"
                    "- In task mode, execute at most ONE major action per step.\n"
                    "- WRITE operations must be serialized by default.\n"
                    "- Do not jump to the next step before the current step has observable result/evidence.\n"
                    "- If current step cannot be completed, return a failure signal for retry or replan."
                ),
            },
            {
                "role": "system",
                "content": (
                    "[TASK_PLAN]\n" + json.dumps(
                        {
                            "task_name": plan.task_name,
                            "steps": plan.steps,
                            "done_criteria": plan.done_criteria,
                        },
                        ensure_ascii=False,
                    )
                ),
            },
        ]

        if failure_reason:
            prompt_messages.append(
                {
                    "role": "system",
                    "content": f"[LAST_FAILURE]\n{failure_reason}",
                }
            )
            prompt_messages.append(
                {
                    "role": "system",
                    "content": (
                        "[FAILURE_ESCALATION]\n"
                        + self._failure_stage_instruction(failure_stage)
                    ),
                }
            )

        for section in self._memory_sections_for_mode(mode="task"):
            prompt_messages.append({"role": "system", "content": section})

        prompt_messages.append(
            {
                "role": "user",
                "content": (
                    f"Overall goal: {user_text}\n"
                    f"Current step: {json.dumps(current_step, ensure_ascii=False)}\n"
                    "Execute or advance only the current step and provide verifiable evidence."
                ),
            }
        )

        messages: list[dict[str, Any]] = list(prompt_messages)
        step_rounds: list[dict[str, Any]] = []
        tool_events: list[dict[str, Any]] = []
        one_shot_intercepted = False

        for _ in range(4):
            resp = self.client.chat(
                messages=messages,
                tools=self.tool_registry.openai_tools_schema(),
                tool_choice="auto",
            )
            self.last_chat_sent_payload = self.client.last_sent_payload
            self.last_chat_returned_payload = self.client.last_returned_payload
            sent = self.client.last_sent_payload or {}
            returned = self.client.last_returned_payload or {}
            msg = self.client.extract_assistant_message(resp)
            step_rounds.append(
                {
                    "sent": sent,
                    "returned": returned,
                    "assistant_content": msg.get("content", ""),
                    "assistant_tool_calls": msg.get("tool_calls", []),
                }
            )
            calls = msg.get("tool_calls", [])
            if not calls:
                text = str(msg.get("content", "") or "").strip()
                if (not one_shot_intercepted) and self._looks_one_shot_plan(text):
                    one_shot_intercepted = True
                    messages.append(
                        {
                            "role": "system",
                            "content": (
                                "Execute the first step and return tool results before providing a full plan."
                            ),
                        }
                    )
                    continue
                return (
                    text,
                    sent,
                    returned,
                    step_rounds,
                    tool_events,
                    {
                        "one_shot_intercepted": one_shot_intercepted,
                        "max_round_reached": False,
                    },
                )

            messages.append(
                {
                    "role": "assistant",
                    "content": str(msg.get("content", "") or ""),
                    "tool_calls": [c.get("raw", {}) for c in calls],
                }
            )
            for call in calls:
                result = self.tool_registry.execute(
                    str(call.get("name", "")),
                    call.get("arguments", {}) or {},
                )
                tool_events.append(
                    {
                        "tool_call_id": str(call.get("id", "")),
                        "name": str(call.get("name", "")),
                        "arguments": call.get("arguments", {}) or {},
                        "result": result,
                    }
                )
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": str(call.get("id", "")),
                        "content": json.dumps(result, ensure_ascii=False),
                    }
                )

        return (
            "Step execution failed: exceeded maximum tool-call rounds.",
            self.client.last_sent_payload or {},
            self.client.last_returned_payload or {},
            step_rounds,
            tool_events,
            {"one_shot_intercepted": one_shot_intercepted, "max_round_reached": True},
        )

    def _build_local_guard(
        self,
        *,
        tool_events: list[dict[str, Any]],
        user_text: str,
        current_step: dict[str, Any],
    ) -> dict[str, Any]:
        total_calls = len(tool_events)
        ok_calls = 0
        needs_user_calls = 0
        error_calls = 0
        ok_web_calls = 0
        ok_web_action_calls = 0
        physical_browser_opened = False
        for ev in tool_events:
            result = ev.get("result", {}) if isinstance(ev, dict) else {}
            status = str(result.get("status", "")).strip().lower()
            tool_name = str(ev.get("name", "")).strip().lower()
            if status == "ok":
                ok_calls += 1
                if tool_name in {"web_scan", "web_execute_js"}:
                    ok_web_calls += 1
                if tool_name == "web_execute_js":
                    output = result.get("output", {}) if isinstance(result.get("output"), dict) else {}
                    action = str(output.get("action", "")).strip().lower()
                    browser_opened = bool(output.get("browser_opened", False))
                    if action in {"navigate", "open_new_tab"}:
                        ok_web_action_calls += 1
                    if browser_opened:
                        physical_browser_opened = True
            elif status == "needs_user":
                needs_user_calls += 1
            else:
                error_calls += 1
        combined_text = f"{user_text}\n{json.dumps(current_step, ensure_ascii=False)}".lower()
        web_intent = any(
            k in combined_text
            for k in ["website", "webpage", "browser", "http://", "https://", "www.", "web", "url"]
        )
        # hard success requirement for executable steps:
        # must have at least one tool call and at least one ok result.
        requirement_met = total_calls > 0 and ok_calls > 0
        if web_intent:
            requirement_met = requirement_met and (
                ok_web_action_calls > 0 or physical_browser_opened
            )
        return {
            "requirement_met": requirement_met,
            "total_tool_calls": total_calls,
            "ok_calls": ok_calls,
            "ok_web_calls": ok_web_calls,
            "ok_web_action_calls": ok_web_action_calls,
            "physical_browser_opened": physical_browser_opened,
            "needs_user_calls": needs_user_calls,
            "error_calls": error_calls,
            "web_intent": web_intent,
        }

    def _memory_sections_for_mode(self, *, mode: str) -> list[str]:
        sections = self.memory.build_memory_system_sections()
        if not sections:
            return []
        # Memory blocks are always in fixed order from MemoryStore.
        # Keep smaller budgets for simple mode and larger for task mode.
        per_section_cap = (
            [900, 700, 900, 900, 700]
            if mode == "simple"
            else [1200, 900, 1200, 1200, 1000]
        )
        compact: list[str] = []
        for idx, raw in enumerate(sections):
            cap = per_section_cap[idx] if idx < len(per_section_cap) else 800
            compact.append(self._truncate_text(raw, cap))
        return compact

    @staticmethod
    def _truncate_text(text: str, max_chars: int) -> str:
        t = str(text or "")
        if max_chars < 64:
            max_chars = 64
        if len(t) <= max_chars:
            return t
        return t[: max_chars - 1] + "…"

    def _observation_fingerprint(self, *, tool_events: list[dict[str, Any]], step_response: str) -> str:
        # Deterministic compact fingerprint used for "no new info" retry blocking.
        pairs: list[str] = []
        for ev in tool_events:
            result = ev.get("result", {}) if isinstance(ev, dict) else {}
            pairs.append(
                "|".join(
                    [
                        str(ev.get("name", "")),
                        str(result.get("status", "")),
                        str(result.get("error", ""))[:120],
                    ]
                )
            )
        pairs.append(str(step_response or "")[:240])
        return "||".join(pairs)

    @staticmethod
    def _failure_stage_instruction(stage: str) -> str:
        s = str(stage or "").strip().lower()
        if s == "check_env":
            return (
                "Second failure stage: inspect environment state first "
                "(cwd/files/process/runtime), then attempt a revised action."
            )
        if s == "switch_strategy":
            return (
                "Third failure stage: switch strategy and avoid repeating the same action path."
            )
        return (
            "First failure stage: read and analyze concrete error details before retrying."
        )

    @staticmethod
    def _looks_one_shot_plan(text: str) -> bool:
        t = str(text or "")
        if len(t) < 500:
            return False
        if "```" in t:
            return True
        long_lines = sum(1 for line in t.splitlines() if len(line.strip()) > 80)
        return long_lines >= 5

    def _render_task_result(
        self,
        *,
        plan: TaskPlan,
        execution_log: list[dict[str, Any]],
        final_status: str,
    ) -> str:
        if final_status == "done":
            return "Done."

        # Compact failure summary.
        failed = None
        for item in reversed(execution_log):
            if item.get("status") != "success":
                failed = item
                break
        if failed is None:
            return "Task failed."
        return (
            f"Task failed at step {failed.get('step_id')}: "
            f"{failed.get('title')} ({failed.get('reason', 'unknown_reason')})"
        )
