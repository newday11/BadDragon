"""Orchestrator loop with mode routing and task-step execution."""

from __future__ import annotations

import json
import re
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
        route_calls = 1
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
        }

        self._progress_callback = None
        return content

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

                deterministic = self._deterministic_step_decision(
                    current_step=step,
                    step_meta=step_meta,
                    tool_events=tool_events,
                    local_guard=local_guard,
                )
                if deterministic is not None:
                    judge = deterministic
                    judge_sent = {"deterministic": True}
                    judge_returned = {"deterministic": True}
                else:
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
        web_profile = self._is_web_observe_step(user_text=user_text, current_step=current_step)
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
        if web_profile:
            prompt_messages.append(
                {
                    "role": "system",
                    "content": (
                        "[WEB_EXECUTION_POLICY]\n"
                        "- For website-identification steps, use a bounded acquisition flow.\n"
                        "- At most ONE navigate/open_new_tab action and ONE web_scan call.\n"
                        "- Stop tool calls once title/text evidence is available.\n"
                        "- If evidence is still insufficient after the budget, return failure for replan.\n"
                        "- Never repeat an identical tool call with identical arguments."
                    ),
                }
            )

        messages: list[dict[str, Any]] = list(prompt_messages)
        step_rounds: list[dict[str, Any]] = []
        tool_events: list[dict[str, Any]] = []
        one_shot_intercepted = False
        seen_tool_calls: set[str] = set()
        duplicate_tool_blocked = False
        duplicate_reasons: list[str] = []
        web_budget = {"navigate": 0, "scan": 0}
        web_evidence: dict[str, Any] = {"url": "", "title": "", "text_len": 0, "has_content": False}

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
                        "web_profile": web_profile,
                        "duplicate_tool_blocked": duplicate_tool_blocked,
                        "duplicate_reasons": duplicate_reasons,
                        "web_budget": dict(web_budget),
                        "web_evidence": web_evidence,
                        "web_evidence_ready": self._web_evidence_ready(web_budget=web_budget, web_evidence=web_evidence),
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
                name = str(call.get("name", "")).strip()
                args = call.get("arguments", {}) or {}
                call_sig = self._tool_call_signature(name=name, arguments=args)
                if call_sig in seen_tool_calls:
                    duplicate_tool_blocked = True
                    duplicate_reasons.append(f"duplicate:{name}")
                    result = self._blocked_tool_result("duplicate_tool_call_blocked")
                    tool_events.append(
                        {
                            "tool_call_id": str(call.get("id", "")),
                            "name": name,
                            "arguments": args,
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
                    continue
                seen_tool_calls.add(call_sig)

                if web_profile:
                    allowed, blocked_reason = self._allow_web_tool_call(
                        name=name,
                        arguments=args,
                        web_budget=web_budget,
                    )
                    if not allowed:
                        duplicate_tool_blocked = True
                        duplicate_reasons.append(blocked_reason)
                        result = self._blocked_tool_result(blocked_reason)
                        tool_events.append(
                            {
                                "tool_call_id": str(call.get("id", "")),
                                "name": name,
                                "arguments": args,
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
                        continue

                result = self.tool_registry.execute(
                    name,
                    args,
                )
                if web_profile:
                    self._consume_web_tool_result(
                        name=name,
                        result=result,
                        web_evidence=web_evidence,
                    )
                tool_events.append(
                    {
                        "tool_call_id": str(call.get("id", "")),
                        "name": name,
                        "arguments": args,
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
            if web_profile and self._web_evidence_ready(web_budget=web_budget, web_evidence=web_evidence):
                return (
                    self._render_web_evidence_summary(web_evidence=web_evidence),
                    sent,
                    returned,
                    step_rounds,
                    tool_events,
                    {
                        "one_shot_intercepted": one_shot_intercepted,
                        "max_round_reached": False,
                        "web_profile": web_profile,
                        "duplicate_tool_blocked": duplicate_tool_blocked,
                        "duplicate_reasons": duplicate_reasons,
                        "web_budget": dict(web_budget),
                        "web_evidence": web_evidence,
                        "web_evidence_ready": True,
                        "deterministic_stop_reason": "web_evidence_ready",
                    },
                )

        return (
            "Step execution failed: exceeded maximum tool-call rounds.",
            self.client.last_sent_payload or {},
            self.client.last_returned_payload or {},
            step_rounds,
            tool_events,
            {
                "one_shot_intercepted": one_shot_intercepted,
                "max_round_reached": True,
                "web_profile": web_profile,
                "duplicate_tool_blocked": duplicate_tool_blocked,
                "duplicate_reasons": duplicate_reasons,
                "web_budget": dict(web_budget),
                "web_evidence": web_evidence,
                "web_evidence_ready": self._web_evidence_ready(web_budget=web_budget, web_evidence=web_evidence),
            },
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
        user_text_l = str(user_text or "").lower()
        step_text_l = json.dumps(current_step, ensure_ascii=False).lower()
        combined_text = f"{user_text_l}\n{step_text_l}"
        web_intent = any(
            k in combined_text
            for k in ["website", "webpage", "browser", "http://", "https://", "www.", "web", "url", "网站", "网页"]
        )
        nav_intent = any(
            k in step_text_l
            for k in ["navigate", "open", "visit", "go to", "打开", "访问", "进入", "跳转"]
        )
        # hard success requirement for executable steps:
        # must have at least one tool call and at least one ok result.
        requirement_met = total_calls > 0 and ok_calls > 0
        if web_intent:
            # For web-related steps, at least one successful web tool call is required.
            requirement_met = requirement_met and (ok_web_calls > 0)
        if nav_intent:
            # Only navigation-like steps require explicit navigation evidence.
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
            "nav_intent": nav_intent,
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

    def _deterministic_step_decision(
        self,
        *,
        current_step: dict[str, Any],
        step_meta: dict[str, Any],
        tool_events: list[dict[str, Any]],
        local_guard: dict[str, Any],
    ) -> dict[str, Any] | None:
        if bool(step_meta.get("max_round_reached", False)):
            return {"status": "replan", "reason": "max_round_reached"}
        if bool(step_meta.get("duplicate_tool_blocked", False)):
            return {"status": "replan", "reason": "duplicate_or_budget_blocked"}

        if bool(step_meta.get("web_profile", False)):
            evidence_ready = bool(step_meta.get("web_evidence_ready", False))
            if evidence_ready and bool(local_guard.get("requirement_met", False)):
                return {"status": "success", "reason": "deterministic_web_evidence_ready"}
            web_budget = step_meta.get("web_budget", {}) if isinstance(step_meta.get("web_budget"), dict) else {}
            nav_used = int(web_budget.get("navigate", 0) or 0)
            scan_used = int(web_budget.get("scan", 0) or 0)
            if nav_used >= 1 and scan_used >= 1 and not evidence_ready:
                return {"status": "replan", "reason": "web_budget_exhausted_without_evidence"}

        # Deterministic fallback: if no valid tool evidence, do not ask LLM judge.
        has_ok_tool = any(
            str((item.get("result", {}) if isinstance(item, dict) else {}).get("status", "")).strip().lower() == "ok"
            for item in tool_events
        )
        if not bool(local_guard.get("requirement_met", False)) and not has_ok_tool:
            step_text = json.dumps(current_step, ensure_ascii=False).lower()
            web_like = any(k in step_text for k in ["http://", "https://", "website", "网站", "网页", "web"])
            if web_like:
                return {"status": "replan", "reason": "no_valid_tool_evidence_web_step"}
            return {"status": "retry", "reason": "no_valid_tool_evidence"}

        return None

    @staticmethod
    def _tool_call_signature(*, name: str, arguments: dict[str, Any]) -> str:
        normalized_name = str(name or "").strip().lower()
        try:
            normalized_args = json.dumps(arguments or {}, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        except Exception:
            normalized_args = str(arguments or {})
        return f"{normalized_name}|{normalized_args}"

    @staticmethod
    def _blocked_tool_result(reason: str) -> dict[str, Any]:
        return {
            "status": "error",
            "output": {"blocked": True, "reason": reason},
            "error": reason,
            "artifacts": [],
            "meta": {"tool": "controller_guard"},
        }

    @staticmethod
    def _is_web_observe_step(*, user_text: str, current_step: dict[str, Any]) -> bool:
        combined = f"{user_text}\n{json.dumps(current_step, ensure_ascii=False)}".lower()
        return any(
            k in combined
            for k in [
                "http://",
                "https://",
                "www.",
                "website",
                "webpage",
                "site",
                "url",
                "浏览器",
                "网站",
                "网页",
                "访问",
                "打开",
            ]
        )

    def _allow_web_tool_call(
        self,
        *,
        name: str,
        arguments: dict[str, Any],
        web_budget: dict[str, int],
    ) -> tuple[bool, str]:
        tool_name = str(name or "").strip().lower()
        if tool_name == "web_scan":
            if int(web_budget.get("scan", 0) or 0) >= 1:
                return False, "web_scan_budget_exhausted"
            web_budget["scan"] = int(web_budget.get("scan", 0) or 0) + 1
            return True, ""
        if tool_name == "web_execute_js":
            script = str(arguments.get("script", "") or "")
            if self._is_navigation_script(script):
                if int(web_budget.get("navigate", 0) or 0) >= 1:
                    return False, "navigate_budget_exhausted"
                web_budget["navigate"] = int(web_budget.get("navigate", 0) or 0) + 1
            return True, ""
        return True, ""

    @staticmethod
    def _is_navigation_script(script: str) -> bool:
        s = str(script or "").strip().lower()
        if not s:
            return False
        return any(k in s for k in ["location.href", "window.location", "window.open", "navigate("])

    @staticmethod
    def _consume_web_tool_result(*, name: str, result: dict[str, Any], web_evidence: dict[str, Any]) -> None:
        tool_name = str(name or "").strip().lower()
        if not isinstance(result, dict):
            return
        output = result.get("output", {}) if isinstance(result.get("output"), dict) else {}
        if tool_name == "web_execute_js":
            url = str(output.get("url", "") or "").strip()
            title = str(output.get("title", "") or "").strip()
            if url:
                web_evidence["url"] = url
            if title and title.lower() != "about:blank":
                web_evidence["title"] = title
            return
        if tool_name == "web_scan":
            url = str(output.get("url", "") or "").strip()
            title = str(output.get("title", "") or "").strip()
            text = str(output.get("text", "") or "")
            html = str(output.get("html", "") or "")
            if url:
                web_evidence["url"] = url
            if title and title.lower() != "about:blank":
                web_evidence["title"] = title
            text_len = len(text.strip()) if text.strip() else 0
            if text_len <= 0 and html.strip():
                # Rough HTML-to-text signal for evidence readiness.
                compact = re.sub(r"<[^>]+>", " ", html)
                text_len = len(re.sub(r"\s+", " ", compact).strip())
            web_evidence["text_len"] = max(int(web_evidence.get("text_len", 0) or 0), int(text_len))
            web_evidence["has_content"] = bool(web_evidence["text_len"] >= 80)

    @staticmethod
    def _web_evidence_ready(*, web_budget: dict[str, int], web_evidence: dict[str, Any]) -> bool:
        nav_used = int(web_budget.get("navigate", 0) or 0) >= 1
        scan_used = int(web_budget.get("scan", 0) or 0) >= 1
        has_url = bool(str(web_evidence.get("url", "") or "").strip())
        has_title = bool(str(web_evidence.get("title", "") or "").strip())
        text_len = int(web_evidence.get("text_len", 0) or 0)
        has_content = bool(web_evidence.get("has_content", False)) or text_len >= 80
        return nav_used and scan_used and has_url and has_title and has_content

    @staticmethod
    def _render_web_evidence_summary(*, web_evidence: dict[str, Any]) -> str:
        url = str(web_evidence.get("url", "") or "").strip()
        title = str(web_evidence.get("title", "") or "").strip()
        text_len = int(web_evidence.get("text_len", 0) or 0)
        return (
            "Web evidence collected.\n"
            f"- url: {url or '(empty)'}\n"
            f"- title: {title or '(empty)'}\n"
            f"- text_len: {text_len}\n"
            "Proceed to conclude this step based on the captured page content."
        )

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
            summary = self._extract_task_result_summary(plan=plan, execution_log=execution_log)
            if summary:
                return summary
            return "Task completed."

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

    @staticmethod
    def _extract_task_result_summary(*, plan: TaskPlan, execution_log: list[dict[str, Any]]) -> str:
        """Extract a user-facing completion answer from task execution logs."""
        if not execution_log:
            return ""

        keyword_hits = (
            "总结",
            "summary",
            "结论",
            "final",
            "功能",
            "用途",
            "一句话",
            "is a",
            "是一个",
        )

        candidate = ""
        for item in reversed(execution_log):
            if str(item.get("status", "")).strip().lower() != "success":
                continue
            title = str(item.get("title", "") or "").lower()
            response = str(item.get("response", "") or "").strip()
            if not response:
                continue
            if any(k in title for k in keyword_hits) or any(k in response.lower() for k in keyword_hits):
                candidate = response
                break
            if not candidate:
                candidate = response

        if not candidate:
            return ""

        text = candidate.replace("**Evidence:**", "").strip()
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        if not lines:
            return ""

        # Prefer a concise conclusion-like line.
        for line in reversed(lines):
            low = line.lower()
            if len(line) < 12:
                continue
            if "step " in low and "completed" in low:
                continue
            if any(k in low for k in keyword_hits):
                return line

        # Fallback to the last meaningful line.
        for line in reversed(lines):
            low = line.lower()
            if "step " in low and "completed" in low:
                continue
            return line
        return ""
