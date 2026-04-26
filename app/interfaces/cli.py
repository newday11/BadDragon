"""CLI helpers for quick BadDragon smoke tests."""

from __future__ import annotations

import json
from pathlib import Path
from urllib import request

from app.context.system_prompt import build_system_prompt_protocol
from app.llm.runtime_provider import resolve_runtime_provider
from app.orchestrator.agent_loop import SimpleAgentLoop


def run_hello_and_capture() -> dict:
    """Send one hello message and capture full request/response payloads."""
    runtime = resolve_runtime_provider()
    if runtime.api_mode != "chat_completions":
        raise ValueError(f"Only chat_completions is supported in this demo, got {runtime.api_mode}")

    payload = {
        "model": runtime.model,
        "messages": [
            {"role": "system", "content": build_system_prompt_protocol()},
            {"role": "user", "content": "hello"},
        ],
        "temperature": runtime.temperature,
        "max_tokens": runtime.max_tokens,
    }

    url = f"{runtime.base_url.rstrip('/')}/chat/completions"
    req = request.Request(
        url=url,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {runtime.api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    with request.urlopen(req, timeout=120) as resp:
        response_body = resp.read().decode("utf-8", errors="replace")

    # Keep logs near where user runs the command, easier to inspect.
    logs_dir = Path.cwd() / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    sent_path = logs_dir / "last_sent_to_llm.json"
    recv_path = logs_dir / "last_returned_from_llm.json"
    text_path = logs_dir / "last_returned_text.txt"

    sent_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    recv_path.write_text(response_body + "\n", encoding="utf-8")

    text_out = ""
    try:
        parsed = json.loads(response_body)
        text_out = (
            parsed.get("choices", [{}])[0]
            .get("message", {})
            .get("content", "")
        )
    except Exception:
        text_out = ""
    text_path.write_text(text_out + "\n", encoding="utf-8")

    return {
        "url": url,
        "sent_path": str(sent_path),
        "recv_path": str(recv_path),
        "text_path": str(text_path),
    }


def run_terminal_chat() -> None:
    """Run minimal terminal chat loop.

    Format:
      > hello
      AI: Hello! ...
      > exit
    """
    run_terminal_chat_with_debug(debug_io=False, debug_lite=False)


def run_terminal_chat_with_debug(debug_io: bool = False, debug_lite: bool = False) -> None:
    """Run minimal terminal chat loop with optional I/O payload debug."""
    # In debug mode we run memory updates synchronously so request/response payloads
    # are available immediately in the same turn.
    loop = SimpleAgentLoop(async_memory_updates=not (debug_io or debug_lite))
    def on_progress(event: str, payload: dict) -> None:
        if event == "task_plan":
            for step in payload.get("steps", []) or []:
                step_id = step.get("step_id")
                title = step.get("title", "")
                print(f"Step {step_id}: {title}", flush=True)
            return
        if event == "step_done":
            print(f"Step {payload.get('step_id')} (DONE)", flush=True)
            return
        if event == "step_failed":
            print(f"Step {payload.get('step_id')} (FAILED)", flush=True)
            return
        if event == "task_all_done":
            print("ALL STEPS COMPLETED", flush=True)
            return
        if event == "task_final_verify_failed":
            print("FINAL VERIFY FAILED", flush=True)
            return

    while True:
        try:
            user_text = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not user_text:
            continue
        if user_text.lower() == "exit":
            break

        answer = loop.ask(user_text, progress_callback=on_progress)
        if debug_io:
            mode = loop.last_mode_decision
            print("[Task Mode]")
            print(
                json.dumps(
                    {"mode": mode.mode, "source": mode.source, "reason": mode.reason},
                    ensure_ascii=False,
                    indent=2,
                )
            )
            print("[Mode Classifier Sent]")
            print(json.dumps(loop.last_mode_sent_payload or {}, ensure_ascii=False, indent=2))
            print("[Mode Classifier Returned]")
            print(json.dumps(loop.last_mode_returned_payload or {}, ensure_ascii=False, indent=2))
            if loop.last_plan_sent_payload is not None:
                print("[Task Planner Sent]")
                print(json.dumps(loop.last_plan_sent_payload or {}, ensure_ascii=False, indent=2))
                print("[Task Planner Returned]")
                print(json.dumps(loop.last_plan_returned_payload or {}, ensure_ascii=False, indent=2))
            if loop.last_step_traces:
                print("[Task Step Traces]")
                print(json.dumps(loop.last_step_traces, ensure_ascii=False, indent=2))
            sent = loop.last_chat_sent_payload or {}
            returned = loop.last_chat_returned_payload or {}
            print("[Sent To LLM]")
            print(json.dumps(sent, ensure_ascii=False, indent=2))
            print("[Returned From LLM]")
            print(json.dumps(returned, ensure_ascii=False, indent=2))
            if loop.last_owner_memory_refresh_success:
                print("[Owner Memory Update Sent]")
                print(json.dumps(loop.memory.last_owner_sent_payload or {}, ensure_ascii=False, indent=2))
                print("[Owner Memory Update Returned]")
                print(json.dumps(loop.memory.last_owner_returned_payload or {}, ensure_ascii=False, indent=2))
            if loop.last_project_memory_refresh_success:
                print("[Project Memory Update Sent]")
                print(json.dumps(loop.memory.last_project_sent_payload or {}, ensure_ascii=False, indent=2))
                print("[Project Memory Update Returned]")
                print(json.dumps(loop.memory.last_project_returned_payload or {}, ensure_ascii=False, indent=2))
        elif debug_lite:
            s = loop.last_turn_llm_stats or {}
            print("[LLM Calls]", flush=True)
            print(
                f"total={s.get('total', 0)} "
                f"(route={s.get('route', 0)}, simple={s.get('simple_answer', 0)}, "
                f"task_execute={s.get('task_execute', 0)}, task_judge={s.get('task_judge', 0)}, "
                f"task_final_verify={s.get('task_final_verify', 0)}, "
                f"task_replan={s.get('task_replan', 0)}, planner={s.get('planner', 0)}, "
                f"owner_memory={s.get('owner_memory', 0)}, project_memory={s.get('project_memory', 0)}, "
                f"other={s.get('other', 0)})",
                flush=True,
            )
            step_traces = [x for x in (loop.last_step_traces or []) if isinstance(x, dict) and "step" in x]
            if step_traces:
                last = step_traces[-1]
                failure_stage = str(last.get("failure_stage", ""))
                decision = last.get("decision", {}) if isinstance(last.get("decision"), dict) else {}
                d_status = str(decision.get("status", ""))
                d_reason = str(decision.get("reason", ""))
                print(
                    f"[Task Decision] status={d_status or '-'} stage={failure_stage or '-'} reason={d_reason or '-'}",
                    flush=True,
                )
        print(f"AI: {answer}", flush=True)
        if loop.last_owner_memory_refresh_success:
            print("=================OWNER MEMORY EVOLUTION SUCCEEDED====================", flush=True)
        if loop.last_project_memory_refresh_success:
            print("=================PROJECT MEMORY EVOLUTION SUCCEEDED====================", flush=True)
