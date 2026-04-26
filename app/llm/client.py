"""LLM gateway for OpenAI-compatible chat completions."""

from __future__ import annotations

import json
from typing import Any
from urllib import request

from app.llm.runtime_provider import RuntimeProvider, resolve_runtime_provider


class LLMClient:
    """Thin wrapper around OpenAI-compatible API."""

    def __init__(self, runtime: RuntimeProvider | None = None) -> None:
        self.runtime = runtime or resolve_runtime_provider()
        self.last_sent_payload: dict[str, Any] | None = None
        self.last_returned_payload: dict[str, Any] | None = None
        self.total_chat_calls = 0
        self._backend = "openai"
        try:
            from openai import OpenAI
        except ModuleNotFoundError as exc:
            self._backend = "http"
            self._client = None
        else:
            self._client = OpenAI(
                api_key=self.runtime.api_key,
                base_url=self.runtime.base_url,
            )

    def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | dict[str, Any] | None = None,
        model: str | None = None,
        temperature: float | int | None = None,
        max_tokens: int | None = None,
    ) -> Any:
        kwargs: dict[str, Any] = {
            "model": model or self.runtime.model,
            "messages": messages,
            "temperature": self.runtime.temperature if temperature is None else temperature,
            "max_tokens": self.runtime.max_tokens if max_tokens is None else max_tokens,
        }
        if tools:
            kwargs["tools"] = tools
        if tool_choice is not None:
            kwargs["tool_choice"] = tool_choice
        self.last_sent_payload = dict(kwargs)
        self.total_chat_calls += 1
        if self._backend == "openai":
            resp = self._client.chat.completions.create(**kwargs)
            try:
                self.last_returned_payload = resp.model_dump()
            except Exception:
                self.last_returned_payload = {"raw": str(resp)}
            return resp
        resp = self._chat_via_http(kwargs)
        self.last_returned_payload = resp
        return resp

    @staticmethod
    def extract_text(resp: Any) -> str:
        """Extract assistant text from either OpenAI SDK object or HTTP dict."""
        if isinstance(resp, dict):
            return (
                resp.get("choices", [{}])[0]
                .get("message", {})
                .get("content", "")
            ) or ""
        try:
            return resp.choices[0].message.content or ""
        except Exception:
            return ""

    @staticmethod
    def extract_assistant_message(resp: Any) -> dict[str, Any]:
        """Return normalized assistant message:
        {"role":"assistant","content":"...","tool_calls":[{"id","name","arguments","raw"}]}
        """
        if isinstance(resp, dict):
            message = (resp.get("choices", [{}])[0].get("message", {}) or {})
            content = str(message.get("content") or "")
            raw_tool_calls = message.get("tool_calls") or []
            return {
                "role": "assistant",
                "content": content,
                "tool_calls": LLMClient._normalize_tool_calls(raw_tool_calls),
            }
        try:
            message = resp.choices[0].message
            content = str(message.content or "")
            raw_tool_calls = message.tool_calls or []
            return {
                "role": "assistant",
                "content": content,
                "tool_calls": LLMClient._normalize_tool_calls(raw_tool_calls),
            }
        except Exception:
            return {"role": "assistant", "content": "", "tool_calls": []}

    @staticmethod
    def _normalize_tool_calls(raw_tool_calls: Any) -> list[dict[str, Any]]:
        items: list[dict[str, Any]] = []
        if not isinstance(raw_tool_calls, list):
            return items
        for idx, tc in enumerate(raw_tool_calls, start=1):
            call_id = ""
            name = ""
            args_json = "{}"
            if isinstance(tc, dict):
                call_id = str(tc.get("id") or f"call_{idx}")
                fn = tc.get("function") or {}
                name = str(fn.get("name") or "")
                args_json = str(fn.get("arguments") or "{}")
            else:
                # OpenAI SDK object
                try:
                    call_id = str(getattr(tc, "id", "") or f"call_{idx}")
                    fn = getattr(tc, "function", None)
                    name = str(getattr(fn, "name", "") or "")
                    args_json = str(getattr(fn, "arguments", "") or "{}")
                except Exception:
                    pass
            if not name:
                continue
            try:
                args_obj = json.loads(args_json)
                if not isinstance(args_obj, dict):
                    args_obj = {}
            except Exception:
                args_obj = {}
            items.append(
                {
                    "id": call_id,
                    "name": name,
                    "arguments": args_obj,
                    "raw": {
                        "id": call_id,
                        "type": "function",
                        "function": {
                            "name": name,
                            "arguments": json.dumps(args_obj, ensure_ascii=False),
                        },
                    },
                }
            )
        return items

    def _chat_via_http(self, kwargs: dict[str, Any]) -> dict[str, Any]:
        url = f"{self.runtime.base_url.rstrip('/')}/chat/completions"
        req = request.Request(
            url=url,
            data=json.dumps(kwargs, ensure_ascii=False).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.runtime.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        with request.urlopen(req, timeout=120) as resp:
            return json.loads(resp.read().decode("utf-8", errors="replace"))
