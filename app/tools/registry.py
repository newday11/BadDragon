"""Tool registry for BadDragon (minimal v1)."""

from __future__ import annotations

from typing import Any, Callable

from . import ask_user as ask_user_tool
from . import code_run as code_run_tool
from . import file_patch as file_patch_tool
from . import file_read as file_read_tool
from . import file_write as file_write_tool
from . import web_execute_js as web_execute_js_tool
from . import web_search as web_search_tool
from . import web_scan as web_scan_tool


ToolFunc = Callable[[dict[str, Any]], dict[str, Any]]


class ToolRegistry:
    def __init__(self) -> None:
        self._handlers: dict[str, ToolFunc] = {
            "code_run": code_run_tool.run,
            "file_read": file_read_tool.run,
            "file_patch": file_patch_tool.run,
            "file_write": file_write_tool.run,
            "web_scan": web_scan_tool.run,
            "web_execute_js": web_execute_js_tool.run,
            "web_search": web_search_tool.run,
            "ask_user": ask_user_tool.run,
        }

    def openai_tools_schema(self) -> list[dict[str, Any]]:
        return [
            {
                "type": "function",
                "function": {
                    "name": "code_run",
                    "description": "Execute python or bash scripts.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "type": {
                                "type": "string",
                                "enum": ["python", "bash"],
                                "description": "Execution type.",
                                "default": "python",
                            },
                            "script": {
                                "type": "string",
                                "description": "Script content to execute.",
                            },
                            "timeout": {
                                "type": "integer",
                                "description": "Timeout in seconds.",
                                "default": 60,
                            },
                            "cwd": {
                                "type": "string",
                                "description": "Optional working directory.",
                            },
                        },
                        "required": ["script"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "file_read",
                    "description": "Read file content with optional line window.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "path": {"type": "string", "description": "File path."},
                            "start": {"type": "integer", "description": "Start line (1-based)."},
                            "count": {"type": "integer", "description": "Number of lines to read."},
                        },
                        "required": ["path"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "ask_user",
                    "description": "Ask user for clarification or decision.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "question": {"type": "string", "description": "Question text."},
                            "candidates": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "Optional candidate answers.",
                            },
                        },
                        "required": ["question"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "file_patch",
                    "description": "Patch file by replacing exactly one unique old_content block.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "path": {"type": "string", "description": "File path."},
                            "old_content": {"type": "string", "description": "Original unique text block."},
                            "new_content": {"type": "string", "description": "Replacement text block."},
                        },
                        "required": ["path", "old_content", "new_content"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "file_write",
                    "description": "Write file content with overwrite/append/prepend mode.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "path": {"type": "string", "description": "File path."},
                            "mode": {
                                "type": "string",
                                "enum": ["overwrite", "append", "prepend"],
                                "default": "overwrite",
                            },
                            "content": {"type": "string", "description": "Content to write."},
                        },
                        "required": ["path", "content"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "web_scan",
                    "description": "Get current web tab info and simplified html/text.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "tabs_only": {"type": "boolean", "default": False},
                            "switch_tab_id": {"type": "string"},
                            "text_only": {"type": "boolean", "default": False},
                        },
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "web_search",
                    "description": "Search the web and return top result links/snippets.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {"type": "string", "description": "Search query text."},
                            "max_results": {
                                "type": "integer",
                                "description": "Max result count (1-10).",
                                "default": 5,
                            },
                            "timeout": {
                                "type": "integer",
                                "description": "Request timeout seconds.",
                                "default": 20,
                            },
                        },
                        "required": ["query"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "web_execute_js",
                    "description": "Execute minimal browser JS actions (navigate/open tab/switch tab).",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "script": {"type": "string"},
                            "save_to_file": {"type": "string"},
                            "no_monitor": {"type": "boolean"},
                            "switch_tab_id": {"type": "string"},
                        },
                    },
                },
            },
        ]

    def execute(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        handler = self._handlers.get(name)
        if handler is None:
            return self._error(name, f"unknown tool: {name}")
        try:
            result = handler(arguments or {})
            if isinstance(result, dict) and "status" in result:
                return result
            return {
                "status": "ok",
                "output": result,
                "error": "",
                "artifacts": [],
                "meta": {"tool": name},
            }
        except Exception as exc:
            return self._error(name, f"tool execution exception: {exc}")

    @staticmethod
    def _error(name: str, message: str) -> dict[str, Any]:
        return {
            "status": "error",
            "output": {},
            "error": str(message),
            "artifacts": [],
            "meta": {"tool": name},
        }
