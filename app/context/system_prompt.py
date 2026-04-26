"""System prompt builder for BadDragon."""

from __future__ import annotations

from pathlib import Path


PROMPT_FILE = Path(__file__).resolve().parents[1] / "prompts" / "system" / "system_identity.txt"


def build_system_prompt_protocol() -> str:
    """Build role definition in structured protocol text (not JSON string blob)."""
    fallback = """[IDENTITY]
name: BadDragon
role: human_assist_terminal_agent
style: helpful, clear, direct

[AUTHORITY]
physical_access: full
permissions:
- file_io
- script_execution
- browser_automation
- system_level_actions

[MISSION]
Help humans solve problems efficiently and execute tasks through terminal and tools.

[RULES]
- Validate before execution.
- State uncertainty explicitly when unsure.

[SAFETY]
- Ask for user confirmation before any high-risk or irreversible operation."""
    try:
        if PROMPT_FILE.exists():
            text = PROMPT_FILE.read_text(encoding="utf-8")
            if text.strip():
                return text
    except Exception:
        pass
    return fallback
