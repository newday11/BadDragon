"""web_execute_js tool.

Minimal implementation:
- Supports navigation-like JS intents:
  - location.href = "https://..."
  - window.location = "https://..."
  - window.open("https://...")
  - navigate("https://...")
- Supports tab switching via switch_tab_id argument.
"""

from __future__ import annotations

import re
import webbrowser
from typing import Any

from .web_state import SESSION, save_json_if_needed


URL_PATTERNS = [
    re.compile(r"""location\.href\s*=\s*['"]([^'"]+)['"]""", re.IGNORECASE),
    re.compile(r"""window\.location\s*=\s*['"]([^'"]+)['"]""", re.IGNORECASE),
    re.compile(r"""navigate\s*\(\s*['"]([^'"]+)['"]\s*\)""", re.IGNORECASE),
]
OPEN_PATTERN = re.compile(
    r"""window\.open\s*\(\s*['"]([^'"]+)['"](?:\s*,\s*['"][^'"]*['"])?\s*\)""",
    re.IGNORECASE,
)


def run(arguments: dict[str, Any]) -> dict[str, Any]:
    script = str(arguments.get("script", "") or "")
    switch_tab_id = str(arguments.get("switch_tab_id", "") or "").strip() or None
    save_to_file = str(arguments.get("save_to_file", "") or "").strip() or None

    if switch_tab_id:
        SESSION.ensure_tab(switch_tab_id)

    action = "none"
    browser_opened = False
    tab = SESSION.ensure_tab(switch_tab_id)

    m_open = OPEN_PATTERN.search(script)
    needs_physical_open = False
    if m_open:
        target = m_open.group(1)
        tab = SESSION.open_new_tab(target)
        browser_opened = _try_open_system_browser(target)
        action = "open_new_tab"
        needs_physical_open = True
    else:
        url = _extract_nav_url(script)
        if url:
            tab = SESSION.navigate(url, switch_tab_id=switch_tab_id)
            browser_opened = _try_open_system_browser(url)
            action = "navigate"
            needs_physical_open = True

    payload = {
        "action": action,
        "current_tab_id": tab.tab_id,
        "url": tab.url,
        "title": tab.title,
        "tabs": SESSION.get_tabs(),
        "browser_opened": browser_opened,
    }
    saved = save_json_if_needed(save_to_file, payload)
    artifacts = []
    if saved:
        artifacts.append({"type": "file", "path": saved})

    status = "ok"
    error = ""
    if needs_physical_open and not browser_opened:
        # Headless/server environments often cannot launch a physical browser.
        # Keep navigation as successful when in-memory tab state is updated.
        payload["warning"] = "physical browser did not open (headless/server mode)"

    return {
        "status": status,
        "output": payload,
        "error": error,
        "artifacts": artifacts,
        "meta": {"tool": "web_execute_js"},
    }


def _extract_nav_url(script: str) -> str | None:
    s = str(script or "")
    for p in URL_PATTERNS:
        m = p.search(s)
        if m:
            return m.group(1)
    return None


def _try_open_system_browser(url: str) -> bool:
    target = str(url or "").strip()
    if not target:
        return False
    # URL scheme RFC-like check; keep '-' at the end to avoid regex char-range issues.
    if not re.match(r"^[a-zA-Z][a-zA-Z0-9+.-]*://", target):
        target = f"https://{target}"
    try:
        return bool(webbrowser.open(target))
    except Exception:
        return False
