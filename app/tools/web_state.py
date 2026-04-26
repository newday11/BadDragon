"""Lightweight web session state for web tools (no external deps)."""

from __future__ import annotations

from dataclasses import dataclass
from html import unescape
from pathlib import Path
import json
import re
from typing import Any
from urllib import request


DEFAULT_UA = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/124.0.0.0 Safari/537.36"
)


@dataclass
class TabData:
    tab_id: str
    url: str
    title: str
    html: str


class WebSession:
    """Very small browser-like state manager.

    It fetches page HTML over HTTP and keeps tab states in memory.
    """

    def __init__(self) -> None:
        self._tabs: dict[str, TabData] = {}
        self._current_tab_id = "tab-1"
        self._tab_counter = 1
        self._tabs[self._current_tab_id] = TabData(
            tab_id=self._current_tab_id,
            url="about:blank",
            title="about:blank",
            html="",
        )

    def get_tabs(self) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for t in self._tabs.values():
            out.append(
                {
                    "tab_id": t.tab_id,
                    "url": t.url,
                    "title": t.title,
                    "current": t.tab_id == self._current_tab_id,
                }
            )
        return out

    def ensure_tab(self, tab_id: str | None) -> TabData:
        if tab_id and tab_id in self._tabs:
            self._current_tab_id = tab_id
            return self._tabs[tab_id]
        return self._tabs[self._current_tab_id]

    def open_new_tab(self, url: str) -> TabData:
        self._tab_counter += 1
        tab_id = f"tab-{self._tab_counter}"
        self._tabs[tab_id] = TabData(tab_id=tab_id, url="about:blank", title="about:blank", html="")
        self._current_tab_id = tab_id
        return self.navigate(url, switch_tab_id=tab_id)

    def navigate(self, url: str, switch_tab_id: str | None = None) -> TabData:
        tab = self.ensure_tab(switch_tab_id)
        cleaned = str(url or "").strip()
        if cleaned and not re.match(r"^[a-zA-Z][a-zA-Z0-9+\-.]*://", cleaned):
            cleaned = f"https://{cleaned}"
        if not cleaned:
            cleaned = "about:blank"

        if cleaned == "about:blank":
            tab.url = cleaned
            tab.title = "about:blank"
            tab.html = ""
            return tab

        html = ""
        title = cleaned
        try:
            req = request.Request(
                url=cleaned,
                headers={"User-Agent": DEFAULT_UA},
                method="GET",
            )
            with request.urlopen(req, timeout=20) as resp:
                html = resp.read().decode("utf-8", errors="replace")
            m = re.search(r"<title[^>]*>(.*?)</title>", html, re.IGNORECASE | re.DOTALL)
            if m:
                title = _clean_text(m.group(1))
        except Exception as exc:
            html = f"<html><body>Fetch failed: {exc}</body></html>"
            title = "fetch_error"
        tab.url = cleaned
        tab.title = title
        tab.html = html
        return tab

    def scan(
        self,
        *,
        tabs_only: bool = False,
        switch_tab_id: str | None = None,
        text_only: bool = False,
    ) -> dict[str, Any]:
        tab = self.ensure_tab(switch_tab_id)
        tabs = self.get_tabs()
        if tabs_only:
            return {
                "tabs": tabs,
                "current_tab_id": tab.tab_id,
            }

        if text_only:
            content = _html_to_text(tab.html)
            content_key = "text"
        else:
            content = _simplify_html(tab.html)
            content_key = "html"

        return {
            "tabs": tabs,
            "current_tab_id": tab.tab_id,
            "url": tab.url,
            "title": tab.title,
            content_key: content,
        }


def _simplify_html(html: str) -> str:
    text = str(html or "")
    text = re.sub(r"<!--[\s\S]*?-->", "", text)
    text = re.sub(r"<script[\s\S]*?</script>", "", text, flags=re.IGNORECASE)
    text = re.sub(r"<style[\s\S]*?</style>", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s+", " ", text)
    return text.strip()[:50000]


def _html_to_text(html: str) -> str:
    text = str(html or "")
    text = re.sub(r"<script[\s\S]*?</script>", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"<style[\s\S]*?</style>", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"<[^>]+>", " ", text)
    text = unescape(text)
    text = _clean_text(text)
    return text[:30000]


def _clean_text(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def save_json_if_needed(path: str | None, data: dict[str, Any]) -> str | None:
    p = str(path or "").strip()
    if not p:
        return None
    target = Path(p)
    if not target.is_absolute():
        target = Path.cwd() / target
    target = target.resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return str(target)


SESSION = WebSession()

