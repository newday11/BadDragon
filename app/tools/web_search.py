"""web_search tool.

Default backend uses public HTML search pages with automatic fallback:
bing -> baidu -> sogou -> duckduckgo.
"""

from __future__ import annotations

from html import unescape
import re
from typing import Any
from urllib import parse, request

from app.infra.config import load_config


def run(arguments: dict[str, Any]) -> dict[str, Any]:
    query = str(arguments.get("query", "") or "").strip()
    if not query:
        return _err("`query` is required.")

    cfg = load_config()
    search_cfg = cfg.get("search", {}) if isinstance(cfg, dict) else {}
    if not isinstance(search_cfg, dict):
        search_cfg = {}

    max_results = _to_int(arguments.get("max_results", search_cfg.get("max_results", 5)), 5)
    timeout = _to_int(arguments.get("timeout", search_cfg.get("timeout", 20)), 20)

    if max_results < 1:
        max_results = 1
    if max_results > 10:
        max_results = 10
    if timeout < 5:
        timeout = 5
    if timeout > 60:
        timeout = 60

    provider = str(search_cfg.get("provider", "auto") or "auto").strip().lower()
    if provider not in {"auto", "bing", "baidu", "sogou", "duckduckgo"}:
        provider = "auto"

    try:
        source, results = _search_with_fallback(
            query=query,
            max_results=max_results,
            timeout=timeout,
            provider=provider,
        )
    except Exception as exc:
        return _err(f"search failed: {exc}")

    return {
        "status": "ok",
        "output": {
            "query": query,
            "count": len(results),
            "results": results,
            "source": source,
        },
        "error": "",
        "artifacts": [],
        "meta": {"tool": "web_search"},
    }


def _search_with_fallback(
    *,
    query: str,
    max_results: int,
    timeout: int,
    provider: str,
) -> tuple[str, list[dict[str, str]]]:
    order = [provider] if provider != "auto" else ["bing", "baidu", "sogou", "duckduckgo"]
    last_error = ""
    for engine in order:
        try:
            if engine == "bing":
                items = _bing_search(query=query, max_results=max_results, timeout=timeout)
                if items:
                    return "bing_html", items
            elif engine == "baidu":
                items = _baidu_search(query=query, max_results=max_results, timeout=timeout)
                if items:
                    return "baidu_html", items
            elif engine == "sogou":
                items = _sogou_search(query=query, max_results=max_results, timeout=timeout)
                if items:
                    return "sogou_html", items
            elif engine == "duckduckgo":
                items = _duckduckgo_search(query=query, max_results=max_results, timeout=timeout)
                if items:
                    return "duckduckgo_html", items
        except Exception as exc:
            last_error = str(exc)
            continue
    if last_error:
        raise RuntimeError(last_error)
    return "none", []


def _http_get(url: str, *, timeout: int) -> str:
    req = request.Request(
        url=url,
        headers={
            "User-Agent": (
                "Mozilla/5.0 (X11; Linux x86_64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/124.0.0.0 Safari/537.36"
            ),
        },
        method="GET",
    )
    with request.urlopen(req, timeout=timeout) as resp:
        return resp.read().decode("utf-8", errors="replace")


def _bing_search(*, query: str, max_results: int, timeout: int) -> list[dict[str, str]]:
    html = _http_get(f"https://www.bing.com/search?q={parse.quote_plus(query)}", timeout=timeout)
    out: list[dict[str, str]] = []
    seen: set[str] = set()
    blocks = re.findall(r"<li class=\"b_algo\"[\s\S]*?</li>", html, flags=re.IGNORECASE)
    for block in blocks:
        if len(out) >= max_results:
            break
        m = re.search(
            r"<h2[^>]*>\s*<a[^>]*href=\"([^\"]+)\"[^>]*>([\s\S]*?)</a>\s*</h2>",
            block,
            flags=re.IGNORECASE,
        )
        if not m:
            continue
        url = _clean_text(_strip_tags(m.group(1)))
        if not _is_http_url(url) or url in seen:
            continue
        seen.add(url)
        title = _clean_text(_strip_tags(m.group(2)))
        s = re.search(r"<p>([\s\S]*?)</p>", block, flags=re.IGNORECASE)
        snippet = _clean_text(_strip_tags(s.group(1))) if s else ""
        out.append({"title": title or url, "url": url, "snippet": snippet})
    return out


def _baidu_search(*, query: str, max_results: int, timeout: int) -> list[dict[str, str]]:
    html = _http_get(f"https://www.baidu.com/s?wd={parse.quote_plus(query)}", timeout=timeout)
    out: list[dict[str, str]] = []
    seen: set[str] = set()
    for m in re.finditer(r"<h3[^>]*>\s*<a[^>]*href=\"([^\"]+)\"[^>]*>([\s\S]*?)</a>", html, flags=re.IGNORECASE):
        if len(out) >= max_results:
            break
        url = _clean_text(_strip_tags(m.group(1)))
        if not _is_http_url(url) or url in seen:
            continue
        seen.add(url)
        title = _clean_text(_strip_tags(m.group(2)))
        out.append({"title": title or url, "url": url, "snippet": ""})
    return out


def _sogou_search(*, query: str, max_results: int, timeout: int) -> list[dict[str, str]]:
    html = _http_get(f"https://www.sogou.com/web?query={parse.quote_plus(query)}", timeout=timeout)
    out: list[dict[str, str]] = []
    seen: set[str] = set()
    for m in re.finditer(r"<h3[^>]*>\s*<a[^>]*href=\"([^\"]+)\"[^>]*>([\s\S]*?)</a>", html, flags=re.IGNORECASE):
        if len(out) >= max_results:
            break
        url = _clean_text(_strip_tags(m.group(1)))
        if not _is_http_url(url) or url in seen:
            continue
        seen.add(url)
        title = _clean_text(_strip_tags(m.group(2)))
        out.append({"title": title or url, "url": url, "snippet": ""})
    return out


def _duckduckgo_search(*, query: str, max_results: int, timeout: int) -> list[dict[str, str]]:
    url = "https://html.duckduckgo.com/html/"
    data = parse.urlencode({"q": query}).encode("utf-8")
    req = request.Request(url=url, data=data, headers={"Content-Type": "application/x-www-form-urlencoded"}, method="POST")
    with request.urlopen(req, timeout=timeout) as resp:
        html = resp.read().decode("utf-8", errors="replace")

    items = _parse_ddg_html(html=html, max_results=max_results)
    if items:
        return items
    return _parse_ddg_lite_fallback(html=html, max_results=max_results)


def _parse_ddg_html(*, html: str, max_results: int) -> list[dict[str, str]]:
    out: list[dict[str, str]] = []
    seen: set[str] = set()
    blocks = re.findall(
        r'<div class="result__body">([\s\S]*?)</div>\s*</div>',
        str(html or ""),
        flags=re.IGNORECASE,
    )
    for block in blocks:
        if len(out) >= max_results:
            break
        m_link = re.search(
            r'<a[^>]*class="result__a"[^>]*href="([^"]+)"[^>]*>([\s\S]*?)</a>',
            block,
            flags=re.IGNORECASE,
        )
        if not m_link:
            continue
        raw_url = _clean_text(_strip_tags(m_link.group(1)))
        title = _clean_text(_strip_tags(m_link.group(2)))
        snippet = ""
        m_snip = re.search(
            r'<a[^>]*class="result__snippet"[^>]*>([\s\S]*?)</a>|<div[^>]*class="result__snippet"[^>]*>([\s\S]*?)</div>',
            block,
            flags=re.IGNORECASE,
        )
        if m_snip:
            snippet_raw = m_snip.group(1) if m_snip.group(1) is not None else m_snip.group(2)
            snippet = _clean_text(_strip_tags(snippet_raw))
        url = _resolve_ddg_redirect(raw_url)
        if not url:
            continue
        if url in seen:
            continue
        seen.add(url)
        out.append({"title": title or url, "url": url, "snippet": snippet})
    return out


def _parse_ddg_lite_fallback(*, html: str, max_results: int) -> list[dict[str, str]]:
    out: list[dict[str, str]] = []
    seen: set[str] = set()
    for m in re.finditer(r'<a[^>]*href="([^"]+)"[^>]*>([\s\S]*?)</a>', str(html or ""), flags=re.IGNORECASE):
        if len(out) >= max_results:
            break
        raw_url = _clean_text(_strip_tags(m.group(1)))
        title = _clean_text(_strip_tags(m.group(2)))
        url = _resolve_ddg_redirect(raw_url)
        if not url:
            continue
        if not (url.startswith("http://") or url.startswith("https://")):
            continue
        if "duckduckgo.com" in url:
            continue
        if url in seen:
            continue
        seen.add(url)
        out.append({"title": title or url, "url": url, "snippet": ""})
    return out


def _resolve_ddg_redirect(url: str) -> str:
    raw = str(url or "").strip()
    if not raw:
        return ""
    if raw.startswith("//"):
        raw = "https:" + raw
    if raw.startswith("/"):
        raw = "https://duckduckgo.com" + raw
    parsed = parse.urlparse(raw)
    if "duckduckgo.com" in parsed.netloc and parsed.path.startswith("/l/"):
        qs = parse.parse_qs(parsed.query)
        uddg = (qs.get("uddg") or [""])[0]
        if uddg:
            return parse.unquote(uddg)
    return raw


def _strip_tags(text: str) -> str:
    t = re.sub(r"<[^>]+>", " ", str(text or ""))
    return unescape(t)


def _clean_text(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def _is_http_url(url: str) -> bool:
    u = str(url or "").strip().lower()
    return u.startswith("http://") or u.startswith("https://")


def _to_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except Exception:
        return default


def _err(message: str) -> dict[str, Any]:
    return {
        "status": "error",
        "output": {},
        "error": str(message),
        "artifacts": [],
        "meta": {"tool": "web_search"},
    }
