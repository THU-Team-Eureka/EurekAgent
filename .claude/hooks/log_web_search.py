"""PostToolUse hook: auto-log web search results to web_search_history.jsonl.

Intercepts results from search-related tools and appends structured entries
to round_state/web_search_history.jsonl.  One JSONL line per result (not
per query) to enable URL-level dedup.  Append-only — never overwrites.

Supported tools:
  - mcp__web-search-prime__web_search_prime (structured results)
  - WebSearch (Claude built-in)
  - mcp__playwright__browser_navigate
  - mcp__playwright__browser_snapshot
"""

from __future__ import annotations

import json
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

_HISTORY_PATH = Path("/workspace/round_state/web_search_history.jsonl")
_DEBUG_PATH = Path("/workspace/round_state/web_search_hook_debug.jsonl")

_URL_RE = re.compile(r"https?://[^\s)>\]\"']+")


def _append(entry: dict) -> None:
    _HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(_HISTORY_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def _debug(entry: dict) -> None:
    if os.environ.get("EUREKA_WEB_SEARCH_HOOK_DEBUG", "").strip().lower() not in {"1", "true", "yes"}:
        return
    _DEBUG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(_DEBUG_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _extract_text_from_response(tool_response: dict | str | None) -> str:
    """Extract the text content from a tool_response structure.

    Handles formats produced by Claude Code's PostToolUse hook:
      - raw string
      - MCP standard: {"content": [{"type": "text", "text": "..."}]}
      - list of content blocks: [{"type": "text", "text": "..."}]
      - legacy: {"output": "text"}
    """
    if tool_response is None:
        return ""
    if isinstance(tool_response, str):
        return tool_response
    if isinstance(tool_response, dict):
        content = tool_response.get("content")
        if isinstance(content, list):
            parts = []
            for block in content:
                if isinstance(block, dict) and block.get("type") == "text":
                    parts.append(block.get("text", ""))
            if parts:
                return "\n".join(parts)
        out = tool_response.get("output")
        if isinstance(out, str):
            return out
    if isinstance(tool_response, list):
        parts = []
        for block in tool_response:
            if isinstance(block, dict) and block.get("type") == "text":
                parts.append(block.get("text", ""))
        return "\n".join(parts)
    return str(tool_response)


def _handle_web_search_prime(tool_input: dict, tool_response: dict | str | None) -> None:
    query = tool_input.get("search_query", "")
    text = _extract_text_from_response(tool_response)
    if not text:
        if query:
            _append({"tool": "web-search-prime", "query": query,
                      "url": "", "title": "", "summary": "[response not captured]",
                      "timestamp": _now()})
        return

    try:
        parsed = json.loads(text)
    except (json.JSONDecodeError, TypeError):
        _append({"tool": "web-search-prime", "query": query,
                  "url": "", "title": "", "summary": text[:500],
                  "timestamp": _now()})
        return

    # Claude Code may double-encode the text field for MCP tools:
    # json.loads('"[{\\"title\\"...}]"') returns a string, not a list.
    if isinstance(parsed, str):
        try:
            parsed = json.loads(parsed)
        except (json.JSONDecodeError, TypeError):
            pass

    if isinstance(parsed, list) and len(parsed) >= 1 and isinstance(parsed[0], dict):
        if "text" in parsed[0] and "title" not in parsed[0]:
            try:
                parsed = json.loads(parsed[0]["text"])
            except (json.JSONDecodeError, TypeError):
                pass

    results = parsed if isinstance(parsed, list) else [parsed]

    if not isinstance(results, list):
        results = [results]

    for item in results:
        if not isinstance(item, dict):
            continue
        _append({
            "tool": "web-search-prime",
            "query": query,
            "url": item.get("link", ""),
            "title": item.get("title", ""),
            "summary": item.get("content", "")[:1000],
            "timestamp": _now(),
        })


def _handle_web_search(tool_input: dict, tool_response: dict | str | None) -> None:
    query = tool_input.get("query", "")
    text = _extract_text_from_response(tool_response)
    if not text:
        return

    urls = list(set(_URL_RE.findall(text)))
    summary = text[:500] if "error" in text.lower() or "Error" in text else ""

    for url in urls:
        _append({
            "tool": "WebSearch",
            "query": query,
            "url": url,
            "title": "",
            "summary": summary,
            "timestamp": _now(),
        })
    if not urls and query:
        _append({
            "tool": "WebSearch",
            "query": query,
            "url": "",
            "title": "",
            "summary": summary,
            "timestamp": _now(),
        })


def _handle_playwright_navigate(tool_input: dict, tool_response: dict | str | None) -> None:
    url = tool_input.get("url", "")
    text = _extract_text_from_response(tool_response)
    if not url and not text:
        return

    title = ""
    m = re.search(r"Page Title:\s*(.+)", text)
    if m:
        title = m.group(1).strip()

    _append({
        "tool": "playwright-navigate",
        "query": "",
        "url": url,
        "title": title,
        "summary": "",
        "timestamp": _now(),
    })


def _handle_playwright_snapshot(tool_input: dict, tool_response: dict | str | None) -> None:
    text = _extract_text_from_response(tool_response)
    if not text:
        return

    url = ""
    m = re.search(r"Page URL:\s*(\S+)", text)
    if m:
        url = m.group(1).strip()

    title = ""
    m = re.search(r"Page Title:\s*(.+)", text)
    if m:
        title = m.group(1).strip()

    snippets: list[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("- text:"):
            val = stripped[len("- text:"):].strip().strip('"')
            if len(val) > 10:
                snippets.append(val)
        if len(snippets) >= 3:
            break
    summary = " | ".join(snippets)[:500]

    _append({
        "tool": "playwright-snapshot",
        "query": "",
        "url": url,
        "title": title,
        "summary": summary,
        "timestamp": _now(),
    })


_DISPATCH = {
    "mcp__web-search-prime__web_search_prime": _handle_web_search_prime,
    "WebSearch": _handle_web_search,
    "mcp__playwright__browser_navigate": _handle_playwright_navigate,
    "mcp__playwright__browser_snapshot": _handle_playwright_snapshot,
}


def main() -> None:
    try:
        raw = sys.stdin.read()
        event = json.loads(raw) if raw.strip() else {}
    except Exception:
        return

    tool_name = str(event.get("tool_name", "") or "")
    tool_input = event.get("tool_input") or {}

    # Try multiple possible key names for the tool response.
    # Claude Code versions may use different keys.
    tool_response = (
        event.get("tool_response")
        or event.get("tool_output")
        or event.get("tool_result")
    )

    handler = _DISPATCH.get(tool_name)
    if handler is None:
        return

    # Debug log: record what we received for diagnostics
    resp_type = type(tool_response).__name__ if tool_response is not None else "None"
    resp_keys = list(tool_response.keys()) if isinstance(tool_response, dict) else None
    text = _extract_text_from_response(tool_response)
    _debug({
        "tool_name": tool_name,
        "response_type": resp_type,
        "response_keys": resp_keys,
        "text_length": len(text) if text else 0,
        "event_keys": sorted(event.keys()),
        "timestamp": _now(),
    })

    try:
        handler(tool_input, tool_response)
    except Exception as e:
        _debug({"error": str(e), "tool_name": tool_name, "timestamp": _now()})
        print(f"[log_web_search] ERROR: {e}", file=sys.stderr)


if __name__ == "__main__":
    main()
