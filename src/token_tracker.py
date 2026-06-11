"""Token tracking and transcript usage aggregation."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

_MILLION = 1_000_000
TOKEN_DIMS = (
    "input_tokens",
    "output_tokens",
    "cache_read_input_tokens",
    "cache_creation_input_tokens",
)


def _safe_int(value: int | None) -> int:
    """Convert a value to int, treating None as 0."""
    return value if value is not None else 0


@dataclass
class SessionUsage:
    """Accumulates token usage for a single session.

    * ``input_tokens``, ``output_tokens``, ``cache_creation_input_tokens``
      are per-call counters that are **summed** on each update.
    * ``cache_read_input_tokens`` is **cumulative** within a session
      (we take the last value, not the sum), because the upstream API
      reports an ever-increasing counter.
    """

    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_input_tokens: int = 0
    cache_creation_input_tokens: int = 0
    _last_cache_read: int = field(default=0, repr=False)
    _seen_message_ids: set[str] = field(default_factory=set, repr=False)
    _last_per_message: dict[str, dict] = field(default_factory=dict, repr=False)

    def update(self, usage: dict, message_id: str = "") -> bool:
        """Merge a per-call usage dict into this session.

        Returns True if the update was applied, False if skipped as duplicate.
        Deduplication is by ``message_id`` — the Claude API's per-response ID
        (e.g. ``msg_0123abc``).  Streaming updates within one API call share
        the same message_id.  When a seen message_id arrives again (streaming
        delta), per-call fields (input, output, cache_creation) are REPLACED
        with the new values so the final count reflects the complete response.
        cache_read_input_tokens remains cumulative (takes the max).
        """
        inp = _safe_int(usage.get("input_tokens"))
        out = _safe_int(usage.get("output_tokens"))
        cache_read = _safe_int(usage.get("cache_read_input_tokens"))
        cache_create = _safe_int(usage.get("cache_creation_input_tokens"))

        if message_id and message_id in self._seen_message_ids:
            # Streaming delta: replace per-call fields with the latest values.
            prev = self._last_per_message.get(message_id)
            if prev is not None:
                # Subtract old, add new (replace semantics)
                self.input_tokens += inp - prev.get("input_tokens", 0)
                self.output_tokens += out - prev.get("output_tokens", 0)
                self.cache_creation_input_tokens += cache_create - prev.get("cache_creation_input_tokens", 0)
            # cache_read is cumulative — take the latest value.
            if cache_read > self._last_cache_read:
                self.cache_read_input_tokens = cache_read
                self._last_cache_read = cache_read
            # Store latest per-call values for future replacements.
            self._last_per_message[message_id] = {
                "input_tokens": inp,
                "output_tokens": out,
                "cache_creation_input_tokens": cache_create,
            }
            return True

        # Skip entirely-zero updates with no new cache_read data.
        if inp == 0 and out == 0 and cache_create == 0 and cache_read <= self._last_cache_read:
            return False

        # Per-call fields are summed.
        self.input_tokens += inp
        self.output_tokens += out
        self.cache_creation_input_tokens += cache_create

        # cache_read is cumulative — take the latest value.
        if cache_read > self._last_cache_read:
            self.cache_read_input_tokens = cache_read
            self._last_cache_read = cache_read

        if message_id:
            self._seen_message_ids.add(message_id)
            self._last_per_message[message_id] = {
                "input_tokens": inp,
                "output_tokens": out,
                "cache_creation_input_tokens": cache_create,
            }

        return True


@dataclass
class TokenTracker:
    """Tracks token usage across multiple sessions and can calculate cost."""

    _input_price: float | None  # per 1M input tokens
    _cache_creation_price: float | None  # per 1M cache creation tokens
    _cache_read_price: float | None  # per 1M cache read tokens
    _output_price: float | None  # per 1M output tokens
    _sessions: dict[str, SessionUsage] = field(default_factory=dict)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update_session(self, session_key: str, usage: dict, message_id: str = "") -> bool:
        """Create a SessionUsage for *session_key* if needed, then delegate."""
        if session_key not in self._sessions:
            self._sessions[session_key] = SessionUsage()
        return self._sessions[session_key].update(usage, message_id=message_id)

    @property
    def totals(self) -> SessionUsage:
        """Return a SessionUsage summing all tracked sessions.

        cache_read_input_tokens are **summed** across sessions because each
        session independently tracks a cumulative counter.
        """
        total = SessionUsage()
        for session in self._sessions.values():
            total.input_tokens += session.input_tokens
            total.output_tokens += session.output_tokens
            total.cache_read_input_tokens += session.cache_read_input_tokens
            total.cache_creation_input_tokens += session.cache_creation_input_tokens
        return total

    def calculate_cost(self) -> float | None:
        """Calculate total cost across all sessions.

        Returns None if all prices are None.
        Each token type uses its own price; if a type-specific price is None,
        it falls back to _input_price (for input-class tokens).
        """
        if self._input_price is None and self._output_price is None and self._cache_creation_price is None and self._cache_read_price is None:
            return None

        t = self.totals
        cost = 0.0
        # Input tokens
        input_p = self._input_price
        if input_p is not None:
            cost += t.input_tokens * input_p / _MILLION
        # Cache creation — own price, fallback to input price
        cc_p = self._cache_creation_price if self._cache_creation_price is not None else self._input_price
        if cc_p is not None:
            cost += t.cache_creation_input_tokens * cc_p / _MILLION
        # Cache read — own price, fallback to input price
        cr_p = self._cache_read_price if self._cache_read_price is not None else self._input_price
        if cr_p is not None:
            cost += t.cache_read_input_tokens * cr_p / _MILLION
        # Output tokens
        if self._output_price is not None:
            cost += t.output_tokens * self._output_price / _MILLION
        return cost

    def session_usage(self, session_key: str) -> SessionUsage | None:
        """Return the SessionUsage for a given key, or None."""
        return self._sessions.get(session_key)

    def session_cost(self, session_key: str) -> float | None:
        """Calculate cost for a single session. Returns None if session doesn't exist."""
        usage = self._sessions.get(session_key)
        if usage is None:
            return None

        if self._input_price is None and self._output_price is None and self._cache_creation_price is None and self._cache_read_price is None:
            return None

        cost = 0.0
        if self._input_price is not None:
            cost += usage.input_tokens * self._input_price / _MILLION
        cc_p = self._cache_creation_price if self._cache_creation_price is not None else self._input_price
        if cc_p is not None:
            cost += usage.cache_creation_input_tokens * cc_p / _MILLION
        cr_p = self._cache_read_price if self._cache_read_price is not None else self._input_price
        if cr_p is not None:
            cost += usage.cache_read_input_tokens * cr_p / _MILLION
        if self._output_price is not None:
            cost += usage.output_tokens * self._output_price / _MILLION
        return cost


def usage_to_dict(usage: SessionUsage) -> dict[str, int]:
    """Return the public token counters as a JSON-serializable dict."""
    return {dim: int(getattr(usage, dim)) for dim in TOKEN_DIMS}


def token_usage_dict(tracker: TokenTracker) -> dict[str, int]:
    """Return total token usage for a tracker."""
    return usage_to_dict(tracker.totals)


def cost_stats(tracker: TokenTracker, currency: str = "USD") -> dict[str, float | str]:
    """Return total and per-dimension cost without duplicating price metadata."""
    t = tracker.totals
    input_cost = (
        t.input_tokens * tracker._input_price / _MILLION
        if tracker._input_price is not None else 0.0
    )
    cc_price = (
        tracker._cache_creation_price
        if tracker._cache_creation_price is not None else tracker._input_price
    )
    cache_creation_cost = (
        t.cache_creation_input_tokens * cc_price / _MILLION
        if cc_price is not None else 0.0
    )
    cr_price = (
        tracker._cache_read_price
        if tracker._cache_read_price is not None else tracker._input_price
    )
    cache_read_cost = (
        t.cache_read_input_tokens * cr_price / _MILLION
        if cr_price is not None else 0.0
    )
    output_cost = (
        t.output_tokens * tracker._output_price / _MILLION
        if tracker._output_price is not None else 0.0
    )
    total = input_cost + cache_creation_cost + cache_read_cost + output_cost
    return {
        "total_cost": total,
        "currency": currency,
        "total_input_cost": input_cost,
        "total_cache_creation_cost": cache_creation_cost,
        "total_cache_read_cost": cache_read_cost,
        "total_output_cost": output_cost,
    }


def format_token_summary(tracker: TokenTracker) -> str:
    """Format tracker totals for terminal/TUI display."""
    t = tracker.totals
    cost = tracker.calculate_cost()
    cost_str = f" · ${cost:.2f}" if cost is not None else " · $N/A"
    return (
        f"{format_token_count(t.input_tokens)} in · "
        f"{format_token_count(t.output_tokens)} out · "
        f"{format_token_count(t.cache_read_input_tokens)} cache_r"
        f"{cost_str}"
    )


def hydrate_tracker_from_summary(
    run_dir: Path,
    tracker: TokenTracker,
    *,
    session_key: str = "_prior",
) -> bool:
    """Load prior run token totals from run_summary.json into a tracker."""
    path = run_dir / "run_summary.json"
    if not path.is_file():
        return False
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    usage = payload.get("token_usage")
    if not _valid_usage(usage):
        return False
    return tracker.update_session(session_key, usage, message_id="_summary")


def hydrate_tracker_from_run(
    run_dir: Path,
    tracker: TokenTracker,
    *,
    session_key: str = "_prior",
) -> bool:
    """Hydrate a tracker from summary, falling back to transcript aggregation."""
    if hydrate_tracker_from_summary(run_dir, tracker, session_key=session_key):
        return True
    usage = aggregate_token_usage(run_dir / "workspace")
    if not _valid_usage(usage):
        return False
    return tracker.update_session(session_key, usage, message_id="_transcripts")


def aggregate_token_usage(workspace_dir: Path) -> dict[str, int]:
    """Sum token usage from controller transcripts and Claude subagent JSONL."""
    totals = {dim: 0 for dim in TOKEN_DIMS}
    transcripts_dir = find_transcripts_dir(workspace_dir)
    if transcripts_dir.is_dir():
        for path in transcripts_dir.rglob("*.jsonl"):
            usage = sum_session_usage(path)
            if usage:
                _add_usage(totals, usage)

    agent_home = workspace_dir / ".agent_home" / ".claude" / "projects" / "-workspace"
    if agent_home.is_dir():
        _aggregate_subagent_usage_from_dir(agent_home, totals)

    return totals


def find_transcripts_dir(workspace: Path) -> Path:
    """Return the run-level transcripts dir, falling back to legacy location."""
    run_dir = workspace.parent
    new = run_dir / "session_data" / "session_transcripts"
    if new.is_dir():
        return new
    return workspace / "session_transcripts"


def sum_session_usage(jsonl_path: Path) -> dict[str, Any] | None:
    """Sum usage across assistant events in a JSONL transcript."""
    try:
        text = jsonl_path.read_text(encoding="utf-8")
    except OSError:
        return None
    return _sum_usage_lines(text.splitlines())


def scan_jsonl_usage_since(path: Path, offset: int) -> tuple[int, list[tuple[dict, str]]]:
    """Read new JSONL data from offset and return final usage per message id."""
    last_per_id: dict[str, dict] = {}
    no_id_usages: list[tuple[dict, str]] = []
    try:
        with open(path, "r", encoding="utf-8") as f:
            f.seek(offset)
            for line in f:
                data = _loads_json_line(line)
                if not data or data.get("type") != "assistant":
                    continue
                msg = data.get("message", {})
                usage = msg.get("usage") if isinstance(msg, dict) else None
                message_id = msg.get("id", "") if isinstance(msg, dict) else ""
                if not _valid_usage(usage):
                    continue
                if message_id:
                    last_per_id[message_id] = usage
                else:
                    no_id_usages.append((usage, ""))
            new_offset = f.tell()
    except OSError:
        new_offset = offset
    return new_offset, [(u, mid) for mid, u in last_per_id.items()] + no_id_usages


def _sum_usage_lines(lines: list[str]) -> dict[str, Any] | None:
    summed_in = 0
    summed_out = 0
    summed_cache_c = 0
    last_cache_r = 0
    last_per_id: dict[str, dict] = {}
    found = False

    for line in lines:
        data = _loads_json_line(line)
        if not data:
            continue
        if data.get("type") == "result":
            usage = data.get("usage")
            return usage if isinstance(usage, dict) else None
        if data.get("type") != "assistant":
            continue
        msg = data.get("message", {})
        if not isinstance(msg, dict):
            continue
        usage = msg.get("usage")
        if not isinstance(usage, dict):
            continue
        found = True
        mid = msg.get("id", "")
        if mid:
            last_per_id[mid] = usage
        else:
            summed_in += int(usage.get("input_tokens") or 0)
            summed_out += int(usage.get("output_tokens") or 0)
            summed_cache_c += int(usage.get("cache_creation_input_tokens") or 0)
            last_cache_r = max(last_cache_r, int(usage.get("cache_read_input_tokens") or 0))

    for usage in last_per_id.values():
        summed_in += int(usage.get("input_tokens") or 0)
        summed_out += int(usage.get("output_tokens") or 0)
        summed_cache_c += int(usage.get("cache_creation_input_tokens") or 0)
        last_cache_r = max(last_cache_r, int(usage.get("cache_read_input_tokens") or 0))

    if not found:
        return None
    return {
        "input_tokens": summed_in,
        "output_tokens": summed_out,
        "cache_read_input_tokens": last_cache_r,
        "cache_creation_input_tokens": summed_cache_c,
    }


def _aggregate_subagent_usage_from_dir(project_dir: Path, totals: dict[str, int]) -> None:
    try:
        for session_dir in project_dir.iterdir():
            if not session_dir.is_dir():
                continue
            subagent_dir = session_dir / "subagents"
            if subagent_dir.is_dir():
                for jsonl_path in subagent_dir.glob("agent-*.jsonl"):
                    usage = sum_session_usage(jsonl_path)
                    if usage:
                        _add_usage(totals, usage)
        for jsonl_path in project_dir.glob("agent-*.jsonl"):
            usage = sum_session_usage(jsonl_path)
            if usage:
                _add_usage(totals, usage)
    except OSError:
        pass


def _add_usage(totals: dict[str, int], usage: dict[str, Any]) -> None:
    for dim in TOKEN_DIMS:
        totals[dim] += int(usage.get(dim) or 0)


def _valid_usage(usage: Any) -> bool:
    if not isinstance(usage, dict):
        return False
    return any(int(usage.get(dim) or 0) > 0 for dim in TOKEN_DIMS)


def _loads_json_line(line: str) -> dict[str, Any] | None:
    line = line.strip()
    if not line:
        return None
    try:
        data = json.loads(line)
    except json.JSONDecodeError:
        return None
    return data if isinstance(data, dict) else None


def format_token_count(n: int) -> str:
    """Format a token count for compact display."""
    if n >= 1_000_000:
        value = n / 1_000_000
        return f"{value:.1f}M" if value != int(value) else f"{int(value)}M"
    if n >= 1_000:
        value = n / 1_000
        return f"{value:.1f}K" if value != int(value) else f"{int(value)}K"
    return str(n)
