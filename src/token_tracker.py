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

_CURRENCY_SYMBOLS = {
    "USD": "$",
    "CNY": "¥",
    "RMB": "¥",
    "GBP": "£",
    "EUR": "€",
}


@dataclass
class BillableUsage:
    """Per-usage token deltas used for context-length tiered pricing."""

    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_input_tokens: int = 0
    cache_creation_input_tokens: int = 0


@dataclass(frozen=True)
class CostBreakdown:
    input_cost: float = 0.0
    cache_creation_cost: float = 0.0
    cache_read_cost: float = 0.0
    output_cost: float = 0.0

    @property
    def total(self) -> float:
        return (
            self.input_cost
            + self.cache_creation_cost
            + self.cache_read_cost
            + self.output_cost
        )


def _safe_int(value: int | None) -> int:
    """Convert a value to int, treating None as 0."""
    return value if value is not None else 0


def currency_symbol(currency: str | None) -> str:
    """Return a display symbol for common currency codes, defaulting to USD."""
    return _CURRENCY_SYMBOLS.get((currency or "").upper(), "$")


def format_cost(value: float | None, currency: str | None = "USD") -> str:
    """Format a cost value with the configured currency symbol."""
    symbol = currency_symbol(currency)
    return f"{symbol}{value:.2f}" if value is not None else f"{symbol}N/A"


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
    _billable_by_message: dict[str, BillableUsage] = field(default_factory=dict, repr=False)
    _billable_no_id: list[BillableUsage] = field(default_factory=list, repr=False)

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
            old_billable = self._billable_by_message.get(message_id, BillableUsage())
            previous_cache_read = int(prev.get("cache_read_input_tokens", 0)) if prev else 0
            cache_read_delta = old_billable.cache_read_input_tokens + max(
                0, cache_read - previous_cache_read
            )
            self._billable_by_message[message_id] = BillableUsage(
                input_tokens=inp,
                output_tokens=out,
                cache_read_input_tokens=cache_read_delta,
                cache_creation_input_tokens=cache_create,
            )
            # Store latest per-call values for future replacements.
            self._last_per_message[message_id] = {
                "input_tokens": inp,
                "output_tokens": out,
                "cache_read_input_tokens": cache_read,
                "cache_read_delta": cache_read_delta,
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
        cache_read_delta = 0
        if cache_read > self._last_cache_read:
            cache_read_delta = cache_read - self._last_cache_read
            self.cache_read_input_tokens = cache_read
            self._last_cache_read = cache_read

        billable = BillableUsage(
            input_tokens=inp,
            output_tokens=out,
            cache_read_input_tokens=cache_read_delta,
            cache_creation_input_tokens=cache_create,
        )

        if message_id:
            self._seen_message_ids.add(message_id)
            self._last_per_message[message_id] = {
                "input_tokens": inp,
                "output_tokens": out,
                "cache_read_input_tokens": cache_read,
                "cache_read_delta": cache_read_delta,
                "cache_creation_input_tokens": cache_create,
            }
            self._billable_by_message[message_id] = billable
        else:
            self._billable_no_id.append(billable)

        return True

    @property
    def billable_usages(self) -> list[BillableUsage]:
        return list(self._billable_by_message.values()) + list(self._billable_no_id)


@dataclass
class TokenTracker:
    """Tracks token usage across multiple sessions and can calculate cost."""

    _input_price: float | None  # per 1M input tokens
    _cache_creation_price: float | None  # per 1M cache creation tokens
    _cache_read_price: float | None  # per 1M cache read tokens
    _output_price: float | None  # per 1M output tokens
    _price_tiers: list[dict[str, float | int | None]] | None = None
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
        breakdown = self.calculate_cost_breakdown()
        return breakdown.total if breakdown is not None else None

    def calculate_cost_breakdown(self, session_key: str | None = None) -> CostBreakdown | None:
        """Calculate total cost and per-token-class costs."""
        if self._price_tiers:
            usages = self._billable_usages(session_key)
            if not usages and not self._has_any_scalar_price():
                return None
            return self._tiered_cost_breakdown(usages)

        if self._input_price is None and self._output_price is None and self._cache_creation_price is None and self._cache_read_price is None:
            return None

        t = self._usage_for_session(session_key)
        cost = 0.0
        input_cost = 0.0
        cache_creation_cost = 0.0
        cache_read_cost = 0.0
        output_cost = 0.0
        # Input tokens
        input_p = self._input_price
        if input_p is not None:
            input_cost = t.input_tokens * input_p / _MILLION
            cost += input_cost
        # Cache creation — own price, fallback to input price
        cc_p = self._cache_creation_price if self._cache_creation_price is not None else self._input_price
        if cc_p is not None:
            cache_creation_cost = t.cache_creation_input_tokens * cc_p / _MILLION
            cost += cache_creation_cost
        # Cache read — own price, fallback to input price
        cr_p = self._cache_read_price if self._cache_read_price is not None else self._input_price
        if cr_p is not None:
            cache_read_cost = t.cache_read_input_tokens * cr_p / _MILLION
            cost += cache_read_cost
        # Output tokens
        if self._output_price is not None:
            output_cost = t.output_tokens * self._output_price / _MILLION
            cost += output_cost
        return CostBreakdown(input_cost, cache_creation_cost, cache_read_cost, output_cost)

    def session_usage(self, session_key: str) -> SessionUsage | None:
        """Return the SessionUsage for a given key, or None."""
        return self._sessions.get(session_key)

    def session_cost(self, session_key: str) -> float | None:
        """Calculate cost for a single session. Returns None if session doesn't exist."""
        breakdown = self.calculate_cost_breakdown(session_key=session_key)
        return breakdown.total if breakdown is not None else None

    def _usage_for_session(self, session_key: str | None) -> SessionUsage:
        if session_key is None:
            return self.totals
        return self._sessions.get(session_key, SessionUsage())

    def _billable_usages(self, session_key: str | None) -> list[BillableUsage]:
        if session_key is not None:
            session = self._sessions.get(session_key)
            return session.billable_usages if session is not None else []
        usages: list[BillableUsage] = []
        for session in self._sessions.values():
            usages.extend(session.billable_usages)
        return usages

    def _has_any_scalar_price(self) -> bool:
        return any(
            price is not None
            for price in (
                self._input_price,
                self._cache_creation_price,
                self._cache_read_price,
                self._output_price,
            )
        )

    def _tiered_cost_breakdown(self, usages: list[BillableUsage]) -> CostBreakdown:
        input_cost = 0.0
        cache_creation_cost = 0.0
        cache_read_cost = 0.0
        output_cost = 0.0
        for usage in usages:
            context_tokens = (
                usage.input_tokens
                + usage.cache_read_input_tokens
                + usage.cache_creation_input_tokens
            )
            tier = self._tier_for_context(context_tokens)
            input_price = float(tier["input"])
            cache_creation_price = float(tier.get("cache_creation", input_price))
            cache_read_price = float(tier["cache_read"])
            output_price = float(tier["output"])
            input_cost += usage.input_tokens * input_price / _MILLION
            cache_creation_cost += (
                usage.cache_creation_input_tokens * cache_creation_price / _MILLION
            )
            cache_read_cost += usage.cache_read_input_tokens * cache_read_price / _MILLION
            output_cost += usage.output_tokens * output_price / _MILLION
        return CostBreakdown(input_cost, cache_creation_cost, cache_read_cost, output_cost)

    def _tier_for_context(self, context_tokens: int) -> dict[str, float | int | None]:
        tiers = self._price_tiers or []
        for tier in tiers:
            max_context = tier.get("max_context_tokens")
            if max_context is None or context_tokens < int(max_context):
                return tier
        return tiers[-1]


def usage_to_dict(usage: SessionUsage) -> dict[str, int]:
    """Return the public token counters as a JSON-serializable dict."""
    return {dim: int(getattr(usage, dim)) for dim in TOKEN_DIMS}


def token_usage_dict(tracker: TokenTracker) -> dict[str, int]:
    """Return total token usage for a tracker."""
    return usage_to_dict(tracker.totals)


def cost_stats(tracker: TokenTracker, currency: str = "USD") -> dict[str, float | str]:
    """Return total and per-dimension cost without duplicating price metadata."""
    breakdown = tracker.calculate_cost_breakdown()
    if breakdown is None:
        breakdown = CostBreakdown()
    return {
        "total_cost": breakdown.total,
        "currency": currency,
        "pricing_mode": "tiered" if tracker._price_tiers else "scalar",
        "total_input_cost": breakdown.input_cost,
        "total_cache_creation_cost": breakdown.cache_creation_cost,
        "total_cache_read_cost": breakdown.cache_read_cost,
        "total_output_cost": breakdown.output_cost,
    }


def format_token_summary(tracker: TokenTracker, currency: str | None = "USD") -> str:
    """Format tracker totals for terminal/TUI display."""
    t = tracker.totals
    cost = tracker.calculate_cost()
    cost_str = f" · {format_cost(cost, currency)}"
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
