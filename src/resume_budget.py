"""Budget resolution rules for resumed stages."""

from __future__ import annotations

from typing import Any

from .duration import parse_time_limit


def effective_resume_budget(
    session_map: dict[str, Any] | None,
    configured_limit: str,
) -> float | None:
    """Return the budget that should control a resumed stage.

    CLI-configured limits are authoritative on resume so users can grant more
    time by changing the command from e.g. 20m to 30m. A persisted budget is
    only preferred when it already includes explicit resume-extra time.
    """
    configured = parse_time_limit(configured_limit)
    if not session_map:
        return configured

    saved = session_map.get("time_budget_seconds")
    if saved is None:
        return configured
    if session_map.get("resume_extra_seconds"):
        if configured is None:
            return float(saved)
        return max(float(configured), float(saved))
    return configured
