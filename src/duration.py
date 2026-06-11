"""Parse time limit strings into seconds."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .config import Config

_SIMPLE = re.compile(r"^(\d+(?:\.\d+)?)\s*(minutes?|hours?)$", re.IGNORECASE)
_COMPOUND = re.compile(
    r"^(\d+(?:\.\d+)?)\s*hours?\s+(\d+(?:\.\d+)?)\s*minutes?$",
    re.IGNORECASE,
)


def parse_time_limit(value: str) -> float | None:
    """Parse a time limit string to seconds.

    Accepted formats:
        "20 minutes", "1.5 hours", "1 hour 30 minutes"
    Returns None if the format is not recognized.
    """
    text = value.strip()
    m = _COMPOUND.match(text)
    if m:
        return float(m.group(1)) * 3600 + float(m.group(2)) * 60
    m = _SIMPLE.match(text)
    if m:
        amount = float(m.group(1))
        unit = m.group(2).lower()
        return amount * 3600 if unit.startswith("hour") else amount * 60
    return None


def resolve_stage_time_limit(config: "Config", stage: str) -> str:
    """Return the explicitly configured time limit for a stage.

    `stage` is one of "propose" or "implement".
    """
    if stage == "propose" and config.propose_time_limit_per_session:
        return config.propose_time_limit_per_session
    if stage == "implement" and config.implement_time_limit_per_session:
        return config.implement_time_limit_per_session
    raise ValueError(f"Missing explicit time limit for stage: {stage}")
