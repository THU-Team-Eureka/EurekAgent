"""ACP message types for the claude CLI stream-json interface.

The claude -p --output-format stream-json command emits NDJSON (one JSON object
per stdout line). This module defines the data types used to represent those
messages within our system.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class SessionRequest:
    """Parameters for launching a claude CLI session."""

    prompt: str
    model: str = ""
    cwd: str = ""
    session_id: str = ""  # pre-assigned UUID; empty = let claude generate one
    permissions: str = ""  # "bypassPermissions" or ""
    log_path: str = ""  # where to write the NDJSON event log on disk
    resume: bool = False  # True → --resume <session_id> instead of --session-id
    completion_check: Callable[[], bool] | None = None
    persist_until_timeout: bool = False  # True → poll-based completion_check won't terminate session
    pause_check: Callable[[], bool] | None = None  # True when agent is waiting for user input (e.g. question.json)
    on_pause: Callable[[], None] | None = None  # Called once when pause_check first confirms agent is waiting
    warning_prompt: str | None = None  # injected at 5-min-before-timeout if deliverable not produced
    max_recoveries: int | None = None  # None=unlimited, int=cap for PTY adapter recovery budget
    env: dict[str, str] = field(default_factory=dict)  # per-session environment variables


@dataclass
class SessionEvent:
    """One NDJSON line from claude -p --output-format stream-json.

    Event types emitted by the CLI:
      init          – session started, contains session_id
      message       – complete assistant/user message
      result        – final result with session_id and text
      system        – retries, compaction, internal events
      stream_event  – token-level delta (with --include-partial-messages)
    """

    type: str  # "init", "message", "result", "system", "stream_event"
    data: dict[str, Any]  # raw parsed JSON from the NDJSON line
    timestamp: float = 0.0


@dataclass
class SessionResult:
    """Outcome of a completed (or killed) claude CLI session."""

    session_id: str
    exit_code: int | None
    elapsed_seconds: float
    log_path: Path  # NDJSON event log on disk
    final_result: dict[str, Any] | None = None  # last "result" event's data
    error: str | None = None  # stderr if process failed
    context_exhausted: bool = False
    recovery_abort_reason: str = ""  # "api_error"|"no_response"|"context_exhausted"|"budget_exhausted"|""
