"""Runtime globals shared between pipeline and nodes.

LangGraph state must be JSON-serializable, so we can't pass callables
(event_queue) through it. This module holds runtime references that nodes
look up at execution time.
"""

from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .acp.stream_adapter import StreamAdapter
    from .acp.pty_adapter import PtyAdapter
    from .config import Config
    from .token_tracker import TokenTracker

log = logging.getLogger(__name__)

_config: Config | None = None
_event_queue: asyncio.Queue | None = None
_docker_container: Any = None
_token_tracker: TokenTracker | None = None
_pipeline_state_path: Path | None = None
_run_dir: Path | None = None

# Active-session registry for pause-and-send.
_active_sessions: dict[str, str] = {}
_active_adapter: StreamAdapter | PtyAdapter | None = None
_user_message_queue: asyncio.Queue | None = None


def set_config(config: Config) -> None:
    global _config
    _config = config


def get_config() -> Config:
    if _config is None:
        raise RuntimeError("Runtime config not set — call set_config() before running the pipeline")
    return _config


def set_pipeline_state_path(path: Path | None) -> None:
    global _pipeline_state_path
    _pipeline_state_path = path


def set_run_dir(path: Path | None) -> None:
    global _run_dir
    _run_dir = path


def get_run_dir() -> Path | None:
    return _run_dir


def write_pipeline_state(**kwargs: Any) -> None:
    """Write current pipeline state to disk for the web monitor to poll.

    Called at stage_change, approaches_registered, run_ended, and token
    update points so the monitor always has fresh data.
    """
    if _pipeline_state_path is None:
        return
    try:
        existing: dict[str, Any] = {}
        if _pipeline_state_path.exists():
            raw = _pipeline_state_path.read_text(encoding="utf-8").strip()
            if raw:
                existing = json.loads(raw)
        existing.update(kwargs)
        _pipeline_state_path.write_text(
            json.dumps(existing, indent=2) + "\n", encoding="utf-8"
        )
    except Exception:
        log.debug("Failed to write pipeline state", exc_info=True)


def set_token_tracker(tracker: TokenTracker | None) -> None:
    global _token_tracker
    _token_tracker = tracker


def get_token_tracker() -> TokenTracker:
    if _token_tracker is None:
        raise RuntimeError("Token tracker not set — call set_token_tracker() before running the pipeline")
    return _token_tracker


def set_event_queue(queue: asyncio.Queue | None) -> None:
    global _event_queue
    _event_queue = queue


def get_event_queue() -> asyncio.Queue | None:
    return _event_queue


def set_docker_container(container: Any) -> None:
    global _docker_container
    _docker_container = container


def get_docker_container() -> Any:
    return _docker_container


def get_container() -> Any:
    """Alias for get_docker_container — used by diagnostic code."""
    return _docker_container


def push_stage_event(
    stage: str, loop_index: int, time_limit_seconds: float | None = None,
    workspace_dir: str = "",
    prior_elapsed_seconds: float = 0.0,
) -> None:
    """Push a synthetic stage-change event to the TUI queue.

    `time_limit_seconds`, when provided, lets the TUI display an accurate
    countdown/progress indicator for the entering stage.
    `workspace_dir`, when provided, lets the TUI locate .agent_home JSONL
    files for stream-mode token usage polling.
    """
    write_pipeline_state(
        current_loop_index=loop_index,
        current_stage=stage,
        pipeline_status="running",
    )
    if _event_queue is not None:
        from .acp.protocol import SessionEvent
        event = SessionEvent(
            type="stage_change",
            data={
                "stage": stage,
                "loop_index": loop_index,
                "time_limit_seconds": time_limit_seconds,
                "workspace_dir": workspace_dir,
                "prior_elapsed_seconds": prior_elapsed_seconds,
            },
        )
        _event_queue.put_nowait(("_system", event))


def push_resume_history(
    transcript_path: Path,
    context_key: str = "prepare",
) -> None:
    """Push a condensed summary of a previous session's transcript to the TUI.

    Reads the JSONL transcript, counts exchanges, extracts tool usage and the
    last assistant text, then pushes a single ``resume_history`` event so the
    user sees context before the resumed session starts producing new events.
    """
    if not transcript_path.exists():
        return
    try:
        lines = transcript_path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return
    import time
    from collections import Counter
    assistant_count = 0
    tool_calls: list[str] = []
    last_assistant_text = ""
    for line in lines:
        line = line.strip()
        if not line:
            continue
        try:
            data = json.loads(line)
        except json.JSONDecodeError:
            continue
        if data.get("type") == "assistant":
            assistant_count += 1
            msg = data.get("message", {})
            if not isinstance(msg, dict):
                continue
            for cb in msg.get("content", []):
                if not isinstance(cb, dict):
                    continue
                if cb.get("type") == "tool_use":
                    tool_calls.append(cb.get("name", ""))
                elif cb.get("type") == "text":
                    txt = cb.get("text", "").strip()
                    if txt:
                        last_assistant_text = txt
    if assistant_count == 0:
        return
    tool_summary = ", ".join(
        f"{name} x{count}" for name, count in Counter(tool_calls).most_common(5)
    )
    parts = [f"[dim]--- Resumed session ({assistant_count} prior exchanges) ---[/dim]"]
    if tool_summary:
        parts.append(f"[dim]Tools used: {tool_summary}[/dim]")
    if last_assistant_text:
        snippet = last_assistant_text[:200].replace("\n", " ")
        parts.append(f"[dim]Last: {snippet}[/dim]")
    parts.append("[dim]--- Continuing from where left off ---[/dim]")
    if _event_queue is not None:
        from .acp.protocol import SessionEvent
        event = SessionEvent(
            type="resume_history",
            data={"summary": "\n".join(parts)},
            timestamp=time.time(),
        )
        _event_queue.put_nowait((context_key, event))


def push_run_end_event(
    status: str,
    *,
    reason: str = "",
    stage: str = "",
    round_label: str = "",
    detail: str = "",
    evidence_paths: list[str] | None = None,
    token_summary: str = "",
    best_score: float | None = None,
    best_approach_id: str = "",
    baseline_score: float | None = None,
    baseline_approach_id: str = "",
) -> None:
    """Push a synthetic run-ended event once the graph has terminated.

    All extra fields are optional; the TUI fills in safe defaults so older
    callers (legacy exception paths) still produce a usable banner.
    """
    write_pipeline_state(
        pipeline_status=status,
        current_stage=stage,
    )
    if _event_queue is not None:
        from .acp.protocol import SessionEvent
        event = SessionEvent(
            type="run_ended",
            data={
                "status": status,
                "reason": reason,
                "stage": stage,
                "round": round_label,
                "detail": detail,
                "evidence_paths": list(evidence_paths or []),
                "token_summary": token_summary,
                "best_score": best_score,
                "best_approach_id": best_approach_id,
                "baseline_score": baseline_score,
                "baseline_approach_id": baseline_approach_id,
            },
        )
        _event_queue.put_nowait(("_system", event))

    # Generate a static HTML snapshot of the monitor page so it can be
    # viewed after the server is gone.
    if _run_dir is not None:
        try:
            from .monitor.server import generate_snapshot
            generate_snapshot(_run_dir)
        except Exception:
            log.debug("Failed to generate monitor snapshot", exc_info=True)


def set_active_adapter(adapter: "StreamAdapter | None") -> None:
    global _active_adapter
    _active_adapter = adapter


def get_active_adapter() -> "StreamAdapter | None":
    return _active_adapter


def register_active_session(context_key: str, session_key: str) -> None:
    """Record which running session backs the given context (e.g. "propose")."""
    _active_sessions[context_key] = session_key


def unregister_active_session(context_key: str) -> None:
    _active_sessions.pop(context_key, None)


def get_active_session(context_key: str) -> str | None:
    return _active_sessions.get(context_key)


def set_user_message_queue(queue: asyncio.Queue | None) -> None:
    global _user_message_queue
    _user_message_queue = queue


def enqueue_user_message(context_key: str, text: str) -> bool:
    """Queue a user message for delivery to the session at `context_key`.

    Returns True if queued, False if no message queue is attached (e.g.
    TUI not running). The actual `adapter.send` call happens on the TUI's
    event loop consumer.
    """
    if _user_message_queue is None:
        return False
    _user_message_queue.put_nowait((context_key, text))
    return True


def push_session_resuming_event(session_key: str) -> None:
    """Push a synthetic event indicating that a session is being resumed.

    Emitted by the TUI when adapter.send() begins so the user sees feedback
    immediately rather than waiting for the new process's first event.
    """
    if _event_queue is not None:
        from .acp.protocol import SessionEvent
        event = SessionEvent(
            type="session_resuming",
            data={"session_key": session_key},
        )
        _event_queue.put_nowait((session_key, event))


def push_session_sending_event(session_key: str) -> None:
    """Push a synthetic event indicating that a message is being sent (PTY mode)."""
    if _event_queue is not None:
        from .acp.protocol import SessionEvent
        event = SessionEvent(
            type="session_sending",
            data={"session_key": session_key},
        )
        _event_queue.put_nowait((session_key, event))


def push_cost_warning(level: str, current_cost: float, limit: float) -> None:
    """Push a cost warning event to the TUI queue."""
    if _event_queue is not None:
        from .acp.protocol import SessionEvent
        event = SessionEvent(
            type="cost_warning",
            data={"level": level, "current_cost": current_cost, "limit": limit},
        )
        _event_queue.put_nowait(("_system", event))


def push_approach_results_event(
    approach_results: dict[str, dict],
) -> None:
    """Push per-approach outcome (status + score) to the TUI.

    Called by implement_node after collecting best_result.jsonl data.
    ``approach_results`` maps approach_id → {"status": "completed"|"failed",
    "score": float|None}.
    """
    if _event_queue is not None:
        from .acp.protocol import SessionEvent
        event = SessionEvent(
            type="approach_results",
            data={"results": approach_results},
        )
        _event_queue.put_nowait(("_system", event))


def push_approaches_event(
    loop_index: int,
    approaches: list[dict],
    session_map: dict[str, str],
    time_limit_seconds: float | None = None,
    prior_elapsed_seconds: float = 0.0,
) -> None:
    """Push approach metadata and session-key-to-approach-id mapping to the TUI.

    time_limit_seconds, when provided, lets the TUI display accurate per-approach
    budget bars instead of falling back to ApproachState's default.
    """
    write_pipeline_state(
        current_loop_index=loop_index,
        current_stage="implement",
        num_approaches=len(approaches),
    )
    if _event_queue is not None:
        from .acp.protocol import SessionEvent
        event = SessionEvent(
            type="approaches_registered",
            data={
                "loop_index": loop_index,
                "approaches": approaches,
                "session_map": session_map,
                "time_limit_seconds": time_limit_seconds,
                "prior_elapsed_seconds": prior_elapsed_seconds,
            },
        )
        _event_queue.put_nowait(("_system", event))
