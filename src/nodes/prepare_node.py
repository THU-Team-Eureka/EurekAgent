"""Prepare node: runs a single Claude Code session to validate problem, test
evaluation, and set up the environment before the propose-implement loop."""

from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from pathlib import Path
from typing import Any

from ..acp.factory import make_adapter
from ..acp.protocol import SessionRequest
from ..artifacts import validate_prepare_artifacts
from ..context_continuation import detect_context_exhaustion, build_resume_preamble, build_continuation_request
from ..runtime import (
    get_config,
    get_event_queue,
    push_resume_history,
    push_stage_event,
    register_active_session,
    set_active_adapter,
    unregister_active_session,
    write_pipeline_state,
)
from ..gpu_cleanup import sweep_gpu_locks
from ..session.manager import SessionManager
from ..time_budget import write_stage_clock
from ..session_map import (
    add_session_map_elapsed,
    load_session_map,
    resolve_stage_elapsed,
    update_session_map_status,
    write_session_map,
)
from ..state import LoopState

log = logging.getLogger(__name__)

_QUESTION_POLL_INTERVAL = 2.0


async def prepare_node(state: LoopState) -> dict:
    """LangGraph node: prepare workspace before the propose-implement loop."""
    config = get_config()
    workspace = Path(state["workspace_dir"]).resolve()

    run_dir = Path(state["run_dir"]).resolve()
    session_data_dir = run_dir / "session_data"
    log_dir = run_dir / "session_data" / "session_transcripts" / "prepare"
    prior_elapsed = resolve_stage_elapsed(
        session_data_dir, stage="prepare",
        transcript_path=log_dir / "prepare.jsonl",
    )

    push_stage_event(
        "prepare", loop_index=0, time_limit_seconds=0,
        workspace_dir=str(workspace),
        prior_elapsed_seconds=prior_elapsed,
    )

    prepare_dir = workspace / "prepare"
    prepare_dir.mkdir(parents=True, exist_ok=True)

    # Fast path: skip prepare entirely when the user opts in.
    if config.skip_prepare:
        log.info("Skipping prepare stage (--skip-prepare)")
        complete_path = workspace / "prepare" / "complete.json"
        complete_path.write_text(
            json.dumps({"status": "skipped"}) + "\n",
            encoding="utf-8",
        )
        summary_path = workspace / "prepare" / "summary.md"
        summary_path.write_text(
            "Prepare stage skipped by user (--skip-prepare).\n",
            encoding="utf-8",
        )
        _write_stage_result(workspace, "prepare", "ready")
        return {
            "prepare_status": "ready",
            "prepare_summary_path": str(summary_path),
            "next_stage": "propose",
            "history": [{"stage": "prepare", "loop_index": 0, "status": "ready"}],
        }

    # Fast path: prepare already completed (e.g. from a previous run).
    complete_path = workspace / "prepare" / "complete.json"
    if validate_prepare_artifacts(workspace).ok:
        log.info("Prepare already completed, skipping")
        summary_path = workspace / "prepare" / "summary.md"
        _write_stage_result(workspace, "prepare", "ready")
        return {
            "prepare_status": "ready",
            "prepare_summary_path": str(summary_path) if summary_path.exists() else "",
            "next_stage": "propose",
            "history": [{"stage": "prepare", "loop_index": 0, "status": "ready"}],
        }

    prompt = (
        "You are a preparation agent in an auto-experiment optimization system. "
        "Your task is to validate the problem setup, test the evaluation pipeline, "
        "and configure the environment before any optimization runs begin.\n\n"
        "## Run Context\n\n"
        "Your current working directory is the workspace root. "
        "All file paths below are relative to it.\n"
        f"Problem: inputs/problem.md\n"
        f"Submission format: inputs/submission_format.md\n"
        f"Initial code: inputs/initial_code/\n\n"
        "GPUs are hidden by the system (empty CUDA_VISIBLE_DEVICES). When checking "
        "or using GPUs, use `gpu_helpers` (`get_gpu_info`, `gpu_session`) — "
        "never `nvidia-smi` or manual CUDA_VISIBLE_DEVICES.\n\n"
        "## Instruction\n\n"
        "Invoke the `prepare-workspace` skill using the Skill tool without any "
        "arguments. The skill will guide you through the full workflow."
    )

    session_id = str(uuid.uuid4())
    log_dir.mkdir(parents=True, exist_ok=True)

    existing_map = load_session_map(session_data_dir, stage="prepare")
    should_resume = (
        existing_map is not None
        and existing_map.get("status") in ("aborted", "running")
    )
    if should_resume:
        session_id = existing_map["session_id"]
        log.info("Resuming prepare session: %s (status was %s)",
                 session_id, existing_map.get("status"))
        push_resume_history(log_dir / "prepare.jsonl", context_key="prepare")
        prior_elapsed = resolve_stage_elapsed(
            session_data_dir, stage="prepare",
            transcript_path=log_dir / "prepare.jsonl",
        )

    complete_path = workspace / "prepare" / "complete.json"

    write_session_map(
        session_data_dir, session_id=session_id, stage="prepare",
        time_budget_seconds=None, status="running",
        elapsed_seconds=prior_elapsed,
    )

    now = time.time()
    write_stage_clock(
        workspace,
        stage="prepare",
        started_at=now - prior_elapsed,
        elapsed_seconds=prior_elapsed,
    )

    question_path = workspace / "prepare" / "question.json"

    def _on_agent_paused() -> None:
        """Called by the adapter when it confirms the agent is waiting for input.

        This fires AFTER the agent has ended its turn and the pause_check
        returns True, so the question won't be interleaved with agent output.
        """
        question_data = _try_parse_question(question_path)
        if question_data is None:
            raw = question_path.read_text(encoding="utf-8", errors="replace")[:500]
            question_data = {"question": raw, "options": []}
        _push_question_event(question_data)

    request = SessionRequest(
        prompt=prompt,
        model=state.get("model", ""),
        cwd=str(workspace),
        session_id=session_id,
        permissions="bypassPermissions",
        log_path=str(log_dir / "prepare.jsonl"),
        completion_check=lambda ws=workspace: validate_prepare_artifacts(ws).ok,
        pause_check=lambda q=question_path: q.exists(),
        on_pause=_on_agent_paused,
        resume=should_resume,
        max_recoveries=50,
    )

    adapter = make_adapter(config, workspace)
    manager = SessionManager(adapter, config, event_queue=get_event_queue())

    register_active_session("prepare", session_id)
    set_active_adapter(adapter)

    # Prepare has no time budget — runs until complete.json exists.
    # remaining_time=None means no timeout, matching the original behavior.
    remaining_time = None
    continuation_count = 0
    current_request = request
    result = None
    attempt_started_at: float | None = None

    try:
        while remaining_time is None or remaining_time > 0:
            attempt_started_at = time.time()

            # Each iteration gets its own monitor task scoped to the session.
            current_session_id = current_request.session_id
            monitor_task = asyncio.create_task(
                _monitor_questions(adapter, current_session_id, workspace)
            )

            try:
                result = await manager.run_single(
                    current_request, timeout=remaining_time,
                )
            finally:
                monitor_task.cancel()
                try:
                    await monitor_task
                except asyncio.CancelledError:
                    pass

            elapsed = time.time() - attempt_started_at
            add_session_map_elapsed(
                session_data_dir, stage="prepare", add_seconds=elapsed,
            )
            attempt_started_at = None
            if remaining_time is not None:
                remaining_time -= elapsed

            if validate_prepare_artifacts(workspace).ok:
                break

            if detect_context_exhaustion(result.log_path):
                continuation_count += 1
                log.warning("Prepare session hit context limit, starting continuation #%d",
                            continuation_count)
                resume_preamble = build_resume_preamble(
                    "prepare", str(workspace), continuation_count,
                )
                current_request = build_continuation_request(
                    request, resume_preamble,
                    int(remaining_time) if remaining_time is not None else 0,
                    str(workspace),
                )
                # Register the new session ID so the monitor can track it.
                register_active_session("prepare", current_request.session_id)
                continue
            else:
                break
    finally:
        unregister_active_session("prepare")
        set_active_adapter(None)
        sweep_gpu_locks(workspace, stage="prepare")
        if attempt_started_at is not None:
            add_session_map_elapsed(
                session_data_dir,
                stage="prepare",
                add_seconds=time.time() - attempt_started_at,
            )
        # Mark aborted on any non-normal exit (cancel, interrupt, etc.).
        # Overwritten to "completed" below on normal exit.
        update_session_map_status(session_data_dir, stage="prepare",
                                  status="aborted")

    log.info("Prepare session completed (continuations=%d)", continuation_count)

    if result is None:
        return {
            "prepare_status": "abort",
            "next_stage": "prepare",
            "history": [{"stage": "prepare", "loop_index": 0, "status": "abort"}],
        }

    validation = validate_prepare_artifacts(workspace)
    status = "ready" if validation.ok else "abort"
    abort_reason = "" if validation.ok else (validation.reason_code or result.recovery_abort_reason)
    abort_detail = "" if validation.ok else validation.detail
    if status == "ready":
        update_session_map_status(session_data_dir, stage="prepare",
                                  status="completed")
    summary_path = workspace / "prepare" / "summary.md"

    _write_stage_result(workspace, "prepare", status)

    return {
        "prepare_status": status,
        "prepare_abort_reason": abort_reason,
        "prepare_abort_detail": abort_detail,
        "prepare_summary_path": str(summary_path) if summary_path.exists() else "",
        "next_stage": "prepare",
        "history": [{"stage": "prepare", "loop_index": 0, "status": status}],
    }


async def _monitor_questions(adapter, session_id: str, workspace: Path) -> None:
    """Poll for prepare/question.json and handle the answer lifecycle.

    The adapter's pause_check prevents auto-continue while question.json
    exists. The question event is pushed to the TUI by the adapter when it
    confirms the agent has ended its turn and is waiting for input.

    This coroutine handles:
    - Headless mode: reading answer.json and forwarding it via adapter.send
    - Cleaning up question.json / answer.json after the answer is delivered
    """
    question_path = workspace / "prepare" / "question.json"
    answer_path = workspace / "prepare" / "answer.json"

    while True:
        await asyncio.sleep(_QUESTION_POLL_INTERVAL)

        if validate_prepare_artifacts(workspace).ok:
            return

        if not question_path.exists():
            continue

        # In headless mode, check for answer.json written by the user.
        # In TUI mode, the answer arrives via adapter.send() directly,
        # and the TUI deletes question.json after sending.
        if answer_path.exists():
            try:
                answer = json.loads(answer_path.read_text(encoding="utf-8"))
                answer_text = answer.get("custom_text", "") or answer.get("selected", "")
                if answer_text:
                    await adapter.send(session_id, answer_text)
            except (json.JSONDecodeError, OSError):
                pass
            finally:
                _safe_unlink(question_path)
                _safe_unlink(answer_path)


def _try_parse_question(path: Path) -> dict | None:
    """Try to parse question.json, returning None on failure."""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, dict) and "question" in data:
            return data
    except (json.JSONDecodeError, OSError):
        pass
    return None


def _push_question_event(question_data: dict) -> None:
    """Push a prepare_question event to the TUI and console."""
    from ..runtime import get_event_queue
    queue = get_event_queue()
    if queue is not None:
        from ..acp.protocol import SessionEvent
        event = SessionEvent(
            type="prepare_question",
            data=question_data,
            timestamp=time.time(),
        )
        queue.put_nowait(("prepare", event))
    q = question_data.get("question", "")
    opts = question_data.get("options", [])
    print(f"\n{'='*60}")
    print(f"AGENT QUESTION: {q}")
    for i, opt in enumerate(opts, 1):
        label = opt.get("label", "") if isinstance(opt, dict) else str(opt)
        desc = opt.get("description", "") if isinstance(opt, dict) else ""
        print(f"  {i}. {label}" + (f" - {desc}" if desc else ""))
    print(f"{'='*60}\n")


def _safe_unlink(path: Path) -> None:
    """Delete a file, ignoring errors if it doesn't exist."""
    try:
        path.unlink(missing_ok=True)
    except OSError:
        pass


def _write_stage_result(workspace: Path, stage: str, status: str) -> None:
    """Write a stage result file for the monitor to read."""
    from datetime import datetime, timezone
    result = {
        "status": status,
        "completed_at": datetime.now(tz=timezone.utc).isoformat(),
    }
    path = workspace / f"{stage}_result.json"
    path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    write_pipeline_state(
        current_loop_index=0,
        current_stage=stage,
        pipeline_status="running",
    )
