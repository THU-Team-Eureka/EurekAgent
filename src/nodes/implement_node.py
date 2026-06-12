"""Implement node: runs K parallel Claude Code sessions, one per approach."""

from __future__ import annotations

import logging
import time
import uuid
from typing import Any
from pathlib import Path

from ..acp.factory import make_adapter
from ..acp.protocol import SessionRequest
from ..artifacts import (
    best_result_is_valid,
    read_best_score,
    validate_implement_artifacts,
    validate_propose_artifacts,
)
from ..context_continuation import detect_context_exhaustion, build_resume_preamble, build_continuation_request
from ..duration import parse_time_limit, resolve_stage_time_limit
from ..gpu_cleanup import sweep_gpu_locks
from ..history import (
    append_to_history,
    collect_round_entries,
    read_manifest,
    resolve_loop_manifest,
    resolve_ranked_history_path,
)
from ..time_budget import write_time_budget
from ..prompts.implement import build_implement_brief
from ..ranking import _load_is_better, grader_is_better_client, rank_history
from ..runtime import (
    get_config,
    get_event_queue,
    push_approach_results_event,
    push_approaches_event,
    push_resume_history,
    push_stage_event,
    register_active_session,
    set_active_adapter,
    unregister_active_session,
)
from ..session.manager import SessionManager
from ..session_map import (
    add_session_map_elapsed,
    load_session_map,
    max_loop_implement_elapsed,
    resolve_stage_elapsed,
    session_remaining_seconds,
    update_session_map_status,
    write_session_map,
)
from ..state import LoopState

log = logging.getLogger(__name__)

_IMPLEMENT_WARNING_PROMPT = (
    "TIME WARNING: Only 5 minutes remain in your session. If you are still "
    "developing your code, stop now and ensure you have submitted at least one "
    "candidate through eureka_score() or eureka_submit.py to get a scored result. "
    "Review the required deliverables in your skill description. If no valid "
    "submission is graded before time runs out, your approach will have no scored "
    "result."
)


def _ranking_is_better(config) -> Any:
    grader_url = getattr(
        config,
        "_controller_grader_url",
        getattr(config, "_grader_url", ""),
    )
    grader_token = getattr(config, "_grader_token", "")
    if grader_url and grader_token:
        return grader_is_better_client(grader_url, grader_token)
    return _load_is_better(config.hidden_eval_dir)


def scaffold_implement_dirs(workspace: Path, manifest_path: Path) -> None:
    """Create submissions/, eval_feedback/, logs/ directories for each approach.
    Called before implement sessions launch so agents don't waste tokens on mkdir.
    """
    manifest = read_manifest(manifest_path)
    if not manifest:
        return
    for approach in manifest.get("approaches", []):
        if not isinstance(approach, dict):
            continue
        aid = str(approach.get("id", "")).strip()
        if not aid:
            continue
        ad = workspace / "approach_details" / aid
        for subdir in ("submissions", "eval_feedback", "logs"):
            (ad / subdir).mkdir(parents=True, exist_ok=True)


async def implement_node(state: LoopState) -> dict:
    """LangGraph node: implement and evaluate all approaches in parallel."""
    config = get_config()
    workspace = Path(state["workspace_dir"]).resolve()
    loop_index = state["loop_index"]
    time_limit_str = resolve_stage_time_limit(config, "implement")
    timeout = parse_time_limit(time_limit_str)

    run_dir = Path(state["run_dir"]).resolve()
    session_data_dir = run_dir / "session_data"
    log_dir = session_data_dir / "session_transcripts" / f"loop_{loop_index}"

    # Read manifest — tolerant to both single-JSON and JSON-Lines shapes,
    # and to the `approach_id` key variant, via history.read_manifest.
    manifest_path = resolve_loop_manifest(
        workspace, loop_index, session_data_dir=session_data_dir,
        sync_current=True,
    ) or Path(state["manifest_path"])
    manifest = read_manifest(manifest_path)
    if not manifest:
        log.error("Manifest not found or unparseable: %s", manifest_path)
        return _abort(loop_index, "missing_manifest", f"Manifest not found or unparseable: {manifest_path}")

    propose_validation = validate_propose_artifacts(
        workspace, loop_index, config.max_num_approaches, manifest_path=manifest_path,
    )
    if not propose_validation.ok:
        log.error("Cannot start implement; propose artifacts incomplete: %s", propose_validation.detail)
        return _abort(loop_index, propose_validation.reason_code, propose_validation.detail)

    scaffold_implement_dirs(workspace, manifest_path)

    # Initialize git repos for version tracking
    import subprocess as _sp
    for approach in manifest.get("approaches", []):
        if not isinstance(approach, dict):
            continue
        aid = str(approach.get("id", "")).strip()
        if not aid:
            continue
        code_dir = workspace / "approach_details" / aid / "code"
        code_dir.mkdir(parents=True, exist_ok=True)
        _sp.run(["git", "init"], cwd=str(code_dir), capture_output=True, check=False)

    approaches = [a for a in manifest.get("approaches", []) if isinstance(a, dict)]

    if not approaches:
        log.error("Manifest contains no approaches")
        return _abort(loop_index, "empty_manifest", "Manifest contains no approaches.")

    approach_ids = [
        str(a.get("id", "")).strip() for a in approaches
        if str(a.get("id", "")).strip()
    ]
    prior_elapsed = max_loop_implement_elapsed(
        session_data_dir, loop_index, approach_ids,
        transcript_dir=log_dir,
    )
    effective_timeout = timeout
    if effective_timeout is not None:
        for aid in approach_ids:
            existing = load_session_map(
                session_data_dir, stage="implement",
                loop_index=loop_index, approach_id=aid,
            )
            budget = existing.get("time_budget_seconds") if existing else None
            # Only prefer saved budget when resume-extra-time was granted;
            # otherwise the CLI value is authoritative (user may shrink the
            # budget on resume).
            if budget is not None and existing and existing.get("resume_extra_seconds"):
                effective_timeout = max(float(effective_timeout), float(budget))

    push_stage_event(
        "implement", loop_index, time_limit_seconds=effective_timeout,
        workspace_dir=state.get("workspace_dir", ""),
        prior_elapsed_seconds=prior_elapsed,
    )

    remaining_time = (
        session_remaining_seconds(
            {"time_budget_seconds": effective_timeout, "elapsed_seconds": prior_elapsed},
        )
        if effective_timeout is not None else None
    )

    if effective_timeout is not None and prior_elapsed >= effective_timeout:
        log.info(
            "Implement budget exhausted for loop %d (%.0fs / %.0fs), finalizing round",
            loop_index, prior_elapsed, effective_timeout,
        )
        return _finalize_implement_round(
            workspace, manifest_path, approaches, loop_index,
            session_data_dir=session_data_dir, config=config,
        )

    if effective_timeout is not None:
        now = time.time()
        write_time_budget(
            workspace,
            started_at=now - prior_elapsed,
            total_seconds=effective_timeout,
            stage="implement",
            deadline_at=now + (remaining_time or 0.0),
            elapsed_seconds=prior_elapsed,
        )

    # Build per-approach session requests with pre-assigned session IDs
    requests: list[SessionRequest] = []
    session_map: dict[str, str] = {}  # session_key → approach_id

    for approach in approaches:
        approach_id = str(approach.get("id", "")).strip()
        if not approach_id:
            continue

        # Claude CLI requires canonical dashed UUID; uuid.uuid4().hex (no dashes) is rejected.
        session_id = str(uuid.uuid4())
        session_map[session_id] = approach_id

        # Check for an aborted previous session to resume
        existing_map = load_session_map(session_data_dir, stage="implement",
                                        loop_index=loop_index,
                                        approach_id=approach_id)
        approach_should_resume = (
            existing_map is not None
            and existing_map.get("status") in ("aborted", "running")
        )
        approach_prior_elapsed = resolve_stage_elapsed(
            session_data_dir, stage="implement",
            loop_index=loop_index, approach_id=approach_id,
            transcript_path=log_dir / f"{approach_id}.jsonl",
        ) if approach_should_resume else 0.0
        if approach_should_resume:
            session_id = existing_map["session_id"]
            session_map[session_id] = approach_id
            log.info("Resuming implement session for %s: %s", approach_id, session_id)
            push_resume_history(
                log_dir / f"{approach_id}.jsonl",
                context_key=f"implement::{approach_id}",
            )

        # Write session map before starting
        write_session_map(
            session_data_dir, session_id=session_id, stage="implement",
            loop_index=loop_index, approach_id=approach_id,
            time_budget_seconds=effective_timeout,
            elapsed_seconds=approach_prior_elapsed,
        )

        brief = build_implement_brief(state, approach, time_limit=time_limit_str)
        # Engine CLI transcripts live under workspace/session_transcripts/.
        # Keep approach-private artifacts (code, experiment logs, results)
        # under approach_details/<id>/ where the SKILL puts them.
        log_dir.mkdir(parents=True, exist_ok=True)

        # `claude -p` does not execute slash commands, so ask the agent to
        # invoke the Skill tool explicitly (see propose_node for context).
        prompt = (
            f"You are an implementation agent in an auto-experiment optimization loop "
            f"(round {loop_index}). Your task is to implement, evaluate, and iterate "
            f"from one initial hypothesis.\n\n"
            f"## Run Context\n\n"
            f"{brief}\n\n"
            f"## Instruction\n\n"
            f"Invoke the `implement-approach` skill using the Skill tool without any "
            f"arguments. The skill will guide you through the full workflow using the "
            f"run context above."
        )
        log.info("Implement prompt size for %s: %d bytes", approach_id, len(prompt.encode()))
        requests.append(SessionRequest(
            prompt=prompt,
            model=state.get("model", ""),
            cwd=str(workspace),
            session_id=session_id,
            # Bypass mode enforced via ACP flag or PTY settings.local.json.

            permissions="bypassPermissions",
            log_path=str(log_dir / f"{approach_id}.jsonl"),
            completion_check=lambda p=workspace / "approach_details" / approach_id / "best_result.jsonl": best_result_is_valid(p),
            persist_until_timeout=True,
            warning_prompt=_IMPLEMENT_WARNING_PROMPT,
            resume=approach_should_resume,
            max_recoveries=None,
            env={
                "EUREKA_STAGE": "implement",
                "EUREKA_CURRENT_LOOP_INDEX": str(loop_index),
                "EUREKA_CURRENT_APPROACH_ID": approach_id,
            },
        ))

    # Notify TUI about approaches before launching sessions
    push_approaches_event(
        loop_index, approaches, session_map, time_limit_seconds=effective_timeout,
        prior_elapsed_seconds=prior_elapsed,
    )

    # Run all in parallel with timeout
    adapter = make_adapter(config, workspace)
    manager = SessionManager(adapter, config, event_queue=get_event_queue(), cost_limit=state.get("cost_limit"))

    # Register sessions so the TUI can route user messages to each approach.
    for sid, aid in session_map.items():
        register_active_session(f"implement::{aid}", sid)
    set_active_adapter(adapter)

    results: dict[str, Any] = {}
    parallel_started_at: float | None = None
    try:
        parallel_started_at = time.time()
        results = await manager.run_parallel(requests, timeout=remaining_time)
    finally:
        for sid, aid in session_map.items():
            unregister_active_session(f"implement::{aid}")
        set_active_adapter(None)
        sweep_gpu_locks(workspace, stage="implement")
        recorded_aids: set[str] = set()
        for session_key, session_result in results.items():
            approach_id = session_map.get(session_key, "")
            if approach_id:
                add_session_map_elapsed(
                    session_data_dir, stage="implement",
                    loop_index=loop_index, approach_id=approach_id,
                    add_seconds=session_result.elapsed_seconds,
                )
                recorded_aids.add(approach_id)
        if parallel_started_at is not None:
            elapsed = time.time() - parallel_started_at
            for aid in approach_ids:
                if aid in recorded_aids:
                    continue
                add_session_map_elapsed(
                    session_data_dir,
                    stage="implement",
                    loop_index=loop_index,
                    approach_id=aid,
                    add_seconds=elapsed,
                )
        # Mark all approaches as aborted on any non-normal exit.
        # Overwritten to "completed" below for successful approaches.
        for approach in approaches:
            approach_id = str(approach.get("id", "")).strip()
            if not approach_id:
                continue
            update_session_map_status(session_data_dir, stage="implement",
                                      loop_index=loop_index,
                                      approach_id=approach_id,
                                      status="aborted")

    # Write history BEFORE abort check so approaches that did produce valid
    # results are recorded even when all sessions abort (e.g. systemic API
    # outage after partial success).
    history_path = resolve_ranked_history_path(workspace, for_write=True)
    entries = collect_round_entries(
        manifest_path, workspace_dir=workspace, loop_index=loop_index,
    )
    append_to_history(history_path, new_entries=entries)
    is_better_fn = _ranking_is_better(config)
    rank_history(workspace, history_path=history_path, is_better=is_better_fn)

    # Check if all approach sessions were aborted for the same consecutive
    # error type. If so, abort the entire run — each session having hit its
    # consecutive-failure threshold for the same reason (api_error /
    # no_response / etc.) is a strong signal of a systemic issue (API
    # outage, grader server failure, network partition) and continuing
    # would only waste token budget without producing usable results.
    abort_reasons = [r.recovery_abort_reason for r in results.values() if r.recovery_abort_reason]
    if len(abort_reasons) == len(results) and len(set(abort_reasons)) == 1:
        shared_reason = abort_reasons[0]
        log.error(
            "All %d approach sessions aborted with the same reason: '%s' "
            "(each hit the consecutive-same-failure threshold). Aborting run "
            "— this indicates a systemic issue (API outage / grader failure / "
            "network partition). Inspect session transcripts and run.log to diagnose.",
            len(results), shared_reason,
        )
        return {
            "next_stage": "implement",
            "implement_status": "abort",
            "implement_abort_reason": shared_reason,
            "history": [{"stage": "implement", "loop_index": loop_index,
                         "status": "abort", "reason": shared_reason}],
        }

    # Check for context-exhausted sessions and start continuations
    if effective_timeout is not None:
        accumulated = max_loop_implement_elapsed(
            session_data_dir, loop_index, approach_ids,
            transcript_dir=log_dir,
        )
        remaining_time = max(0.0, effective_timeout - accumulated)
    else:
        remaining_time = None

    for session_key, result in results.items():
        approach_id = session_map.get(session_key, "")
        if not approach_id:
            continue
        result_path = workspace / "approach_details" / approach_id / "best_result.jsonl"

        if best_result_is_valid(result_path):
            continue

        if not detect_context_exhaustion(result.log_path):
            continue

        # Find the original request for this approach
        orig_req = None
        for req in requests:
            if approach_id in req.prompt:
                orig_req = req
                break
        if orig_req is None:
            continue

        approach_remaining = remaining_time

        cont_count = 0

        while approach_remaining is None or approach_remaining > 0:
            cont_count += 1
            log.warning("Implement session for %s hit context limit, starting continuation #%d",
                        approach_id, cont_count)
            resume_preamble = build_resume_preamble(
                "implement", workspace, cont_count,
                approach_id=approach_id, loop_index=loop_index,
            )
            cont_request = build_continuation_request(
                orig_req, resume_preamble, approach_remaining or 0, workspace,
            )
            started = time.time()
            cont_adapter = make_adapter(config, workspace)
            cont_manager = SessionManager(cont_adapter, config, event_queue=get_event_queue(), cost_limit=state.get("cost_limit"))
            cont_result = None
            try:
                cont_result = await cont_manager.run_single(cont_request, timeout=approach_remaining)
            finally:
                add_session_map_elapsed(
                    session_data_dir,
                    stage="implement",
                    loop_index=loop_index,
                    approach_id=approach_id,
                    add_seconds=time.time() - started,
                )
            elapsed = time.time() - started
            if approach_remaining is not None:
                approach_remaining -= elapsed

            if best_result_is_valid(result_path):
                break

            if not detect_context_exhaustion(cont_result.log_path):
                break

    log.info("Implement stage: %d sessions completed", len(results))

    # Collect results — only count approaches with a non-empty best_result.jsonl
    # whose "valid" field is True.
    validation = validate_implement_artifacts(
        workspace, loop_index, manifest_path,
        max_num_approaches=config.max_num_approaches,
    )
    approach_ids_with_results = list(validation.metadata.get("succeeded_ids") or [])

    # Update session map status: promote aborted → completed for successful approaches
    for approach in approaches:
        approach_id = str(approach.get("id", "")).strip()
        if not approach_id:
            continue
        if approach_id in approach_ids_with_results:
            update_session_map_status(session_data_dir, stage="implement",
                                      loop_index=loop_index,
                                      approach_id=approach_id,
                                      status="completed")

    # Push per-approach results to the TUI so it can show completed/failed
    # status and scores instead of marking everything "failed".
    approach_results: dict[str, dict] = {}
    for approach in approaches:
        approach_id = str(approach.get("id", "")).strip()
        if not approach_id:
            continue
        if approach_id in approach_ids_with_results:
            score = read_best_score(workspace / "approach_details" / approach_id)
            approach_results[approach_id] = {"status": "completed", "score": score}
        else:
            approach_results[approach_id] = {"status": "failed", "score": None}
    push_approach_results_event(approach_results)

    # Determine status (informational — routing is based solely on loop limit)
    total = len(approaches)
    succeeded = len(approach_ids_with_results)
    if succeeded == total:
        status = "all_succeeded"
    elif succeeded > 0:
        status = "partial_succeeded"
    else:
        status = "all_failed"
        log.warning("No approach produced a best_result.jsonl")

    # Update pipeline state so the monitor refreshes.
    from ..runtime import write_pipeline_state
    write_pipeline_state(
        current_loop_index=loop_index,
        current_stage="implement",
        pipeline_status="running",
    )

    return {
        "next_stage": "implement",
        "implement_status": status,
        "history": [{
            "stage": "implement",
            "loop_index": loop_index,
            "status": status,
            "approaches_with_results": approach_ids_with_results,
            "artifact_detail": validation.detail,
        }],
    }


def _finalize_implement_round(
    workspace: Path,
    manifest_path: Path,
    approaches: list[dict],
    loop_index: int,
    *,
    session_data_dir: Path,
    config: Any,
) -> dict:
    """Complete implement when time budget is already exhausted (no relaunch)."""
    history_path = resolve_ranked_history_path(workspace, for_write=True)
    entries = collect_round_entries(
        manifest_path, workspace_dir=workspace, loop_index=loop_index,
    )
    append_to_history(history_path, new_entries=entries)
    is_better_fn = _ranking_is_better(config)
    rank_history(workspace, history_path=history_path, is_better=is_better_fn)

    validation = validate_implement_artifacts(
        workspace, loop_index, manifest_path,
        max_num_approaches=config.max_num_approaches,
    )
    approach_ids_with_results = list(validation.metadata.get("succeeded_ids") or [])
    for approach_id in approach_ids_with_results:
        update_session_map_status(
            session_data_dir, stage="implement",
            loop_index=loop_index, approach_id=approach_id,
            status="completed",
        )

    approach_results: dict[str, dict] = {}
    for approach in approaches:
        approach_id = str(approach.get("id", "")).strip()
        if not approach_id:
            continue
        if approach_id in approach_ids_with_results:
            score = read_best_score(workspace / "approach_details" / approach_id)
            approach_results[approach_id] = {"status": "completed", "score": score}
        else:
            approach_results[approach_id] = {"status": "failed", "score": None}
    push_approach_results_event(approach_results)

    total = len(approaches)
    succeeded = len(approach_ids_with_results)
    if succeeded == total:
        status = "all_succeeded"
    elif succeeded > 0:
        status = "partial_succeeded"
    else:
        status = "all_failed"

    from ..runtime import write_pipeline_state
    write_pipeline_state(
        current_loop_index=loop_index,
        current_stage="implement",
        pipeline_status="running",
    )

    return {
        "next_stage": "implement",
        "implement_status": status,
        "history": [{
            "stage": "implement",
            "loop_index": loop_index,
            "status": status,
            "approaches_with_results": approach_ids_with_results,
            "artifact_detail": validation.detail,
        }],
    }


def _abort(loop_index: int, reason: str = "missing_required_artifact", detail: str = "") -> dict:
    return {
        "loop_index": loop_index,
        "next_stage": "implement",
        "implement_status": "abort",
        "implement_abort_reason": reason,
        "implement_abort_detail": detail,
        "history": [{
            "stage": "implement",
            "loop_index": loop_index,
            "status": "abort",
            "reason": reason,
            "artifact_detail": detail,
        }],
    }


def _is_valid_result(result_path: Path) -> bool:
    """Check if best_result.jsonl exists, is non-empty, and valid==True."""
    return best_result_is_valid(result_path)


def _read_best_score(approach_dir: Path) -> float | None:
    """Read the score from an approach's best_result.jsonl."""
    return read_best_score(approach_dir)


def _coerce_score(val: Any) -> float | None:
    try:
        return float(val)
    except (TypeError, ValueError):
        return None
