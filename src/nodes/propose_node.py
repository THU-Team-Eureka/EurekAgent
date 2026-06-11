"""Propose node: runs a single Claude Code session to generate K approaches."""

from __future__ import annotations


import logging
import time
import uuid
from pathlib import Path

from ..acp.factory import make_adapter
from ..acp.protocol import SessionRequest
from ..duration import parse_time_limit, resolve_stage_time_limit
from ..gpu_cleanup import sweep_gpu_locks
from ..artifacts import validate_propose_artifacts
from ..time_budget import write_time_budget
from ..prompts.propose import build_propose_brief
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
from ..context_continuation import detect_context_exhaustion, build_resume_preamble, build_continuation_request
from ..history import (
    current_manifest_path,
    read_manifest,
    resolve_ranked_history_path,
    snapshot_manifest_for_loop,
)
from ..session.manager import SessionManager
from ..session_map import (
    add_session_map_elapsed,
    load_session_map,
    resolve_stage_elapsed,
    session_remaining_seconds,
    update_session_map_status,
    write_session_map,
)
from ..state import LoopState

log = logging.getLogger(__name__)

_PROPOSE_WARNING_PROMPT = (
    "TIME WARNING: Only 5 minutes remain in your session. If you are still "
    "exploring or researching approaches, stop immediately. Review the required "
    "deliverables in your skill description and write them now. Incomplete or "
    "missing deliverables will cause this stage to abort and prevent progression "
    "to the next stage."
)


def scaffold_approach_dirs(workspace: Path, manifest_path: Path) -> None:
    """Create approach_details/<id>/code/, best_result.jsonl, and intermediate_results.jsonl
    for each approach in the manifest. Skips files that already exist.
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
        ad.mkdir(parents=True, exist_ok=True)
        (ad / "code").mkdir(exist_ok=True)


def validate_approach_details(workspace: Path, manifest: dict) -> str | None:
    """Require a non-empty approach.md for every proposed approach."""
    loop_index = int(str(manifest.get("round_number") or 0) or 0)
    if loop_index:
        result = validate_propose_artifacts(
            workspace, loop_index, max_num_approaches=10_000,
        )
        if result.ok:
            return None
        return result.reason_code
    for approach in manifest.get("approaches", []):
        if not isinstance(approach, dict):
            continue
        aid = str(approach.get("id", "")).strip()
        if not aid:
            continue
        path = workspace / "approach_details" / aid / "approach.md"
        if not path.exists():
            return f"missing_approach_md:{aid}"
        try:
            text = path.read_text(encoding="utf-8")
        except OSError:
            return f"unreadable_approach_md:{aid}"
        if not text.strip():
            return f"empty_approach_md:{aid}"
    return None


def _finalize_propose_outputs(
    workspace: Path,
    manifest_path: Path,
    loop_index: int,
    *,
    max_count: int,
) -> tuple[str, str, Path | None]:
    """Validate and snapshot propose outputs before implement can start."""
    validation = validate_propose_artifacts(
        workspace, loop_index, max_num_approaches=max_count,
        manifest_path=manifest_path,
    )
    if not validation.ok:
        return "abort", validation.reason_code, None

    loop_manifest = snapshot_manifest_for_loop(
        workspace, loop_index, manifest_path, sync_current=True,
    )
    if loop_manifest is None:
        return "abort", "manifest_round_mismatch", None

    scaffold_approach_dirs(workspace, manifest_path)
    return "ready", "", loop_manifest


def _propose_outputs_ready(workspace: Path, loop_index: int, max_count: int) -> bool:
    return validate_propose_artifacts(
        workspace, loop_index, max_num_approaches=max_count,
    ).ok


async def propose_node(state: LoopState) -> dict:
    """LangGraph node: propose K candidate approaches for this round."""
    config = get_config()
    workspace = Path(state["workspace_dir"]).resolve()
    loop_index = state.get("loop_index", 0) + 1

    log.info("Propose node starting: loop_index=%d, workspace=%s", loop_index, workspace)

    time_limit_str = resolve_stage_time_limit(config, "propose")
    timeout = parse_time_limit(time_limit_str)

    run_dir = Path(state["run_dir"]).resolve()
    session_data_dir = run_dir / "session_data"
    log_dir = run_dir / "session_data" / "session_transcripts" / f"loop_{loop_index}"
    prior_elapsed = resolve_stage_elapsed(
        session_data_dir, stage="propose", loop_index=loop_index,
        transcript_path=log_dir / "propose.jsonl",
    )
    pre_existing_map = load_session_map(session_data_dir, stage="propose",
                                        loop_index=loop_index)
    displayed_timeout = timeout
    if pre_existing_map and pre_existing_map.get("time_budget_seconds") is not None:
        if pre_existing_map.get("resume_extra_seconds"):
            displayed_timeout = float(pre_existing_map["time_budget_seconds"])

    push_stage_event(
        "propose", loop_index, time_limit_seconds=displayed_timeout,
        workspace_dir=state.get("workspace_dir", ""),
        prior_elapsed_seconds=prior_elapsed,
    )

    # Clear stale manifest from a previous round so a crashed propose
    # session doesn't leave the system with outdated approach data.
    if loop_index > 1:
        old_manifest = current_manifest_path(workspace)
        snapshot_manifest_for_loop(workspace, loop_index - 1, old_manifest)
        if old_manifest.exists():
            old_manifest.unlink()

    log.info("Building propose brief for loop %d", loop_index)
    brief = build_propose_brief(state, loop_index, time_limit=time_limit_str)
    log.info("Propose brief built: %d bytes", len(brief.encode()))

    # All engine-captured CLI transcripts for a run live under
    # workspace/session_transcripts/loop_<N>/. Propose is a single session
    # per loop; implement emits one file per approach under the same dir.
    log_dir.mkdir(parents=True, exist_ok=True)
    # `claude -p` does not execute slash commands, so ask the agent to invoke
    # the Skill tool explicitly. This preserves the TUI's "Tool: Skill" visual
    # while guaranteeing SKILL.md content is loaded into the agent's context.
    max_loops = state.get("max_loops", 1)
    prompt = (
        f"You are a proposal agent in an auto-experiment optimization loop "
        f"(round {loop_index}/{max_loops}). Your task is to propose diverse "
        f"optimization approaches.\n\n"
        f"## Run Context\n\n"
        f"{brief}\n\n"
        f"## Instruction\n\n"
        f"Invoke the `propose-approaches` skill using the Skill tool without any "
        f"arguments. The skill will guide you through the full workflow using the "
        f"run context above."
    )
    log.info("Propose prompt size: %d bytes (loop %d)", len(prompt.encode()), loop_index)
    # Pre-assign the session_id so we can register the active session with
    # the runtime BEFORE run_single starts the subprocess. That gives the
    # TUI a stable handle for routing pause-and-send messages as soon as
    # the agent comes online.
    session_id = str(uuid.uuid4())

    existing_map = pre_existing_map
    should_resume = (
        existing_map is not None
        and existing_map.get("status") in ("aborted", "running")
    )
    if should_resume:
        session_id = existing_map["session_id"]
        log.info("Resuming propose session for loop %d: %s (status was %s)",
                 loop_index, session_id, existing_map.get("status"))
        push_resume_history(log_dir / "propose.jsonl", context_key="propose")
        prior_elapsed = resolve_stage_elapsed(
            session_data_dir, stage="propose", loop_index=loop_index,
            transcript_path=log_dir / "propose.jsonl",
        )

    write_session_map(
        session_data_dir, session_id=session_id, stage="propose",
        loop_index=loop_index, time_budget_seconds=timeout,
        elapsed_seconds=prior_elapsed,
    )
    active_map = load_session_map(session_data_dir, stage="propose",
                                  loop_index=loop_index)
    effective_timeout = (
        float(active_map.get("time_budget_seconds"))
        if active_map and active_map.get("time_budget_seconds") is not None
        else timeout
    )

    remaining_time = (
        session_remaining_seconds(
            {"time_budget_seconds": effective_timeout, "elapsed_seconds": prior_elapsed},
        )
        if effective_timeout is not None else None
    )

    manifest_path = current_manifest_path(workspace)

    if effective_timeout is not None and prior_elapsed >= effective_timeout:
        log.info(
            "Propose budget exhausted for loop %d (%.0fs / %.0fs), completing stage",
            loop_index, prior_elapsed, effective_timeout,
        )
        validation = validate_propose_artifacts(
            workspace, loop_index, max_num_approaches=config.max_num_approaches,
            manifest_path=manifest_path,
        )
        status = "ready" if validation.ok else "abort"
        abort_reason = "" if validation.ok else validation.reason_code
        abort_detail = "" if validation.ok else validation.detail
        loop_manifest = None
        if status == "ready":
            status, abort_reason, loop_manifest = _finalize_propose_outputs(
                workspace,
                manifest_path,
                loop_index,
                max_count=config.max_num_approaches,
            )
            if status == "ready":
                update_session_map_status(session_data_dir, stage="propose",
                                          loop_index=loop_index, status="completed")
            else:
                update_session_map_status(session_data_dir, stage="propose",
                                          loop_index=loop_index, status="aborted")
        write_pipeline_state(
            current_loop_index=loop_index,
            current_stage="propose",
            pipeline_status="running",
        )
        return {
            "loop_index": loop_index,
            "next_stage": "propose",
            "propose_status": status,
            "propose_abort_reason": abort_reason,
            "propose_abort_detail": abort_detail,
            "manifest_path": str(loop_manifest or manifest_path),
            "ranked_history_path": str(resolve_ranked_history_path(workspace)),
            "history": [{"stage": "propose", "loop_index": loop_index, "status": status}],
        }

    if effective_timeout is not None:
        now = time.time()
        write_time_budget(
            workspace,
            started_at=now - prior_elapsed,
            total_seconds=effective_timeout,
            stage="propose",
            deadline_at=now + (remaining_time or 0.0),
            elapsed_seconds=prior_elapsed,
        )

    request = SessionRequest(
        prompt=prompt,
        model=state.get("model", ""),
        cwd=str(workspace),
        session_id=session_id,
        # Bypass mode: enforced via --dangerously-skip-permissions (ACP) or
        # settings.local.json (PTY/docker).

        permissions="bypassPermissions",
        log_path=str(log_dir / "propose.jsonl"),
        completion_check=(
            lambda ws=workspace, li=loop_index, mc=config.max_num_approaches:
            _propose_outputs_ready(ws, li, mc)
        ),
        warning_prompt=_PROPOSE_WARNING_PROMPT,
        resume=should_resume,
    )

    adapter = make_adapter(config, workspace)
    manager = SessionManager(adapter, config, event_queue=get_event_queue(), cost_limit=state.get("cost_limit"))

    register_active_session("propose", session_id)
    set_active_adapter(adapter)
    result = None
    attempt_started_at: float | None = None
    try:
        continuation_count = 0
        current_request = request

        while remaining_time is None or remaining_time > 0:
            attempt_started_at = time.time()
            result = await manager.run_single(current_request, timeout=remaining_time)
            elapsed = time.time() - attempt_started_at
            add_session_map_elapsed(
                session_data_dir, stage="propose", loop_index=loop_index,
                add_seconds=elapsed,
            )
            attempt_started_at = None
            if remaining_time is not None:
                remaining_time = session_remaining_seconds(
                    load_session_map(session_data_dir, stage="propose",
                                     loop_index=loop_index),
                )

            manifest_path = current_manifest_path(workspace)
            if _propose_outputs_ready(workspace, loop_index, config.max_num_approaches):
                break

            if detect_context_exhaustion(result.log_path):
                continuation_count += 1
                log.warning("Propose session hit context limit, starting continuation #%d",
                            continuation_count)
                resume_preamble = build_resume_preamble(
                    "propose", workspace, continuation_count, loop_index=loop_index,
                )
                current_request = build_continuation_request(
                    request, resume_preamble, remaining_time or 0, workspace,
                )
                continue
            else:
                break
    finally:
        unregister_active_session("propose")
        set_active_adapter(None)
        sweep_gpu_locks(workspace, stage="propose")
        if attempt_started_at is not None:
            add_session_map_elapsed(
                session_data_dir,
                stage="propose",
                loop_index=loop_index,
                add_seconds=time.time() - attempt_started_at,
            )
        # Mark aborted on any non-normal exit (cancel, interrupt, etc.).
        # Overwritten to "completed" below on normal exit.
        update_session_map_status(session_data_dir, stage="propose",
                                  loop_index=loop_index,
                                  status="aborted")

    if result is None:
        return {
            "loop_index": loop_index,
            "next_stage": "propose",
            "propose_status": "abort",
            "history": [{"stage": "propose", "loop_index": loop_index, "status": "abort"}],
        }

    log.info(
        "Propose session finished: exit_code=%s elapsed=%.1fs",
        result.exit_code,
        result.elapsed_seconds,
    )

    manifest_path = current_manifest_path(workspace)
    validation = validate_propose_artifacts(
        workspace, loop_index, max_num_approaches=config.max_num_approaches,
        manifest_path=manifest_path,
    )
    status = "ready" if validation.ok else "abort"
    loop_manifest = None
    abort_reason = "" if validation.ok else validation.reason_code
    abort_detail = "" if validation.ok else validation.detail

    if status == "ready":
        status, abort_reason, loop_manifest = _finalize_propose_outputs(
            workspace,
            manifest_path,
            loop_index,
            max_count=config.max_num_approaches,
        )
        if status == "ready":
            update_session_map_status(session_data_dir, stage="propose",
                                      loop_index=loop_index,
                                      status="completed")

    if status == "abort":
        log.warning("Propose stage artifacts incomplete: %s", abort_detail or abort_reason)
    if status == "abort" and not abort_reason:
        if timeout is not None and result.elapsed_seconds >= timeout:
            abort_reason = "missing_manifest_after_timeout"
        else:
            abort_reason = result.recovery_abort_reason or "missing_required_artifact"
    if status == "abort" and not abort_detail:
        abort_detail = abort_reason

    # Update pipeline state so the monitor refreshes.
    write_pipeline_state(
        current_loop_index=loop_index,
        current_stage="propose",
        pipeline_status="running",
    )

    return {
        "loop_index": loop_index,
        "next_stage": "propose",
        "propose_status": status,
        "propose_abort_reason": abort_reason,
        "propose_abort_detail": abort_detail,
        "manifest_path": str(loop_manifest or manifest_path),
        "ranked_history_path": str(resolve_ranked_history_path(workspace)),
        "history": [{"stage": "propose", "loop_index": loop_index, "status": status}],
    }
