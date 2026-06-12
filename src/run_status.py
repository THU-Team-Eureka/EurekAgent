"""Run terminal status, summary context, and run-ended event emission."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Any, Callable

from .artifacts import artifact_reason_text
from .history import load_ranked_history, resolve_ranked_history_path
from .ranking import grader_is_better_client
from .runtime import get_token_tracker, push_run_end_event
from .token_tracker import aggregate_token_usage, format_token_count, format_token_summary


def finalize_run(
    *,
    run_id: str,
    run_dir: Path,
    final_state: dict[str, Any],
    max_loops: int,
    config: Any,
    save_summary: Callable[[Path, dict[str, Any]], None],
) -> tuple[str, str]:
    """Persist final state and notify terminal UIs through one shared path."""
    save_summary(run_dir, final_state)
    status, last_stage = resolve_run_status(final_state)
    ctx = terminal_context(
        final_state=final_state, run_dir=run_dir, max_loops=max_loops,
        last_stage=last_stage, config=config,
    )
    push_run_end_event(status, reason=run_end_reason(final_state), **ctx)
    return status, last_stage


def resolve_run_status(state: dict[str, Any]) -> tuple[str, str]:
    """Return (terminal_status, last_stage) from final graph state."""
    explicit = state.get("status")
    if explicit in ("interrupted", "error"):
        return explicit, str(state.get("next_stage") or "")
    if state.get("end_reason") == "cost_limit" or _cost_limit_reached(state):
        return "abort", ""

    history = state.get("history") or []
    if isinstance(history, list):
        for entry in reversed(history):
            if not isinstance(entry, dict):
                continue
            stage = str(entry.get("stage") or "")
            status = str(entry.get("status") or "")
            if stage == "prepare" and status == "abort":
                return "abort", "prepare"
            if stage == "propose" and status == "abort":
                return "abort", "propose"
            if stage == "implement":
                if status in ("abort", "all_failed"):
                    return "abort", "implement"
                if status:
                    return status, "implement"

    prepare = state.get("prepare_status")
    if prepare == "abort":
        return "abort", "prepare"
    propose = state.get("propose_status")
    if propose == "abort":
        return "abort", "propose"
    implement = state.get("implement_status")
    if implement == "abort" or implement == "all_failed":
        return "abort", "implement"
    if implement is not None:
        return str(implement), "implement"
    if propose is not None:
        return str(propose), "propose"
    if prepare is not None:
        return str(prepare), "prepare"
    return "unknown", ""


def abort_reason_text(reason: str) -> str:
    artifact_text = artifact_reason_text(reason)
    if artifact_text:
        return artifact_text
    return {
        "api_error": "API connectivity failure",
        "no_response": "agent not responding to continue prompts",
        "budget_exhausted": "recovery budget exhausted",
        "context_exhausted": "context window limit",
        "missing_required_artifact": "required artifact missing",
        "missing_manifest_after_timeout": "time budget exhausted before manifest was written",
    }.get(reason, "unknown error")


def _cost_limit_reached(state: dict[str, Any]) -> bool:
    """Re-check cost limit from the token tracker.

    Conditional-edge routing functions cannot persist state mutations in
    LangGraph, so ``end_reason == "cost_limit"`` may be lost.  This fallback
    re-queries the tracker when the state field is empty.
    """
    if state.get("end_reason") == "cost_limit":
        return True
    cost_limit = state.get("cost_limit")
    if cost_limit is None:
        return False
    tracker = get_token_tracker()
    current_cost = tracker.calculate_cost()
    return current_cost is not None and current_cost >= cost_limit


def run_end_reason(state: dict[str, Any]) -> str:
    """Human-readable explanation of why the pipeline terminated."""
    if state.get("end_reason") == "cost_limit":
        return "Cost limit reached"
    if state.get("prepare_status") == "abort":
        reason = state.get("prepare_abort_reason", "")
        return f"Prepare aborted - {abort_reason_text(reason)}" if reason else "Prepare aborted"
    if state.get("propose_status") == "abort":
        reason = state.get("propose_abort_reason", "")
        return f"Propose aborted - {abort_reason_text(reason)}" if reason else "Propose aborted"
    if state.get("implement_status") == "abort":
        reason = state.get("implement_abort_reason", "")
        return f"Implement aborted - {abort_reason_text(reason)}" if reason else "Implement aborted"
    impl = state.get("implement_status", "")
    if impl in ("all_succeeded", "partial_succeeded"):
        if _cost_limit_reached(state):
            return "Cost limit reached"
        return "max-loops reached"
    if impl == "all_failed":
        return "No approach produced a valid best_result"
    return ""


def round_label(state: dict[str, Any], max_loops: int) -> str:
    loop_index = int(state.get("loop_index", 0) or 0)
    return f"{loop_index + 1}/{max_loops}" if max_loops else ""


def terminal_context(
    *,
    final_state: dict[str, Any],
    run_dir: Path,
    max_loops: int,
    last_stage: str = "",
    config: Any | None = None,
) -> dict[str, Any]:
    """Build the context payload consumed by the TUI terminal banner."""
    loop_index = int(final_state.get("loop_index", 0) or 0)
    display_round = f"{loop_index}/{max_loops}" if loop_index else ""
    propose_status = final_state.get("propose_status")
    implement_status = final_state.get("implement_status")

    if last_stage == "prepare":
        stage = "prepare"
        abort_reason = final_state.get("prepare_abort_reason", "")
        abort_detail = final_state.get("prepare_abort_detail", "")
        if abort_reason:
            detail = str(abort_detail or f"Prepare aborted - {abort_reason_text(abort_reason)}")
        elif final_state.get("prepare_status") == "ready":
            detail = "Workspace prepared"
        else:
            detail = "Pipeline ended before propose stage"
    elif last_stage == "propose":
        stage = "propose"
        abort_reason = final_state.get("propose_abort_reason", "")
        abort_detail = final_state.get("propose_abort_detail", "")
        if propose_status == "ready":
            detail = "Manifest produced"
        elif abort_reason:
            detail = str(abort_detail or f"Propose aborted - {abort_reason_text(abort_reason)}")
        else:
            detail = "No manifest written to round_state/"
    elif last_stage == "implement":
        stage = "implement"
        history = final_state.get("history") or []
        last_impl = next(
            (h for h in reversed(history) if isinstance(h, dict) and h.get("stage") == "implement"),
            {},
        )
        attempted = final_state.get("max_num_approaches") or 0
        produced = len(last_impl.get("approaches_with_results") or [])
        if implement_status in ("all_succeeded", "partial_succeeded"):
            detail = ""
        elif implement_status == "abort":
            reason = final_state.get("implement_abort_reason", "")
            abort_detail = final_state.get("implement_abort_detail", "")
            detail = str(abort_detail or (f"Implement aborted - {abort_reason_text(reason)}" if reason else "Implement aborted"))
        else:
            detail = str(last_impl.get("artifact_detail") or f"{produced} / {attempted} approaches produced a best_result")
    elif propose_status is None and implement_status is None:
        stage = "prepare"
        abort_reason = final_state.get("prepare_abort_reason", "")
        abort_detail = final_state.get("prepare_abort_detail", "")
        detail = (
            str(abort_detail or f"Prepare aborted - {abort_reason_text(abort_reason)}")
            if abort_reason else "Pipeline ended before propose stage"
        )
    else:
        stage = last_stage or "implement"
        detail = ""

    transcripts = run_dir / "session_data" / "session_transcripts"
    if not transcripts.is_dir():
        transcripts = run_dir / "workspace" / "session_transcripts"
    if loop_index:
        transcripts = transcripts / f"loop_{loop_index}"
    evidence_paths = [
        str(run_dir / "run.log"),
        str(transcripts) + "/",
        str(run_dir / "run_summary.json"),
    ]

    token_summary = _token_summary(run_dir)
    is_better = None
    if config and getattr(config, "hidden_eval_dir", ""):
        try:
            grader_url = getattr(
                config,
                "_controller_grader_url",
                getattr(config, "_grader_url", ""),
            )
            grader_token = getattr(config, "_grader_token", "")
            if grader_url and grader_token:
                is_better = grader_is_better_client(grader_url, grader_token)
            else:
                is_better = load_is_better(config.hidden_eval_dir)
        except Exception:
            pass
    best_score, best_approach_id, baseline_score, baseline_approach_id = (
        extract_score_summary(run_dir / "workspace", is_better=is_better)
    )

    return {
        "stage": stage,
        "round_label": display_round,
        "detail": detail,
        "evidence_paths": evidence_paths,
        "token_summary": token_summary,
        "best_score": best_score,
        "best_approach_id": best_approach_id,
        "baseline_score": baseline_score,
        "baseline_approach_id": baseline_approach_id,
    }


def load_is_better(hidden_eval_dir: str) -> Callable[[float, float], bool]:
    eval_path = Path(hidden_eval_dir) / "evaluate.py"
    spec = importlib.util.spec_from_file_location("_eval_is_better", eval_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {eval_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    fn = getattr(module, "is_better", None)
    if not callable(fn):
        raise RuntimeError(f"{eval_path} must define `is_better(new_score, old_score) -> bool`")
    return fn


def extract_score_summary(
    workspace: Path,
    is_better: Callable[[float, float], bool] | None = None,
) -> tuple[float | None, str, float | None, str]:
    """Extract best score and first valid baseline score across approaches."""
    details_dir = workspace / "approach_details"
    if not details_dir.is_dir():
        return None, "", None, ""

    best_score: float | None = None
    best_aid = ""
    round1_aids = {
        str(entry.get("approach_id", "")).strip()
        for entry in load_ranked_history(resolve_ranked_history_path(workspace))
        if entry.get("loop_index") == 1 and str(entry.get("approach_id", "")).strip()
    }

    for approach_dir in details_dir.iterdir():
        if not approach_dir.is_dir():
            continue
        data = _read_result_json(approach_dir / "best_result.jsonl")
        if not data or data.get("controller_status") != "secure_graded":
            continue
        if data.get("valid") is not True:
            continue
        score = _coerce_score(data.get("score"))
        if score is not None and (
            best_score is None
            or (is_better is not None and is_better(score, best_score))
        ):
            best_score = score
            best_aid = approach_dir.name

    baseline_score, baseline_aid = _find_first_valid_score(
        details_dir, filter_aids=round1_aids if round1_aids else None,
    )
    return best_score, best_aid, baseline_score, baseline_aid


def _token_summary(run_dir: Path) -> str:
    try:
        tracker = get_token_tracker()
        if any(getattr(tracker.totals, dim) for dim in (
            "input_tokens", "output_tokens", "cache_read_input_tokens",
            "cache_creation_input_tokens",
        )):
            return format_token_summary(tracker)
    except RuntimeError:
        pass
    usage = aggregate_token_usage(run_dir / "workspace")
    if not (usage.get("input_tokens") or usage.get("output_tokens")):
        return ""
    return (
        f"{format_token_count(usage.get('input_tokens', 0))} in · "
        f"{format_token_count(usage.get('output_tokens', 0))} out · "
        f"{format_token_count(usage.get('cache_read_input_tokens', 0))} cache_r"
    )


def _find_first_valid_score(
    details_dir: Path,
    *,
    filter_aids: set[str] | None = None,
) -> tuple[float | None, str]:
    earliest_time = ""
    earliest_score: float | None = None
    earliest_aid = ""
    for approach_dir in details_dir.iterdir():
        if not approach_dir.is_dir():
            continue
        aid = approach_dir.name
        if filter_aids is not None and aid not in filter_aids:
            continue
        result_path = approach_dir / "intermediate_results.jsonl"
        if not result_path.exists():
            continue
        try:
            lines = result_path.read_text(encoding="utf-8").splitlines()
        except OSError:
            continue
        for line in lines:
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if rec.get("valid") is not True or rec.get("controller_status") != "secure_graded":
                continue
            ts = rec.get("evaluated_at", "")
            if earliest_time and ts >= earliest_time:
                continue
            score = _coerce_score(rec.get("score"))
            if score is not None:
                earliest_score = score
                earliest_aid = aid
                earliest_time = ts
    return earliest_score, earliest_aid


def _read_result_json(path: Path) -> dict[str, Any] | None:
    try:
        text = path.read_text(encoding="utf-8").strip()
        data = json.loads(text)
    except (OSError, json.JSONDecodeError):
        return None
    return data if isinstance(data, dict) else None


def _coerce_score(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
