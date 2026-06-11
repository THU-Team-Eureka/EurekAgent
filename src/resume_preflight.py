"""Preflight checks for resume runs that cannot progress without extra time."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .artifacts import validate_implement_artifacts, validate_propose_artifacts
from .history import current_manifest_path, resolve_loop_manifest
from .resume_budget import effective_resume_budget
from .session_map import load_session_map, session_map_path

MIN_RESUME_EXTRA_SECONDS = 5 * 60


@dataclass
class ResumePreflightResult:
    ok: bool
    needs_extra_time: bool = False
    stage: str = ""
    loop_index: int = 0
    elapsed_seconds: float = 0.0
    budget_seconds: float = 0.0
    missing_artifacts: list[str] = field(default_factory=list)
    session_map_paths: list[Path] = field(default_factory=list)
    message: str = ""


def check_resume_preflight(
    run_dir: Path,
    state: dict[str, Any],
    config: Any,
) -> ResumePreflightResult:
    """Return whether resume can proceed with the current persisted budget."""
    workspace = run_dir / "workspace"
    session_data_dir = run_dir / "session_data"
    next_stage = state.get("next_stage")
    if next_stage == "propose":
        loop_index = int(state.get("loop_index", 0) or 0) + 1
        session_map = load_session_map(
            session_data_dir, stage="propose", loop_index=loop_index,
        )
        budget = effective_resume_budget(
            session_map, config.propose_time_limit_per_session,
        )
        elapsed = float((session_map or {}).get("elapsed_seconds") or 0)
        manifest = current_manifest_path(workspace)
        validation = validate_propose_artifacts(
            workspace, loop_index, config.max_num_approaches,
            manifest_path=manifest,
        )
        if budget is not None and elapsed >= budget and not validation.ok:
            return ResumePreflightResult(
                ok=False,
                needs_extra_time=True,
                stage="propose",
                loop_index=loop_index,
                elapsed_seconds=elapsed,
                budget_seconds=budget,
                missing_artifacts=validation.missing_artifacts + validation.invalid_artifacts,
                session_map_paths=[
                    session_map_path(session_data_dir, stage="propose", loop_index=loop_index)
                ],
                message=(
                    "The previous propose stage used its full time budget but "
                    "did not write all required propose artifacts."
                ),
            )
    elif next_stage == "implement":
        loop_index = int(state.get("loop_index", 0) or 0)
        result = _check_implement(run_dir, loop_index, config)
        if result.needs_extra_time:
            return result
    return ResumePreflightResult(ok=True)


def apply_resume_extra_time(
    result: ResumePreflightResult,
    extra_seconds: float,
) -> None:
    """Extend the effective budget of the blocked stage without resetting elapsed."""
    if not result.needs_extra_time:
        return
    if extra_seconds < MIN_RESUME_EXTRA_SECONDS:
        raise ValueError(
            f"Resume extra time must be at least {MIN_RESUME_EXTRA_SECONDS / 60:.0f} minutes"
        )
    now = datetime.now(tz=timezone.utc).isoformat()
    for path in result.session_map_paths:
        if not path.exists():
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        elapsed = max(float(payload.get("elapsed_seconds") or 0), result.elapsed_seconds)
        previous_extra = float(payload.get("resume_extra_seconds") or 0)
        payload["time_budget_seconds"] = elapsed + extra_seconds
        payload["resume_extra_seconds"] = previous_extra + extra_seconds
        payload["resume_extra_granted_at"] = now
        payload["resume_extra_reason"] = "missing_required_artifact_after_timeout"
        path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _check_implement(
    run_dir: Path,
    loop_index: int,
    config: Any,
) -> ResumePreflightResult:
    workspace = run_dir / "workspace"
    session_data_dir = run_dir / "session_data"
    maps_dir = session_data_dir / "session_maps"
    if not maps_dir.is_dir():
        return ResumePreflightResult(ok=True)

    blocked_paths: list[Path] = []
    missing: list[str] = []
    max_elapsed = 0.0
    max_budget = 0.0
    for path in maps_dir.glob(f"loop_{loop_index}_*_session_map.json"):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if payload.get("stage") != "implement":
            continue
        if payload.get("status") not in ("aborted", "running"):
            continue
        elapsed = float(payload.get("elapsed_seconds") or 0)
        budget = effective_resume_budget(payload, config.implement_time_limit_per_session)
        if budget is None or elapsed < budget:
            continue
        approach_id = str(payload.get("approach_id") or "")
        manifest_path = resolve_loop_manifest(
            workspace, loop_index, session_data_dir=session_data_dir,
        ) or (workspace / "round_state" / "current_round_approaches.jsonl")
        validation = validate_implement_artifacts(
            workspace, loop_index, manifest_path,
            max_num_approaches=config.max_num_approaches,
        )
        if approach_id in set(validation.metadata.get("succeeded_ids") or []):
            continue
        blocked_paths.append(path)
        result_path = workspace / "approach_details" / approach_id / "best_result.jsonl"
        missing.append(str(result_path))
        max_elapsed = max(max_elapsed, elapsed)
        max_budget = max(max_budget, budget)

    if not blocked_paths:
        return ResumePreflightResult(ok=True)
    return ResumePreflightResult(
        ok=False,
        needs_extra_time=True,
        stage="implement",
        loop_index=loop_index,
        elapsed_seconds=max_elapsed,
        budget_seconds=max_budget,
        missing_artifacts=missing,
        session_map_paths=blocked_paths,
        message=(
            "The previous implement stage used its full time budget but "
            "one or more approaches did not write a valid best_result."
        ),
    )

