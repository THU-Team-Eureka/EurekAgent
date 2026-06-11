"""Session map persistence for pipeline resume support.

Each stage attempt writes a small JSON file recording the Claude CLI session
ID, stage name, loop index, time budget, and status. On resume, nodes read
these files to recover the session ID and pass `resume=True` to the adapter.
"""

from __future__ import annotations

import json
from pathlib import Path


def session_map_path(
    session_data_dir: Path,
    *,
    stage: str,
    loop_index: int | None = None,
    approach_id: str | None = None,
) -> Path:
    """Return the path to the session map file for a given stage attempt."""
    maps_dir = session_data_dir / "session_maps"
    if stage == "prepare":
        return maps_dir / "prepare_session_map.json"
    if approach_id:
        return maps_dir / f"loop_{loop_index}_{approach_id}_session_map.json"
    return maps_dir / f"loop_{loop_index}_{stage}_session_map.json"


def resolve_stage_elapsed(
    session_data_dir: Path,
    *,
    stage: str,
    loop_index: int | None = None,
    approach_id: str | None = None,
    transcript_path: Path | None = None,
) -> float:
    """Return persisted elapsed seconds for a stage.

    Session maps are the only source of truth for consumed stage time. Transcript
    timestamps span resumed attempts and can include downtime, so they are never
    used for budget decisions.
    """
    session_map = load_session_map(
        session_data_dir, stage=stage,
        loop_index=loop_index, approach_id=approach_id,
    )
    _ = transcript_path
    return float(session_map.get("elapsed_seconds") or 0) if session_map else 0.0


def write_session_map(
    session_data_dir: Path,
    *,
    session_id: str,
    stage: str,
    loop_index: int | None = None,
    approach_id: str | None = None,
    time_budget_seconds: int | float | None = None,
    status: str = "running",
    elapsed_seconds: float = 0.0,
) -> None:
    """Write or update a session map, preserving elapsed budget state."""
    path = session_map_path(session_data_dir, stage=stage,
                            loop_index=loop_index, approach_id=approach_id)
    path.parent.mkdir(parents=True, exist_ok=True)

    existing: dict | None = None
    if path.exists():
        try:
            existing = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            existing = None

    prior_elapsed = float(existing.get("elapsed_seconds") or 0) if existing else 0.0
    elapsed = max(prior_elapsed, float(elapsed_seconds))

    effective_budget = time_budget_seconds
    if existing and existing.get("resume_extra_seconds"):
        existing_budget = existing.get("time_budget_seconds")
        if existing_budget is not None:
            if effective_budget is None:
                effective_budget = existing_budget
            else:
                effective_budget = max(float(effective_budget), float(existing_budget))

    payload: dict = {
        "session_id": session_id,
        "stage": stage,
        "loop_index": loop_index,
        "approach_id": approach_id,
        "time_budget_seconds": effective_budget,
        "elapsed_seconds": max(0.0, elapsed),
        "status": status,
    }
    if existing and existing.get("resume_extra_seconds"):
        payload["resume_extra_seconds"] = existing.get("resume_extra_seconds")
        payload["resume_extra_granted_at"] = existing.get("resume_extra_granted_at")
        payload["resume_extra_reason"] = existing.get("resume_extra_reason")
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def load_session_map(
    session_data_dir: Path,
    *,
    stage: str,
    loop_index: int | None = None,
    approach_id: str | None = None,
) -> dict | None:
    """Load a session map file, returning None if it doesn't exist."""
    path = session_map_path(session_data_dir, stage=stage,
                            loop_index=loop_index, approach_id=approach_id)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def update_session_map_status(
    session_data_dir: Path,
    *,
    stage: str,
    loop_index: int | None = None,
    approach_id: str | None = None,
    status: str = "completed",
) -> None:
    """Update the status field of an existing session map file."""
    path = session_map_path(session_data_dir, stage=stage,
                            loop_index=loop_index, approach_id=approach_id)
    if not path.exists():
        return
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return
    payload["status"] = status
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def persist_stage_elapsed(
    session_data_dir: Path,
    *,
    stage: str,
    loop_index: int | None = None,
    approach_id: str | None = None,
    add_seconds: float = 0.0,
) -> None:
    """Accumulate measured elapsed seconds for a stage attempt."""
    add_session_map_elapsed(
        session_data_dir,
        stage=stage,
        loop_index=loop_index,
        approach_id=approach_id,
        add_seconds=add_seconds,
    )


def add_session_map_elapsed(
    session_data_dir: Path,
    *,
    stage: str,
    loop_index: int | None = None,
    approach_id: str | None = None,
    add_seconds: float,
) -> None:
    """Accumulate wall time consumed by a stage attempt (for resume budgets)."""
    if add_seconds <= 0:
        return
    path = session_map_path(session_data_dir, stage=stage,
                            loop_index=loop_index, approach_id=approach_id)
    if not path.exists():
        return
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return
    payload["elapsed_seconds"] = float(payload.get("elapsed_seconds") or 0) + add_seconds
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def session_remaining_seconds(session_map: dict | None) -> float | None:
    """Return remaining budget from a session map, or None if unlimited."""
    if not session_map:
        return None
    budget = session_map.get("time_budget_seconds")
    if budget is None:
        return None
    elapsed = float(session_map.get("elapsed_seconds") or 0)
    return max(0.0, float(budget) - elapsed)


def prepare_has_open_session(session_data_dir: Path) -> bool:
    """True if prepare session is still aborted/running."""
    payload = load_session_map(session_data_dir, stage="prepare")
    if not payload:
        return False
    return payload.get("status") in ("aborted", "running")


def loop_implement_has_aborted_sessions(
    session_data_dir: Path, loop_index: int,
) -> bool:
    """True if any implement session for this loop was aborted (needs resume)."""
    maps_dir = session_data_dir / "session_maps"
    if not maps_dir.is_dir():
        return False
    propose_name = f"loop_{loop_index}_propose_session_map.json"
    for path in maps_dir.glob(f"loop_{loop_index}_*_session_map.json"):
        if path.name == propose_name:
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        if payload.get("stage") != "implement":
            continue
        if payload.get("status") == "aborted":
            return True
    return False


def loop_implement_has_open_sessions(
    session_data_dir: Path, loop_index: int,
) -> bool:
    """True if any implement session for this loop is still aborted/running."""
    maps_dir = session_data_dir / "session_maps"
    if not maps_dir.is_dir():
        return False
    propose_name = f"loop_{loop_index}_propose_session_map.json"
    for path in maps_dir.glob(f"loop_{loop_index}_*_session_map.json"):
        if path.name == propose_name:
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        if payload.get("stage") != "implement":
            continue
        if payload.get("status") in ("aborted", "running"):
            return True
    return False


def latest_open_propose_loop(session_data_dir: Path) -> int | None:
    """Return highest loop index with an aborted/running propose session map."""
    maps_dir = session_data_dir / "session_maps"
    if not maps_dir.is_dir():
        return None
    latest: int | None = None
    for path in maps_dir.glob("loop_*_propose_session_map.json"):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        if payload.get("stage") != "propose":
            continue
        if payload.get("status") not in ("aborted", "running"):
            continue
        loop_index = payload.get("loop_index")
        if not isinstance(loop_index, int):
            continue
        latest = loop_index if latest is None else max(latest, loop_index)
    return latest


def max_loop_implement_elapsed(
    session_data_dir: Path,
    loop_index: int,
    approach_ids: list[str],
    *,
    transcript_dir: Path | None = None,
) -> float:
    """Max elapsed across parallel implement approaches for one loop."""
    elapsed = 0.0
    for aid in approach_ids:
        transcript = None
        if transcript_dir is not None:
            transcript = transcript_dir / f"{aid}.jsonl"
        elapsed = max(
            elapsed,
            resolve_stage_elapsed(
                session_data_dir, stage="implement",
                loop_index=loop_index, approach_id=aid,
                transcript_path=transcript,
            ),
        )
    return elapsed
