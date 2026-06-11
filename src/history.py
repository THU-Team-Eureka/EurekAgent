"""Manifest I/O and history accumulation."""

from __future__ import annotations

import json
import logging
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_APPROACH_ID_RE = re.compile(r"^round_\d+_(?!approach_\d+$)[a-z][a-z0-9_-]*$")

log = logging.getLogger(__name__)


def read_manifest(path: Path) -> dict[str, Any] | None:
    """Read a manifest file tolerant to both single-JSON and JSON-Lines forms.

    The `current_round_approaches.jsonl` filename implies JSON Lines, but the
    original schema was a single JSON object with an `approaches` array. Agents
    reasonably follow the extension and write one approach per line, so we
    accept both shapes and normalise any legacy `approach_id` → `id` key drift.
    """
    if not path.exists():
        return None
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return None

    # Preferred: a single JSON object containing {"approaches": [...]}.
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        payload = None
    if isinstance(payload, dict):
        return _normalize_manifest(payload)

    # Fallback: JSON Lines — one approach object per line.
    approaches: list[dict[str, Any]] = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            return None
        if isinstance(obj, dict):
            approaches.append(obj)
    if not approaches:
        return None
    return _normalize_manifest({"approaches": approaches})


def _normalize_manifest(payload: dict[str, Any]) -> dict[str, Any]:
    """Coerce alternate keys so downstream code can rely on `id`."""
    approaches = payload.get("approaches")
    if isinstance(approaches, list):
        for a in approaches:
            if isinstance(a, dict) and not a.get("id") and a.get("approach_id"):
                a["id"] = a["approach_id"]
    return payload


def current_manifest_path(workspace: Path) -> Path:
    return workspace / "round_state" / "current_round_approaches.jsonl"


def loop_manifest_path(workspace: Path, loop_index: int) -> Path:
    """Legacy proposal snapshot path kept for reading old runs."""
    return workspace / "round_state" / f"loop_{loop_index}_approaches.jsonl"


def round_manifest_path(workspace: Path, loop_index: int) -> Path:
    """Proposal snapshot path for a round's initial hypotheses."""
    return workspace / "round_state" / f"round_{loop_index}_approaches.jsonl"


def ranked_best_solutions_path(workspace: Path) -> Path:
    return workspace / "round_state" / "ranked_past_best_solutions.jsonl"


def legacy_ranked_history_path(workspace: Path) -> Path:
    return workspace / "round_state" / "ranked_past_approaches.jsonl"


def resolve_ranked_history_path(workspace: Path, *, for_write: bool = False) -> Path:
    """Return the ranked best-solutions path, with legacy read fallback."""
    new_path = ranked_best_solutions_path(workspace)
    if for_write or new_path.exists():
        return new_path
    legacy_path = legacy_ranked_history_path(workspace)
    return legacy_path if legacy_path.exists() else new_path


def write_manifest(path: Path, payload: dict[str, Any]) -> None:
    """Write a normalized manifest as a single JSON object."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_normalize_manifest(payload), indent=2) + "\n",
        encoding="utf-8",
    )


def snapshot_manifest_for_loop(
    workspace: Path,
    loop_index: int,
    source_path: Path | None = None,
    *,
    sync_current: bool = False,
) -> Path | None:
    """Persist the current manifest under a round-specific filename."""
    source = source_path or current_manifest_path(workspace)
    payload = read_manifest(source)
    if not payload or not _manifest_matches_loop(payload, loop_index):
        return None
    target = round_manifest_path(workspace, loop_index)
    write_manifest(target, payload)
    if sync_current:
        write_manifest(current_manifest_path(workspace), payload)
    return target


def resolve_loop_manifest(
    workspace: Path,
    loop_index: int,
    *,
    session_data_dir: Path | None = None,
    sync_current: bool = False,
) -> Path | None:
    """Return a readable manifest for a round, recovering legacy runs if needed."""
    for manifest_path in (
        round_manifest_path(workspace, loop_index),
        loop_manifest_path(workspace, loop_index),
    ):
        payload = read_manifest(manifest_path)
        if payload and _manifest_matches_loop(payload, loop_index):
            if sync_current:
                write_manifest(current_manifest_path(workspace), payload)
            return manifest_path

    current_path = current_manifest_path(workspace)
    payload = read_manifest(current_path)
    if payload and _manifest_matches_loop(payload, loop_index):
        return snapshot_manifest_for_loop(
            workspace, loop_index, current_path, sync_current=sync_current,
        )

    return recover_loop_manifest(
        workspace, loop_index, session_data_dir=session_data_dir,
        sync_current=sync_current,
    )


def recover_loop_manifest(
    workspace: Path,
    loop_index: int,
    *,
    session_data_dir: Path | None = None,
    sync_current: bool = False,
) -> Path | None:
    """Rebuild a minimal round manifest from approach dirs and session maps."""
    ids = _recover_loop_approach_ids(workspace, loop_index, session_data_dir)
    if not ids:
        return None
    payload = {
        "approaches": [
            {"id": aid, "name": aid, "description": ""} for aid in ids
        ],
    }
    target = round_manifest_path(workspace, loop_index)
    write_manifest(target, payload)
    if sync_current:
        write_manifest(current_manifest_path(workspace), payload)
    log.warning("Recovered missing manifest for round %d at %s", loop_index, target)
    return target


def _manifest_matches_loop(payload: dict[str, Any], loop_index: int) -> bool:
    approaches = payload.get("approaches")
    if not isinstance(approaches, list) or not approaches:
        return False
    for approach in approaches:
        if not isinstance(approach, dict):
            return False
        aid = str(approach.get("id", "")).strip()
        if not _approach_id_matches_loop(aid, loop_index):
            return False
    return True


def _recover_loop_approach_ids(
    workspace: Path,
    loop_index: int,
    session_data_dir: Path | None,
) -> list[str]:
    seen: set[str] = set()
    ids: list[str] = []

    def add(aid: str) -> None:
        aid = aid.strip()
        if aid not in seen and _approach_id_matches_loop(aid, loop_index):
            seen.add(aid)
            ids.append(aid)

    if session_data_dir is not None:
        maps_dir = session_data_dir / "session_maps"
        if maps_dir.is_dir():
            for path in sorted(maps_dir.glob(f"loop_{loop_index}_*_session_map.json")):
                try:
                    payload = json.loads(path.read_text(encoding="utf-8"))
                except (json.JSONDecodeError, OSError):
                    continue
                if payload.get("stage") == "implement":
                    add(str(payload.get("approach_id") or ""))

    details_dir = workspace / "approach_details"
    if details_dir.is_dir():
        for path in sorted(details_dir.glob(f"round_{loop_index}_*")):
            if path.is_dir():
                add(path.name)

    return ids


def _approach_id_matches_loop(aid: str, loop_index: int) -> bool:
    return bool(_APPROACH_ID_RE.match(aid)) and aid.startswith(f"round_{loop_index}_")


def validate_manifest(
    path: Path, *, max_count: int
) -> tuple[dict[str, Any] | None, str | None]:
    """Validate a round manifest. Returns (manifest, None) or (None, error).

    Approaches may be fewer than max_count (agent decides how many to generate),
    but must not exceed it.
    """
    payload = read_manifest(path)
    if payload is None:
        return None, f"Manifest not found or unreadable: {path}"

    approaches = payload.get("approaches")
    if not isinstance(approaches, list):
        return None, "Manifest must contain an 'approaches' list."
    if len(approaches) > max_count:
        return None, f"Manifest has {len(approaches)} approaches, exceeds max of {max_count}."

    for i, a in enumerate(approaches, 1):
        if not isinstance(a, dict) or not str(a.get("id", "")).strip():
            return None, f"Approach #{i} is missing 'id'."
        aid = str(a.get("id", "")).strip()
        if not _APPROACH_ID_RE.match(aid):
            return None, (
                f"Approach #{i} id='{aid}' does not match required format "
                f"'round_{{N}}_{{short_name}}' (e.g. round_1_cmaes_optim)."
            )

    return payload, None

def collect_round_entries(
    manifest_path: Path, *, workspace_dir: Path, loop_index: int
) -> list[dict[str, Any]]:
    """Collect ranked best-solution entries for a completed round."""
    payload = read_manifest(manifest_path)
    if not payload:
        return []

    entries: list[dict[str, Any]] = []
    for approach in payload.get("approaches", []):
        if not isinstance(approach, dict):
            continue
        aid = str(approach.get("id", "")).strip()
        if not aid:
            continue

        result_path = workspace_dir / "approach_details" / aid / "best_result.jsonl"
        completed_at = ""
        status = "failed"
        score: float | None = None
        trusted = False
        result_data: dict[str, Any] | None = None
        if result_path.exists() and result_path.stat().st_size > 0:
            mtime = result_path.stat().st_mtime
            completed_at = datetime.fromtimestamp(mtime, tz=timezone.utc).isoformat()
            result_data = _read_best_result(result_path)
            if result_data:
                if result_data.get("valid") is True:
                    status = "completed"
                score = _coerce_float(result_data.get("score"))
                trusted = result_data.get("controller_status") == "secure_graded"

        entries.append({
            "approach_id": aid,
            "loop_index": loop_index,
            "initial_hypothesis_name": str(approach.get("name", "")).strip(),
            "initial_hypothesis_description": str(approach.get("description", "")).strip(),
            "description": _result_description(result_data),
            "status": status,
            "completed_at": completed_at,
            "score": score,
            "trusted": trusted,
            "result_path": f"approach_details/{aid}/best_result.jsonl",
            "approach_dir": f"approach_details/{aid}",
            "submission_path": _result_string(result_data, "submission_path"),
            "evaluated_at": _result_string(result_data, "evaluated_at"),
            "code_git_hash": _result_string(result_data, "code_git_hash") or None,
            "code_git_dirty": (
                result_data.get("code_git_dirty")
                if isinstance(result_data, dict) and isinstance(result_data.get("code_git_dirty"), bool)
                else None
            ),
            "code_commit_message": _result_string(result_data, "code_commit_message"),
        })

    return entries


def _result_string(result: dict[str, Any] | None, key: str) -> str:
    if not isinstance(result, dict):
        return ""
    value = result.get(key)
    return value.strip() if isinstance(value, str) else ""


def _result_description(result: dict[str, Any] | None) -> str:
    """Return the scored solution description, with legacy result fallback."""
    desc = _result_string(result, "description")
    if desc:
        return desc
    if isinstance(result, dict):
        solution = result.get("solution")
        if isinstance(solution, dict):
            value = solution.get("description")
            if isinstance(value, str):
                return value.strip()
    return ""


def _coerce_float(value: object) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value.strip())
        except ValueError:
            return None
    return None


def _read_best_result(path: Path) -> dict[str, Any] | None:
    """Read a best_result.jsonl file as a single JSON dict."""
    if not path.exists():
        return None
    try:
        text = path.read_text(encoding="utf-8").strip()
    except OSError:
        return None
    if not text:
        return None
    try:
        data = json.loads(text)
        return data if isinstance(data, dict) else None
    except json.JSONDecodeError:
        return None


def load_ranked_history(path: Path) -> list[dict[str, Any]]:
    """Read ranked history file. Returns entries in file order."""
    if path.name == "ranked_past_approaches.jsonl":
        new_path = path.with_name("ranked_past_best_solutions.jsonl")
        if new_path.exists():
            path = new_path
    payload = read_manifest(path)
    if not payload:
        return []
    entries = payload.get("entries")
    if isinstance(entries, list):
        return [e for e in entries if isinstance(e, dict)]
    return []


def append_to_history(
    history_path: Path, *, new_entries: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Append entries to history, deduplicating by approach_id.

    If an approach_id already exists, the entry is replaced when the new
    entry has a better status (completed > failed) or a strictly higher
    score.  This handles the resume case where an approach initially
    recorded as ``failed`` later produces a valid result.
    """
    existing = load_ranked_history(history_path)
    existing_by_id: dict[str, dict[str, Any]] = {
        str(e.get("approach_id", "")): e for e in existing
    }
    for entry in new_entries:
        aid = str(entry.get("approach_id", ""))
        if not aid:
            continue
        old = existing_by_id.get(aid)
        if old is None:
            existing_by_id[aid] = entry
        elif _entry_is_better(entry, old):
            existing_by_id[aid] = entry
    merged = list(existing_by_id.values())
    write_json(history_path, {"entries": merged})
    return merged


def _entry_is_better(new: dict[str, Any], old: dict[str, Any]) -> bool:
    """Return True if *new* should replace *old* for the same approach_id.

    Per-approach best_result.jsonl is already chosen by the grader's
    problem-specific is_better function, so a completed refreshed entry is
    authoritative regardless of score direction.
    """
    new_ok = new.get("status") == "completed"
    old_ok = old.get("status") == "completed"
    if new_ok:
        return True
    return not old_ok


def write_json(path: Path, payload: dict[str, Any]) -> None:
    """Write a dict as pretty-printed JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
