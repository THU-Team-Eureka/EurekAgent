"""Filesystem-derived stage status for resume and monitor views."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from .artifacts import best_result_is_valid, validate_prepare_artifacts, validate_propose_artifacts
from .history import current_manifest_path, round_manifest_path
from .session_map import loop_implement_has_aborted_sessions
from .token_tracker import find_transcripts_dir

_APPROACH_ROUND_RE = re.compile(r"^round_(\d+)_")


def scan_stage_results(workspace: Path) -> dict[int, dict[str, dict[str, Any]]]:
    """Infer prepare/propose/implement status from run artifacts."""
    results: dict[int, dict[str, dict[str, Any]]] = {}
    details_dir = workspace / "approach_details"
    pipeline_state = _read_pipeline_state(workspace)
    current_loop = int(pipeline_state.get("current_loop_index") or 0)
    current_stage = str(pipeline_state.get("current_stage") or "")
    pipeline_status = str(pipeline_state.get("pipeline_status") or "running")
    session_data_dir = workspace.parent / "session_data"

    if not details_dir.is_dir():
        _scan_prepare(workspace, results, pipeline_status)
        _scan_aborted_propose_transcripts(
            workspace, results, current_loop=current_loop,
            pipeline_status=pipeline_status,
        )
        return results

    loop_approaches: dict[int, list[Path]] = {}
    for approach_dir in details_dir.iterdir():
        if not approach_dir.is_dir():
            continue
        match = _APPROACH_ROUND_RE.match(approach_dir.name)
        if match:
            loop_approaches.setdefault(int(match.group(1)), []).append(approach_dir)

    for loop_idx, approach_dirs in loop_approaches.items():
        propose_ready = _propose_ready(workspace, loop_idx)
        if propose_ready:
            results.setdefault(loop_idx, {})["propose"] = {"status": "ready"}
        elif (
            pipeline_status in ("abort", "interrupted", "error")
            and current_stage == "propose"
            and loop_idx == current_loop
        ):
            results.setdefault(loop_idx, {})["propose"] = {"status": "abort"}
            continue

        if (
            pipeline_status in ("abort", "interrupted", "error")
            and current_stage == "implement"
            and loop_idx == current_loop
        ):
            results.setdefault(loop_idx, {})["implement"] = {"status": "interrupted"}
            continue

        if loop_implement_has_aborted_sessions(session_data_dir, loop_idx):
            results.setdefault(loop_idx, {})["implement"] = {"status": "interrupted"}
            continue

        if pipeline_status == "running" and loop_idx >= current_loop:
            if current_stage == "implement" and loop_idx == current_loop:
                results.setdefault(loop_idx, {})["implement"] = {"status": "running"}
            continue

        succeeded = sum(
            1 for approach_dir in approach_dirs
            if best_result_is_valid(approach_dir / "best_result.jsonl")
        )
        if succeeded == len(approach_dirs):
            status = "all_succeeded"
        elif succeeded > 0:
            status = "partial_succeeded"
        elif pipeline_status in ("abort", "interrupted", "error"):
            status = "interrupted"
        else:
            status = "all_failed"
        results.setdefault(loop_idx, {})["implement"] = {"status": status}

    _scan_aborted_propose_transcripts(
        workspace, results, current_loop=current_loop,
        pipeline_status=pipeline_status,
    )
    _scan_prepare(workspace, results, pipeline_status)
    return results


def _scan_prepare(
    workspace: Path,
    results: dict[int, dict[str, dict[str, Any]]],
    pipeline_status: str,
) -> None:
    complete = workspace / "prepare" / "complete.json"
    result_file = workspace / "prepare_result.json"
    if validate_prepare_artifacts(workspace).ok or result_file.exists():
        payload = _read_json_or_jsonl(result_file)
        status = payload.get("status", "ready") if payload else "ready"
        results.setdefault(0, {})["prepare"] = {"status": status}
    elif (find_transcripts_dir(workspace) / "prepare").exists():
        status = "running" if pipeline_status == "running" else "abort"
        results.setdefault(0, {})["prepare"] = {"status": status}


def _scan_aborted_propose_transcripts(
    workspace: Path,
    results: dict[int, dict[str, dict[str, Any]]],
    *,
    current_loop: int,
    pipeline_status: str,
) -> None:
    transcripts_dir = find_transcripts_dir(workspace)
    if not transcripts_dir.is_dir():
        return
    for loop_dir in transcripts_dir.iterdir():
        if not loop_dir.is_dir():
            continue
        match = re.match(r"^loop_(\d+)$", loop_dir.name)
        if not match:
            continue
        loop_idx = int(match.group(1))
        if loop_idx in results:
            continue
        if not (loop_dir / "propose.jsonl").exists():
            continue
        if loop_idx < current_loop or pipeline_status != "running":
            results.setdefault(loop_idx, {})["propose"] = {"status": "abort"}


def _read_pipeline_state(workspace: Path) -> dict[str, Any]:
    path = workspace / ".pipeline_state.json"
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _read_json_or_jsonl(path: Path) -> dict[str, Any] | None:
    if not path.exists() or path.stat().st_size == 0:
        return None
    try:
        text = path.read_text(encoding="utf-8").strip()
    except OSError:
        return None
    if not text:
        return None
    line = text.splitlines()[-1]
    try:
        payload = json.loads(line)
    except json.JSONDecodeError:
        return None
    return payload if isinstance(payload, dict) else None


def _propose_ready(workspace: Path, loop_idx: int) -> bool:
    for manifest_path in (round_manifest_path(workspace, loop_idx), current_manifest_path(workspace)):
        if validate_propose_artifacts(
            workspace, loop_idx, max_num_approaches=10_000,
            manifest_path=manifest_path,
        ).ok:
            return True
    return False
