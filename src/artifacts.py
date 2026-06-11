"""Canonical stage artifact validation.

All stage routing, resume preflight, and terminal reporting should derive
artifact readiness from this module instead of ad-hoc file existence checks.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .history import current_manifest_path, validate_manifest


@dataclass
class ArtifactValidationResult:
    ok: bool
    stage: str
    loop_index: int | None = None
    reason_code: str = ""
    message: str = ""
    missing_artifacts: list[str] = field(default_factory=list)
    invalid_artifacts: list[str] = field(default_factory=list)
    completed_artifacts: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def detail(self) -> str:
        if self.message:
            return self.message
        paths = self.missing_artifacts or self.invalid_artifacts
        if paths:
            return ", ".join(paths[:3]) + (f", ... and {len(paths) - 3} more" if len(paths) > 3 else "")
        return ""


def validate_prepare_artifacts(workspace: Path) -> ArtifactValidationResult:
    complete = workspace / "prepare" / "complete.json"
    if not _nonempty_file(complete):
        return ArtifactValidationResult(
            ok=False,
            stage="prepare",
            reason_code="missing_prepare_complete",
            message="Missing or empty prepare/complete.json.",
            missing_artifacts=[str(complete)],
        )
    try:
        payload = json.loads(complete.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return ArtifactValidationResult(
            ok=False,
            stage="prepare",
            reason_code="invalid_prepare_complete",
            message="prepare/complete.json is not valid JSON.",
            invalid_artifacts=[str(complete)],
        )
    if not isinstance(payload, dict):
        return ArtifactValidationResult(
            ok=False,
            stage="prepare",
            reason_code="invalid_prepare_complete",
            message="prepare/complete.json must contain a JSON object.",
            invalid_artifacts=[str(complete)],
        )
    return ArtifactValidationResult(
        ok=True,
        stage="prepare",
        completed_artifacts=[str(complete)],
        metadata={"complete": payload},
    )


def validate_propose_artifacts(
    workspace: Path,
    loop_index: int,
    max_num_approaches: int,
    manifest_path: Path | None = None,
) -> ArtifactValidationResult:
    manifest_path = manifest_path or current_manifest_path(workspace)
    manifest, err = validate_manifest(manifest_path, max_count=max_num_approaches)
    if err or manifest is None:
        missing = [str(manifest_path)] if not _nonempty_file(manifest_path) else []
        invalid = [] if missing else [str(manifest_path)]
        return ArtifactValidationResult(
            ok=False,
            stage="propose",
            loop_index=loop_index,
            reason_code="missing_manifest" if missing else "invalid_manifest",
            message=err or "Approaches manifest is invalid.",
            missing_artifacts=missing,
            invalid_artifacts=invalid,
        )

    approaches = [a for a in manifest.get("approaches", []) if isinstance(a, dict)]
    if not approaches:
        return ArtifactValidationResult(
            ok=False,
            stage="propose",
            loop_index=loop_index,
            reason_code="empty_manifest",
            message="Approaches manifest contains no approaches.",
            invalid_artifacts=[str(manifest_path)],
            metadata={"manifest": manifest, "approaches": []},
        )

    missing: list[str] = []
    invalid: list[str] = []
    completed = [str(manifest_path)]
    approach_ids: list[str] = []
    for approach in approaches:
        aid = str(approach.get("id", "")).strip()
        approach_ids.append(aid)
        if not aid.startswith(f"round_{loop_index}_"):
            return ArtifactValidationResult(
                ok=False,
                stage="propose",
                loop_index=loop_index,
                reason_code=f"manifest_round_mismatch:{aid}",
                message=f"Manifest approach {aid} does not belong to round {loop_index}.",
                invalid_artifacts=[str(manifest_path)],
                metadata={"manifest": manifest, "approaches": approaches, "approach_ids": approach_ids},
            )
        approach_md = workspace / "approach_details" / aid / "approach.md"
        if not approach_md.exists():
            missing.append(str(approach_md))
            continue
        try:
            text = approach_md.read_text(encoding="utf-8")
        except OSError:
            invalid.append(str(approach_md))
            continue
        if not text.strip():
            invalid.append(str(approach_md))
            continue
        completed.append(str(approach_md))

    if missing or invalid:
        first_path = Path((missing or invalid)[0])
        aid = first_path.parent.name
        code = f"missing_approach_md:{aid}" if missing else f"invalid_approach_md:{aid}"
        missing_ids = [Path(p).parent.name for p in missing]
        invalid_ids = [Path(p).parent.name for p in invalid]
        parts = []
        if missing_ids:
            parts.append("missing approach.md for " + ", ".join(missing_ids))
        if invalid_ids:
            parts.append("empty or unreadable approach.md for " + ", ".join(invalid_ids))
        return ArtifactValidationResult(
            ok=False,
            stage="propose",
            loop_index=loop_index,
            reason_code=code,
            message="; ".join(parts),
            missing_artifacts=missing,
            invalid_artifacts=invalid,
            completed_artifacts=completed,
            metadata={"manifest": manifest, "approaches": approaches, "approach_ids": approach_ids},
        )

    return ArtifactValidationResult(
        ok=True,
        stage="propose",
        loop_index=loop_index,
        completed_artifacts=completed,
        metadata={"manifest": manifest, "approaches": approaches, "approach_ids": approach_ids},
    )


def validate_implement_artifacts(
    workspace: Path,
    loop_index: int,
    manifest_path: Path,
    *,
    max_num_approaches: int | None = None,
    require_all: bool = False,
) -> ArtifactValidationResult:
    max_count = max_num_approaches if max_num_approaches is not None else 10_000
    manifest, err = validate_manifest(manifest_path, max_count=max_count)
    if err or manifest is None:
        missing = [str(manifest_path)] if not _nonempty_file(manifest_path) else []
        invalid = [] if missing else [str(manifest_path)]
        return ArtifactValidationResult(
            ok=False,
            stage="implement",
            loop_index=loop_index,
            reason_code="missing_manifest" if missing else "invalid_manifest",
            message=err or "Approaches manifest is invalid.",
            missing_artifacts=missing,
            invalid_artifacts=invalid,
        )

    approaches = [a for a in manifest.get("approaches", []) if isinstance(a, dict)]
    missing: list[str] = []
    invalid: list[str] = []
    completed: list[str] = []
    succeeded_ids: list[str] = []
    failed_ids: list[str] = []
    scores: dict[str, float | None] = {}
    for approach in approaches:
        aid = str(approach.get("id", "")).strip()
        if not aid.startswith(f"round_{loop_index}_"):
            return ArtifactValidationResult(
                ok=False,
                stage="implement",
                loop_index=loop_index,
                reason_code=f"manifest_round_mismatch:{aid}",
                message=f"Manifest approach {aid} does not belong to round {loop_index}.",
                invalid_artifacts=[str(manifest_path)],
                metadata={"manifest": manifest, "approaches": approaches},
            )
        result_path = workspace / "approach_details" / aid / "best_result.jsonl"
        payload = read_best_result(result_path)
        if payload and payload.get("valid") is True:
            completed.append(str(result_path))
            succeeded_ids.append(aid)
            scores[aid] = _coerce_score(payload.get("score"))
        else:
            failed_ids.append(aid)
            scores[aid] = None
            if not _nonempty_file(result_path):
                missing.append(str(result_path))
            else:
                invalid.append(str(result_path))

    ok = bool(succeeded_ids) and (not require_all or len(succeeded_ids) == len(approaches))
    if ok:
        code = "all_succeeded" if len(succeeded_ids) == len(approaches) else "partial_succeeded"
        message = f"{len(succeeded_ids)} / {len(approaches)} approaches produced valid best_result.jsonl."
    else:
        code = "missing_best_result" if missing else "invalid_best_result"
        message = (
            f"Missing or invalid best_result.jsonl for {len(failed_ids)} / "
            f"{len(approaches)} approaches."
        )

    return ArtifactValidationResult(
        ok=ok,
        stage="implement",
        loop_index=loop_index,
        reason_code=code,
        message=message,
        missing_artifacts=missing,
        invalid_artifacts=invalid,
        completed_artifacts=completed,
        metadata={
            "manifest": manifest,
            "approaches": approaches,
            "succeeded_ids": succeeded_ids,
            "failed_ids": failed_ids,
            "scores": scores,
        },
    )


def read_best_result(path: Path) -> dict[str, Any] | None:
    if not _nonempty_file(path):
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def best_result_is_valid(path: Path) -> bool:
    payload = read_best_result(path)
    return bool(payload and payload.get("valid") is True)


def read_best_score(approach_dir: Path) -> float | None:
    payload = read_best_result(approach_dir / "best_result.jsonl")
    if payload and payload.get("valid") is True:
        return _coerce_score(payload.get("score"))
    return None


def artifact_reason_text(reason: str) -> str:
    if reason.startswith("missing_approach_md"):
        return "required propose approach details missing"
    if reason.startswith("invalid_approach_md"):
        return "required propose approach details invalid"
    if reason.startswith("manifest_round_mismatch"):
        return "manifest belongs to the wrong round"
    return {
        "missing_prepare_complete": "prepare completion artifact missing",
        "invalid_prepare_complete": "prepare completion artifact invalid",
        "missing_manifest": "required approaches manifest missing",
        "invalid_manifest": "required approaches manifest invalid",
        "empty_manifest": "required approaches manifest empty",
        "missing_best_result": "no valid best_result produced",
        "invalid_best_result": "no valid best_result produced",
    }.get(reason, "")


def _nonempty_file(path: Path) -> bool:
    try:
        return path.is_file() and path.stat().st_size > 0
    except OSError:
        return False


def _coerce_score(val: Any) -> float | None:
    try:
        return float(val)
    except (TypeError, ValueError):
        return None
