"""Host-side secure grader service.

Receives submission requests from agent sessions, grades them using a private
evaluator, writes result files to the workspace, and returns scores.

Result file writing lives here (server-side) so that agents cannot tamper
with scores by modifying their submit script.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import importlib.util
import json
import logging
import os
import re
import subprocess
import sys
import tempfile
import threading
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import uvicorn
from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel, Field

log = logging.getLogger(__name__)


class GradeRequest(BaseModel):
    submission_path: str
    approach_id: str = ""
    gpu_lock_token: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)


class CompareRequest(BaseModel):
    new_score: float
    old_score: float


def _load_grader(
    grader_file: Path,
) -> tuple[Callable[[str, dict[str, Any]], dict[str, Any]], Callable[[float, float], bool]]:
    spec = importlib.util.spec_from_file_location("eureka_private_grader", grader_file)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load private grader from {grader_file}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    grader = getattr(module, "grade_submission", None)
    if not callable(grader):
        raise RuntimeError(
            f"{grader_file} must define callable `grade_submission(submission_path, context)`."
        )
    is_better_fn = getattr(module, "is_better", None)
    if not callable(is_better_fn):
        raise RuntimeError(
            f"{grader_file} must define callable `is_better(new_score, old_score) -> bool`."
        )
    return grader, is_better_fn


# -- GPU lock resolution + subprocess evaluation --
#
# The grader server itself starts with CUDA_VISIBLE_DEVICES="" (default-deny).
# For each /grade request we discover which physical GPUs the calling agent
# currently holds (by reading workspace_root/.gpu_locks/) and run the
# evaluator in a fresh subprocess with CUDA_VISIBLE_DEVICES set to exactly
# those GPUs. The subprocess's CUDA context is created from scratch, so the
# visibility constraint always takes effect (unlike modifying os.environ in
# the long-lived server process, which has no effect once torch is loaded).
#
# This design keeps the grader stateless w.r.t. GPU coordination — the lock
# files are the single source of truth, and the agent never has to tell the
# grader which GPUs to use.

# Evaluator result transport
# --------------------------
# The grader subprocess returns its result by writing JSON to the file at
# ``$EUREKA_RESULT_PATH`` (path injected by the parent). This decouples
# result transport from stdout/stderr, so the evaluator is free to print
# progress lines, logging messages, tqdm bars, or any other diagnostics
# without breaking parsing.
#
# Backward compatibility: the runner also writes the JSON to stdout, and
# ``_parse_grader_stdout`` can recover a JSON object from a noisy stdout
# stream by scanning backwards for a decodable suffix. So an evaluator that
# only relies on the (older) stdout contract still works.

_EVAL_RUNNER_SCRIPT = r"""
import json, os, sys, importlib.util
spec = importlib.util.spec_from_file_location("_eureka_private_grader", os.environ["EUREKA_GRADER_FILE"])
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
ctx = json.loads(os.environ["EUREKA_CTX"])
result = mod.grade_submission(os.environ["EUREKA_SUBMISSION_PATH"], ctx)
payload = json.dumps(result)
result_path = os.environ.get("EUREKA_RESULT_PATH", "")
if result_path:
    # Primary transport: write the result to a file the parent owns.
    # Use a tmp-then-rename pattern so the parent never reads a partial write.
    tmp_path = result_path + ".partial"
    with open(tmp_path, "w", encoding="utf-8") as fh:
        fh.write(payload)
    os.replace(tmp_path, result_path)
# Secondary transport (back-compat): also write to stdout.
sys.stdout.write(payload)
"""


def _parse_grader_stdout(stdout: str) -> dict[str, Any]:
    """Extract the last JSON object from ``stdout``.

    The fast path treats the whole stdout as a single JSON document. The
    slow path scans backwards for ``{`` characters and tries to decode the
    suffix starting at each one, so leading progress lines such as
    ``[Running]`` or tqdm output do not break parsing.

    Raises:
        ValueError: no JSON object could be decoded anywhere in ``stdout``.
    """
    text = stdout.strip()
    if not text:
        raise ValueError("grader subprocess stdout is empty")
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        data = None
    if isinstance(data, dict):
        return data
    # Slow path: scan from the last '{' backwards for a JSON-decodable suffix.
    idx = text.rfind("{")
    while idx != -1:
        try:
            data = json.loads(text[idx:])
        except json.JSONDecodeError:
            data = None
        if isinstance(data, dict):
            return data
        idx = text.rfind("{", 0, idx)
    raise ValueError("no JSON object found in grader subprocess stdout")


def _resolve_gpu_ids_from_locks(
    workspace_root: Path,
    approach_id: str,
    gpu_lock_token: str,
) -> list[int]:
    """Return physical GPU ids locked by ``approach_id`` and token.

    Reads ``workspace_root/.gpu_locks/gpu_*.lock`` and returns the ids whose
    owner fields match both ``approach_id`` and ``gpu_lock_token``. An empty
    list means the caller holds no authorized GPUs — in that case the
    evaluation subprocess runs with CUDA_VISIBLE_DEVICES="" (no GPU access),
    which is the right default for CPU-only tasks. Returns sorted ids for
    deterministic env values.
    """
    if not approach_id or not gpu_lock_token:
        return []
    lock_dir = workspace_root / ".gpu_locks"
    if not lock_dir.is_dir():
        return []
    held: list[int] = []
    for lock_path in lock_dir.glob("gpu_*.lock"):
        try:
            data = json.loads(lock_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if data.get("approach_id") != approach_id:
            continue
        if data.get("gpu_lock_token") != gpu_lock_token:
            continue
        try:
            gid = int(lock_path.stem.split("_", 1)[1])
        except (ValueError, IndexError):
            continue
        held.append(gid)
    return sorted(held)


def _load_gpu_lock_records(workspace_root: Path) -> list[dict[str, Any]]:
    """Return parsed GPU lock records from ``workspace_root/.gpu_locks``."""
    lock_dir = workspace_root / ".gpu_locks"
    if not lock_dir.is_dir():
        return []
    records: list[dict[str, Any]] = []
    for lock_path in sorted(lock_dir.glob("gpu_*.lock")):
        try:
            data = json.loads(lock_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(data, dict):
            continue
        try:
            gpu_id = int(lock_path.stem.split("_", 1)[1])
        except (ValueError, IndexError):
            continue
        record = dict(data)
        record["gpu_id"] = gpu_id
        records.append(record)
    return records


def _looks_like_cuda_visibility_failure(exc: Exception) -> bool:
    text = str(exc)
    return "no CUDA-capable device is detected" in text or "CUDA_VISIBLE_DEVICES" in text


def _gpu_authorization_diagnostic(
    workspace_root: Path,
    *,
    approach_id: str,
    gpu_lock_token: str,
) -> dict[str, Any] | None:
    """Return a structured explanation for an authorization mismatch.

    This is only used after a CUDA visibility failure, so it intentionally
    stays conservative: if there is no relevant lock state to inspect, it
    returns ``None`` instead of guessing.
    """
    records = _load_gpu_lock_records(workspace_root)
    if not records:
        return None

    matching_approach = [r for r in records if str(r.get("approach_id", "")) == approach_id]
    matching_token = (
        [r for r in records if str(r.get("gpu_lock_token", "")) == gpu_lock_token]
        if gpu_lock_token
        else []
    )
    if not gpu_lock_token:
        if matching_approach:
            message = (
                "The submission did not include a gpu_lock_token, but this approach already "
                "holds GPU lock(s). Submit from inside the same gpu_session(..., "
                f"approach_id={approach_id!r}) block and keep the with-block open until "
                "eureka_submit.py finishes."
            )
        else:
            message = (
                "The submission did not include a gpu_lock_token. If this submission needs "
                "CUDA, call eureka_submit.py from inside gpu_session(..., "
                f"approach_id={approach_id!r})."
            )
        return {
            "error": "gpu_authorization_failed",
            "reason": "missing_gpu_lock_token",
            "message": message,
            "approach_id": approach_id,
            "gpu_lock_token_present": False,
            "matching_gpu_ids": [int(r["gpu_id"]) for r in matching_approach],
        }

    if matching_approach:
        if not matching_token:
            return {
                "error": "gpu_authorization_failed",
                "reason": "gpu_lock_token_mismatch",
                "message": (
                    "A GPU lock exists for this approach_id, but the submitted gpu_lock_token "
                    "does not match it. Submit while the same gpu_session(..., "
                    f"approach_id={approach_id!r}) block is still open; do not leave the "
                    "context before invoking eureka_submit.py."
                ),
                "approach_id": approach_id,
                "gpu_lock_token_present": True,
                "matching_gpu_ids": [int(r["gpu_id"]) for r in matching_approach],
            }
        return None

    if matching_token:
        locked_owner = str(matching_token[0].get("approach_id", ""))
        return {
            "error": "gpu_authorization_failed",
            "reason": "approach_id_mismatch",
            "message": (
                f"The submitted gpu_lock_token belongs to approach_id={locked_owner!r}, but "
                f"the submission used approach_id={approach_id!r}. Use the same approach_id "
                "in gpu_session(...) and eureka_submit.py."
            ),
            "approach_id": approach_id,
            "gpu_lock_token_present": True,
            "matching_gpu_ids": [int(r["gpu_id"]) for r in matching_token],
        }

    return {
        "error": "gpu_authorization_failed",
        "reason": "stale_or_unmatched_gpu_lock_token",
        "message": (
            "The submitted gpu_lock_token did not match any current GPU lock. The token is "
            "likely stale, or eureka_submit.py ran after leaving gpu_session(...)."
        ),
        "approach_id": approach_id,
        "gpu_lock_token_present": True,
        "matching_gpu_ids": [],
    }


_gpu_condition = threading.Condition()
_busy_gpu_ids: set[int] = set()


@contextmanager
def _reserve_grader_gpus(gpu_ids: list[int], timeout: float | None = None):
    """Reserve a physical GPU set for one grader subprocess."""
    requested = set(gpu_ids)
    if not requested:
        yield
        return

    deadline = None if timeout is None else time.monotonic() + timeout
    with _gpu_condition:
        while requested & _busy_gpu_ids:
            if deadline is None:
                _gpu_condition.wait()
                continue
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("Timed out waiting for grader GPU reservation.")
            _gpu_condition.wait(timeout=remaining)
        _busy_gpu_ids.update(requested)

    try:
        yield
    finally:
        with _gpu_condition:
            _busy_gpu_ids.difference_update(requested)
            _gpu_condition.notify_all()


def _run_grader_subprocess(
    grader_file: Path,
    submission_path: str,
    context: dict[str, Any],
    gpu_ids: list[int],
    timeout: float,
) -> dict[str, Any]:
    """Run the private grader in a subprocess with the given GPU visibility.

    Result transport is file-based (see ``_EVAL_RUNNER_SCRIPT``): the parent
    creates a temp file, injects its path via ``$EUREKA_RESULT_PATH``, and
    the child writes the result JSON there. We fall back to stdout parsing
    so older evaluators that only print JSON keep working.
    """
    fd, result_path = tempfile.mkstemp(prefix="eureka_grade_", suffix=".json")
    os.close(fd)
    try:
        env = dict(os.environ)
        env["CUDA_VISIBLE_DEVICES"] = ",".join(str(g) for g in gpu_ids) if gpu_ids else ""
        env["EUREKA_GRADER_FILE"] = str(grader_file)
        env["EUREKA_SUBMISSION_PATH"] = submission_path
        env["EUREKA_CTX"] = json.dumps(context)
        env["EUREKA_RESULT_PATH"] = result_path
        proc = subprocess.run(
            [sys.executable, "-c", _EVAL_RUNNER_SCRIPT],
            env=env, capture_output=True, text=True, timeout=timeout,
        )
        if proc.returncode != 0:
            # Surface a bounded tail of stderr to aid debugging without dumping huge logs.
            tail = (proc.stderr or "").strip()
            if len(tail) > 1500:
                tail = "...(truncated)...\n" + tail[-1500:]
            raise RuntimeError(
                f"Grader subprocess exited with code {proc.returncode}. stderr:\n{tail}"
            )

        # Primary: read the result file the child wrote.
        try:
            with open(result_path, "r", encoding="utf-8") as fh:
                text = fh.read()
            if text.strip():
                data = json.loads(text)
                if isinstance(data, dict):
                    return data
        except (OSError, json.JSONDecodeError):
            pass

        # Fallback: parse stdout (back-compat with older evaluators that
        # only printed JSON and didn't honor EUREKA_RESULT_PATH).
        try:
            return _parse_grader_stdout(proc.stdout or "")
        except ValueError:
            pass

        # Both transports failed — surface stdout+stderr tails so the
        # caller (and the agent that submitted) can debug what blew up.
        stdout_tail = (proc.stdout or "")[-1500:]
        stderr_tail = (proc.stderr or "")[-1500:]
        raise RuntimeError(
            "Grader subprocess produced no parseable result. "
            f"stdout tail: {stdout_tail!r} stderr tail: {stderr_tail!r}"
        )
    finally:
        try:
            os.unlink(result_path)
        except OSError:
            pass


def _run_reserved_grader_subprocess(
    grader_file: Path,
    submission_path: str,
    context: dict[str, Any],
    gpu_ids: list[int],
    timeout: float,
) -> dict[str, Any]:
    with _reserve_grader_gpus(gpu_ids, timeout=timeout):
        return _run_grader_subprocess(
            grader_file=grader_file,
            submission_path=submission_path,
            context=context,
            gpu_ids=gpu_ids,
            timeout=timeout,
        )


# -- Path resolution (handles both Docker container paths and host paths) --


def _resolve_submission_path(raw_path: str, workspace_root: Path) -> Path:
    """Resolve a submission path that may be relative or container-absolute.

    Agents inside Docker send relative paths (from /workspace) or absolute
    container paths (/workspace/...). The server translates these to host paths
    using workspace_root.
    """
    p = Path(raw_path)

    # Already an absolute path that exists under this grader runtime.
    if p.is_absolute() and p.is_file():
        return p

    # Absolute container path: strip /workspace prefix and resolve under workspace_root
    if p.is_absolute():
        try:
            rel = p.relative_to("/workspace")
            host_path = workspace_root / rel
            if host_path.is_file():
                return host_path
        except ValueError:
            pass

    # Relative path: resolve under workspace_root
    host_path = workspace_root / raw_path
    if host_path.is_file():
        return host_path

    raise FileNotFoundError(f"Submission file not found: {raw_path}")


# -- Result recording (server-authoritative) --


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


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


def _coerce_bool(value: object, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"1", "true", "yes"}:
            return True
        if lowered in {"0", "false", "no"}:
            return False
    return default


def _read_best_result(path: Path) -> dict[str, Any] | None:
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


def _load_submission_payload(submission_path: Path) -> dict[str, Any]:
    try:
        data = json.loads(submission_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("Submission must be a JSON object with a non-empty `description`.") from exc
    if not isinstance(data, dict):
        raise ValueError("Submission must be a JSON object with a non-empty `description`.")
    return data


def _submission_description(submission_payload: dict[str, Any]) -> str:
    desc = submission_payload.get("description")
    if not isinstance(desc, str) or not desc.strip():
        raise ValueError(
            "Submission must include a non-empty `description` describing the exact submitted solution."
        )
    desc = desc.strip()
    word_count = len(re.findall(r"[A-Za-z0-9_@.+/-]+", desc))
    if len(desc) < 80 or word_count < 12:
        raise ValueError(
            "Submission `description` must be a standalone solution summary of at least "
            "80 characters and 12 words, covering the key method/design choices."
        )
    if re.match(
        r"^(?:v\d+|retry|rerun|tuned?|updated?|bug\s*fix|fix|debug|minor|small|delay)\b",
        desc.lower(),
    ):
        raise ValueError(
            "Submission `description` must describe the complete solution, not a version, "
            "retry, bug fix, or relative change."
        )
    return desc


def _ensure_submission_under_approach_dir(
    submission_path: Path,
    workspace_root: Path,
    approach_id: str,
) -> None:
    submissions_dir = (
        workspace_root / "approach_details" / approach_id / "submissions"
    ).resolve()
    try:
        submission_path.resolve().relative_to(submissions_dir)
    except ValueError as exc:
        raise ValueError(
            "Submission path must be under approach_details/<approach_id>/submissions/."
        ) from exc


def _submission_sha256(submission_path: Path) -> str:
    digest = hashlib.sha256()
    with submission_path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_metadata(code_dir: Path) -> dict[str, Any]:
    def run_git(args: list[str]) -> str:
        proc = subprocess.run(
            ["git", "-C", str(code_dir), *args],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        if proc.returncode != 0:
            return ""
        return proc.stdout.strip()

    if not (code_dir / ".git").exists():
        return {
            "code_git_hash": None,
            "code_git_dirty": None,
            "code_commit_message": "",
        }

    full_hash = run_git(["rev-parse", "HEAD"])
    status = run_git(["status", "--porcelain"])
    commit_message = run_git(["log", "-1", "--format=%B"])
    return {
        "code_git_hash": full_hash or None,
        "code_git_dirty": bool(status) if full_hash or status else None,
        "code_commit_message": commit_message,
    }


def _make_is_better(score_is_better: Callable[[float, float], bool]) -> Callable[[dict[str, Any], dict[str, Any]], bool]:
    """Create a record-level comparator using the score-level is_better function."""
    def _is_better(new: dict[str, Any], old: dict[str, Any]) -> bool:
        """Trusted records always beat untrusted; otherwise compare valid+score."""
        new_trusted = new.get("controller_status") == "secure_graded"
        old_trusted = old.get("controller_status") == "secure_graded"
        if new_trusted and not old_trusted:
            return True
        if not new_trusted and old_trusted:
            return False
        new_valid = 1 if _coerce_bool(new.get("valid"), False) else 0
        old_valid = 1 if _coerce_bool(old.get("valid"), False) else 0
        if new_valid != old_valid:
            return new_valid > old_valid
        new_score = _coerce_float(new.get("score")) or 0.0
        old_score = _coerce_float(old.get("score")) or 0.0
        return score_is_better(new_score, old_score)
    return _is_better


def _record_evaluation(
    *,
    workspace_root: Path,
    approach_id: str,
    submission_path: Path,
    evaluation: dict[str, Any],
    is_better: Callable[[dict[str, Any], dict[str, Any]], bool],
    metadata: dict[str, Any] | None = None,
) -> bool:
    """Write graded results to workspace files.

    Returns True if best_result was updated (i.e., this is the best score so far).
    All writes happen server-side so agents cannot tamper with result files.
    """
    approach_dir = workspace_root / "approach_details" / approach_id

    submission_payload = _load_submission_payload(submission_path)
    description = _submission_description(submission_payload)

    try:
        rel_path = str(submission_path.resolve().relative_to(workspace_root.resolve()))
    except ValueError:
        rel_path = str(submission_path.resolve())

    git_meta = _git_metadata(approach_dir / "code")

    record = {
        "approach_id": approach_id,
        "description": description,
        "score": evaluation["score"],
        "valid": evaluation.get("valid", True),
        "opt_target_met": evaluation.get("opt_target_met", False),
        "notes": evaluation.get("message", ""),
        "error": evaluation.get("error", ""),
        "public_metrics": evaluation.get("public_metrics", {}),
        "submission_path": rel_path,
        "submission_sha256": _submission_sha256(submission_path),
        "solution": submission_payload,
        "evaluated_at": evaluation.get("graded_at", "") or _utc_now_iso(),
        "controller_status": "secure_graded",
        **git_meta,
        "metadata": metadata or {},
    }

    # Write feedback
    feedback_path = approach_dir / "eval_feedback" / "latest_feedback.json"
    feedback_path.parent.mkdir(parents=True, exist_ok=True)
    feedback_path.write_text(
        json.dumps(record, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    # Append to intermediate_results
    intermediate_path = approach_dir / "intermediate_results.jsonl"
    intermediate_path.parent.mkdir(parents=True, exist_ok=True)
    with intermediate_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    # Update best_result if this is the best so far
    best_path = approach_dir / "best_result.jsonl"
    current_best = _read_best_result(best_path)
    best_updated = current_best is None or is_better(record, current_best)
    if best_updated:
        best_path.write_text(
            json.dumps(record, ensure_ascii=False) + "\n", encoding="utf-8"
        )

    try:
        submission_path.unlink()
    except FileNotFoundError:
        pass

    return best_updated


# -- FastAPI app --


def create_app(
    *,
    workspace_root: Path,
    grader_file: Path,
    token: str,
) -> FastAPI:
    # We still load the grader module here to validate it (presence of
    # grade_submission + is_better) at startup and to capture `is_better` for
    # cross-submission ranking. The grade_submission callable returned here
    # is NOT invoked directly; evaluation runs in a subprocess to apply
    # CUDA_VISIBLE_DEVICES dynamically per request.
    _grade_submission_unused, is_better_fn = _load_grader(grader_file)
    del _grade_submission_unused  # silence linters about unused local
    _is_better = _make_is_better(is_better_fn)
    app = FastAPI(title="Eureka Secure Grader")

    @app.get("/healthz")
    def healthz() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/is_better")
    def is_better(
        req: CompareRequest,
        authorization: str = Header(default=""),
    ) -> dict[str, bool]:
        expected = f"Bearer {token}"
        if token and authorization != expected:
            raise HTTPException(status_code=401, detail="Unauthorized")
        return {"is_better": bool(is_better_fn(req.new_score, req.old_score))}

    @app.post("/grade")
    async def grade(
        req: GradeRequest,
        authorization: str = Header(default=""),
    ) -> dict[str, Any]:
        expected = f"Bearer {token}"
        if token and authorization != expected:
            raise HTTPException(status_code=401, detail="Unauthorized")

        # Resolve submission path (handles both container and host paths)
        try:
            submission_path = _resolve_submission_path(req.submission_path, workspace_root)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        try:
            submission_path.relative_to(workspace_root.resolve())
        except ValueError as exc:
            raise HTTPException(
                status_code=400,
                detail="Submission path must stay under the mounted workspace root.",
            ) from exc

        try:
            _ensure_submission_under_approach_dir(
                submission_path, workspace_root, req.approach_id,
            )
            _submission_description(_load_submission_payload(submission_path))
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        # Discover which GPUs the caller currently holds (lock file is the
        # single source of truth — the agent does NOT tell us which GPUs).
        gpu_ids = _resolve_gpu_ids_from_locks(
            workspace_root,
            req.approach_id,
            req.gpu_lock_token,
        )
        eval_timeout = float(os.environ.get("EUREKA_GRADER_TIMEOUT_SECONDS", "1200"))

        try:
            loop = asyncio.get_event_loop()
            raw = await asyncio.wait_for(
                loop.run_in_executor(
                    None,
                    _run_reserved_grader_subprocess,
                    grader_file,
                    str(submission_path),
                    {
                        "workspace_root": str(workspace_root.resolve()),
                        "approach_id": req.approach_id,
                        "metadata": req.metadata,
                    },
                    gpu_ids,
                    eval_timeout,
                ),
                # Cover both GPU-reservation wait and subprocess evaluation,
                # plus a small buffer so inner errors surface first.
                timeout=(2 * eval_timeout) + 30,
            )
        except asyncio.TimeoutError:
            raise HTTPException(status_code=504, detail="Evaluation timed out on the server.")
        except Exception as exc:
            if not gpu_ids and _looks_like_cuda_visibility_failure(exc):
                diagnostic = _gpu_authorization_diagnostic(
                    workspace_root,
                    approach_id=req.approach_id,
                    gpu_lock_token=req.gpu_lock_token,
                )
                if diagnostic is not None:
                    raise HTTPException(status_code=400, detail=diagnostic) from exc
            raise HTTPException(status_code=500, detail=f"Private grader failed: {exc}") from exc
        if not isinstance(raw, dict):
            raise HTTPException(status_code=500, detail="Private grader must return a JSON object.")

        score = _coerce_float(raw.get("score"))
        if score is None:
            raise HTTPException(
                status_code=500, detail="Private grader response is missing numeric `score`."
            )
        public_metrics = raw.get("public_metrics", {})
        if not isinstance(public_metrics, dict):
            public_metrics = {}

        response = {
            "score": score,
            "valid": _coerce_bool(raw.get("valid", raw.get("correct")), True),
            "opt_target_met": _coerce_bool(raw.get("opt_target_met", raw.get("target_met")), False),
            "message": str(raw.get("message", "")).strip(),
            "error": str(raw.get("error", "")).strip(),
            "public_metrics": public_metrics,
            "graded_at": str(raw.get("graded_at", "")).strip() or _utc_now_iso(),
        }

        # Write result files server-side (authoritative, tamper-proof)
        try:
            best_updated = _record_evaluation(
                workspace_root=workspace_root,
                approach_id=req.approach_id,
                submission_path=submission_path,
                evaluation=response,
                is_better=_is_better,
                metadata=req.metadata,
            )
            response["best_result_updated"] = best_updated
        except Exception as exc:
            raise HTTPException(
                status_code=500,
                detail=f"Failed to write evaluation record: {exc}",
            ) from exc

        return response

    return app


def main() -> None:
    parser = argparse.ArgumentParser(description="Start the Eureka secure grader service.")
    parser.add_argument("--workspace-root", required=True)
    parser.add_argument("--hidden-eval-dir", required=True)
    parser.add_argument("--grader-file", default="evaluate.py")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--token", default="")
    args = parser.parse_args()

    workspace_root = Path(args.workspace_root).expanduser().resolve()
    hidden_eval_dir = Path(args.hidden_eval_dir).expanduser().resolve()
    grader_file = (hidden_eval_dir / args.grader_file).resolve()
    if not workspace_root.is_dir():
        raise SystemExit(f"Workspace root not found: {workspace_root}")
    if not grader_file.is_file():
        raise SystemExit(f"Private grader file not found: {grader_file}")
    try:
        hidden_eval_dir.relative_to(workspace_root)
    except ValueError:
        pass
    else:
        raise SystemExit("Private grader directory must be outside the public workspace root.")

    app = create_app(workspace_root=workspace_root, grader_file=grader_file, token=args.token)
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
