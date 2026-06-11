"""Thin agent-facing client: submit candidates to the grader service.

This client ONLY makes an HTTP call to the grader server and prints the
response.

Usage inside Docker:
    python3 /workspace/eval/eureka_submit.py \
        --approach-dir approach_details/<id> \
        --submission approach_details/<id>/submissions/candidate_001.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any
from urllib import error, request


class SecureEvalError(RuntimeError):
    """Raised when secure grading fails."""


def _normalize_submit_url(value: str) -> str:
    stripped = value.strip().rstrip("/")
    if not stripped:
        raise SecureEvalError("Secure grading is enabled but EUREKA_SECURE_SUBMIT_URL is not set.")
    if stripped.endswith("/grade"):
        return stripped
    return stripped + "/grade"


def submit_for_grading(
    *,
    submit_url: str,
    submit_token: str,
    submission_path: Path,
    approach_id: str,
    timeout_seconds: float = 600.0,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """POST a candidate to the grader server and return the response dict.

    Sends a relative path from /workspace so the server can resolve it
    against its workspace root.
    """
    try:
        rel_submission = str(submission_path.resolve().relative_to(Path("/workspace")))
    except ValueError:
        rel_submission = str(submission_path.resolve())

    payload = {
        "submission_path": rel_submission,
        "approach_id": approach_id,
        "gpu_lock_token": os.environ.get("EUREKA_GPU_LOCK_TOKEN", ""),
        "metadata": metadata or {},
    }
    body = json.dumps(payload).encode("utf-8")
    req = request.Request(
        _normalize_submit_url(submit_url),
        data=body,
        method="POST",
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {submit_token}",
        },
    )
    opener = request.build_opener(request.ProxyHandler({}))
    try:
        with opener.open(req, timeout=timeout_seconds) as response:
            raw = response.read().decode("utf-8")
    except error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise SecureEvalError(
            f"Secure grader rejected submission with HTTP {exc.code}: {detail}"
        ) from exc
    except error.URLError as exc:
        raise SecureEvalError(f"Failed to reach secure grader: {exc}") from exc

    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise SecureEvalError("Secure grader returned invalid JSON.") from exc
    if not isinstance(data, dict):
        raise SecureEvalError("Secure grader returned a non-object response.")
    return data


def main() -> None:
    """CLI entrypoint: python3 /workspace/eval/eureka_submit.py --approach-dir ... --submission ..."""
    parser = argparse.ArgumentParser(description="Submit a candidate to the secure grader.")
    parser.add_argument("--approach-dir", required=True)
    parser.add_argument("--submission", required=True)
    parser.add_argument("--approach-id", default="")
    parser.add_argument("--submit-url", default=os.environ.get("EUREKA_SECURE_SUBMIT_URL", ""))
    parser.add_argument("--submit-token", default=os.environ.get("EUREKA_SECURE_SUBMIT_TOKEN", ""))
    parser.add_argument("--timeout-seconds", type=float,
                        default=float(os.environ.get("EUREKA_EVAL_TIMEOUT_SECONDS", "600")))
    parser.add_argument("--metadata", default=os.environ.get("EUREKA_METADATA", "{}"),
                        help="Optional JSON metadata to pass to the grader")
    args = parser.parse_args()

    approach_dir = Path(args.approach_dir).expanduser().resolve()
    submission_path = Path(args.submission).expanduser().resolve()
    approach_id = args.approach_id.strip() or approach_dir.name

    if not approach_dir.is_dir():
        print(f"Approach directory not found: {approach_dir}", file=sys.stderr)
        sys.exit(1)
    if not submission_path.is_file():
        print(f"Submission file not found: {submission_path}", file=sys.stderr)
        sys.exit(1)

    try:
        metadata = json.loads(args.metadata)
    except json.JSONDecodeError:
        metadata = {}

    try:
        raw = submit_for_grading(
            submit_url=args.submit_url,
            submit_token=args.submit_token,
            submission_path=submission_path,
            approach_id=approach_id,
            timeout_seconds=args.timeout_seconds,
            metadata=metadata,
        )
    except SecureEvalError as exc:
        print(str(exc), file=sys.stderr)
        sys.exit(1)

    print(json.dumps(raw, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
