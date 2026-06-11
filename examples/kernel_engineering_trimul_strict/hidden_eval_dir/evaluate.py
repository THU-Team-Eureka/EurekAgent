"""Eureka adapter for TTT-Discover's TriMul kernel evaluator.

This file is a *thin wrapper*: it extracts ``kernel_code`` from the Eureka
submission JSON and hands it off to the **UNMODIFIED** TTT-Discover
evaluation pipeline shipped under ``_ttt_lib/``. All scoring decisions
(18 correctness tests, 7-benchmark geometric mean, subprocess isolation,
Cantor-paired seeds, ``err/mean < 0.001`` convergence, 30s per-case
benchmark cap, recheck=True on every leaderboard iteration, ...) are made
by TTT-Discover code -- this file only translates I/O formats.

Pipeline:
    Eureka JSON ─► kernel_code
                   │
                   ▼
            build_task_config(task=trimul, submission=code,
                              mode=LEADERBOARD)
                   │
                   ▼ run_config (lib/libkernelbot/run_eval.py)
            spawns python3 subprocess running
            lib/bioml/trimul/eval.py (multiprocessing.Pool(1) spawn)
                   │
                   ▼
            FullResult ─► compute_score (geom mean, seconds)
                   │
                   ▼
            score_us = score_sec * 1e6  ──►  Eureka result dict
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
import types
from pathlib import Path
from typing import Any

EVAL_DIR = Path(__file__).resolve().parent
TTT_LIB = EVAL_DIR / "_ttt_lib"

# Expose libkernelbot/* as top-level package, same as TTT-Discover's own
# examples/gpu_mode/ layout (their env.py does `import libkernelbot.consts`).
# Insert at position 0 so we win over any conflicting package on PYTHONPATH.
if str(TTT_LIB) not in sys.path:
    sys.path.insert(0, str(TTT_LIB))


def _stub_irrelevant_deps() -> None:
    """``libkernelbot/submission.py`` and ``leaderboard_db.py`` were authored
    for the GPUMode Discord bot + Postgres leaderboard backend. They import
    ``better_profanity`` (filename profanity check used only in
    ``SubmissionRequest``) and ``psycopg2`` (Postgres driver) at module top.

    Neither is exercised by the evaluation path we use (``compute_score``).
    Rather than touch TTT-Discover source we inject minimal stubs into
    ``sys.modules`` *before* importing libkernelbot.submission.
    """
    if "better_profanity" not in sys.modules:
        bp = types.ModuleType("better_profanity")

        class _Profanity:
            def contains_profanity(self, s):  # pragma: no cover - unused path
                return False

            def load_censor_words(self, *a, **kw):  # pragma: no cover
                return None

        bp.profanity = _Profanity()
        sys.modules["better_profanity"] = bp

    if "psycopg2" not in sys.modules:
        pg = types.ModuleType("psycopg2")
        pg_extras = types.ModuleType("psycopg2.extras")
        pg_errors = types.ModuleType("psycopg2.errors")

        class _Err(Exception):  # pragma: no cover - unused path
            pass

        pg.Error = _Err
        pg.DatabaseError = _Err
        pg.OperationalError = _Err
        pg.connect = lambda *a, **kw: (_ for _ in ()).throw(
            RuntimeError("psycopg2 stubbed out — evaluator does not need a DB")
        )
        pg.extras = pg_extras
        pg.errors = pg_errors
        sys.modules["psycopg2"] = pg
        sys.modules["psycopg2.extras"] = pg_extras
        sys.modules["psycopg2.errors"] = pg_errors


_stub_irrelevant_deps()

from libkernelbot.consts import SubmissionMode  # noqa: E402
from libkernelbot.run_eval import run_config  # noqa: E402
from libkernelbot.submission import compute_score  # noqa: E402
from libkernelbot.task import build_task_config, make_task_definition  # noqa: E402

TRIMUL_TASK_YML = TTT_LIB / "bioml" / "trimul" / "task.yml"

# Eureka deployment hint: TRIMUL_EVAL_PYTHON should point to a python with
# torch+triton+CUDA. TTT-Discover's run_eval.py spawns ``python3`` as a literal
# argv[0], so we either need ``python3`` to resolve to that interpreter via
# PATH, or we drop a symlink into a private bin directory and prepend it.
_PYTHON3_SHIM_DIR: Path | None = None


def _ensure_python3_resolves_to_cuda_python() -> None:
    """Make sure ``python3`` in PATH points at a CUDA/torch/triton-capable
    interpreter, because TTT-Discover's run_program() literally argv[0]s
    ``python3``. We do **not** modify TTT-Discover code -- we just bend the
    environment to make its hard-coded ``python3`` resolve correctly.

    We can't simply symlink ``python3 -> .venv/bin/python``: when a venv's
    python is launched via a symlink located *outside* the venv tree, the
    venv site-packages logic (which inspects ``sys.executable``'s directory
    for ``pyvenv.cfg``) does not engage, so ``import torch`` fails. The
    workaround is a thin shell wrapper that ``exec``s the real interpreter
    by its venv-internal path, preserving ``sys.executable`` semantics.
    """
    global _PYTHON3_SHIM_DIR
    desired = os.environ.get("TRIMUL_EVAL_PYTHON") or sys.executable
    if not desired or not Path(desired).exists():
        return  # best effort; subprocess will fail loudly if mis-set

    if _PYTHON3_SHIM_DIR is None:
        _PYTHON3_SHIM_DIR = Path(tempfile.mkdtemp(prefix="trimul_py3_shim_"))
        shim = _PYTHON3_SHIM_DIR / "python3"
        # Shell wrapper instead of symlink, so the launched interpreter's
        # sys.executable preserves the venv layout and pyvenv.cfg resolution.
        shim.write_text(f'#!/bin/sh\nexec "{desired}" "$@"\n')
        shim.chmod(0o755)
    cur_path = os.environ.get("PATH", "")
    shim_str = str(_PYTHON3_SHIM_DIR)
    if shim_str not in cur_path.split(os.pathsep):
        os.environ["PATH"] = shim_str + os.pathsep + cur_path


def grade_submission(submission_path: str, context: dict[str, Any]) -> dict[str, Any]:
    """Grade a TriMul kernel submission using the unmodified TTT-Discover pipeline.

    Args:
        submission_path: JSON file with ``{"kernel_code": "...", "description": "..."}``.
        context: Grader context (workspace_root, approach_id, metadata). Unused.

    Returns:
        Dict with score (float us, lower better), valid (bool), message (str),
        opt_target_met (bool), public_metrics (dict).
    """
    # 1. Read and validate submission JSON
    try:
        with open(submission_path, "r", encoding="utf-8") as f:
            payload = json.load(f)
    except (OSError, json.JSONDecodeError) as exc:
        return _invalid(f"Failed to read submission JSON: {exc}")
    if not isinstance(payload, dict):
        return _invalid("Submission must be a JSON object.")
    kernel_code = payload.get("kernel_code", "")
    if not kernel_code or not isinstance(kernel_code, str):
        return _invalid("Submission must contain non-empty 'kernel_code' string.")
    if "@triton.jit" not in kernel_code:
        return _invalid("kernel_code must contain at least one @triton.jit function.")

    # 2. Ensure subprocess will find a CUDA-capable python3
    _ensure_python3_resolves_to_cuda_python()

    # 3. Build TTT-Discover task config from the ORIGINAL task.yml
    try:
        task_def = make_task_definition(TRIMUL_TASK_YML)
        task = task_def.task
        config = build_task_config(
            task=task,
            submission_content=kernel_code,
            arch=None,  # Triton .py: arch unused
            mode=SubmissionMode.LEADERBOARD,  # runs tests + leaderboard benchmark
        )
    except Exception as exc:
        return _invalid(f"Failed to build TTT-Discover task config: {exc}")

    # 4. run_config writes sources into CWD, so use a fresh tempdir
    orig_cwd = Path.cwd()
    with tempfile.TemporaryDirectory(prefix="trimul_eureka_") as tmpdir:
        try:
            os.chdir(tmpdir)
            result = run_config(config)
        except Exception as exc:
            os.chdir(orig_cwd)
            return _invalid(f"TTT-Discover run_config raised: {exc!r}")
        finally:
            os.chdir(orig_cwd)

    if not result.success:
        return _invalid(f"TTT-Discover runner failed: {result.error}")

    # 5. Gate on correctness (test mode result must exist and be passed)
    test_res = result.runs.get("test")
    if test_res is None or test_res.run is None:
        return _invalid("TTT-Discover did not produce a 'test' run result.")
    if not test_res.run.success:
        return _invalid(
            f"Test run did not succeed (exit={test_res.run.exit_code}): "
            f"{(test_res.run.stderr or '')[-400:]}"
        )
    if not test_res.run.passed:
        # Surface which correctness case failed if available
        failed_msg = ""
        for k, v in (test_res.run.result or {}).items():
            if k.endswith(".error"):
                failed_msg = f"{k}={v}"
                break
        return _invalid(f"Correctness failed. {failed_msg}")

    # 6. Compute leaderboard score (geom mean of nanoseconds, returned in seconds)
    if "leaderboard" not in result.runs:
        return _invalid("No leaderboard run in result.")
    lb_res = result.runs["leaderboard"]
    if lb_res.run is None or not lb_res.run.success:
        return _invalid(
            f"Leaderboard run did not succeed: {(lb_res.run.stderr if lb_res.run else '')[-400:]}"
        )

    try:
        score_sec = compute_score(result, task, submission_id=-1)
    except Exception as exc:
        return _invalid(f"compute_score failed: {exc!r}")
    score_us = float(score_sec) * 1_000_000.0

    # Pull per-benchmark stats for public metrics
    lb_result_dict = lb_res.run.result or {}
    try:
        num_bench = int(lb_result_dict.get("benchmark-count", 0))
    except (TypeError, ValueError):
        num_bench = 0
    per_bench_us = []
    for i in range(num_bench):
        v = lb_result_dict.get(f"benchmark.{i}.mean")
        if v is not None:
            try:
                per_bench_us.append(round(float(v) / 1000.0, 1))  # ns -> us
            except (TypeError, ValueError):
                pass

    test_result_dict = (test_res.run.result or {}) if test_res and test_res.run else {}
    try:
        tests_passed = int(test_result_dict.get("test-count", 0))
    except (TypeError, ValueError):
        tests_passed = 0

    return {
        "score": score_us,
        "valid": True,
        "opt_target_met": score_us < 3000.0,
        "message": (
            f"Geometric mean across {num_bench} benchmarks: {score_us:.1f} us "
            f"(A100-SXM4-80GB, TTT-Discover pipeline)"
        ),
        "public_metrics": {
            "per_benchmark_us": per_bench_us,
            "tests_passed": tests_passed,
            "benchmarks_run": num_bench,
        },
    }


def is_better(new_score: float, old_score: float) -> bool:
    """Lower runtime (us) is better for the TriMul kernel optimization task."""
    return new_score < old_score


def _invalid(message: str) -> dict[str, Any]:
    """Standardized invalid-submission result."""
    return {
        "score": float("inf"),
        "valid": False,
        "opt_target_met": False,
        "message": message,
        "public_metrics": {},
    }
