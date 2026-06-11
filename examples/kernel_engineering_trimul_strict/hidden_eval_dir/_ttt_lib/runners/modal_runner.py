import signal
import subprocess
import traceback
from contextlib import contextmanager
from typing import Optional

import modal  # pyright: ignore[reportMissingImports]

from libkernelbot.run_eval import FullResult, SystemInfo, run_config

class TimeoutException(Exception):
    pass


class ModalRequeueRequest(Exception):
    """Raise to force Modal to retry (requeue) the call."""

_REQUEUE_SENTINEL = "[MODAL_REQUEUE]"

# If we detect any GPU with one of these reported `nvidia-smi` names, we refuse to run.
# For consistency with the backend, we only allow A100 PCIe 80GB and H100 HBM3 PCIe 80GB.
_BANNED_GPU_NAMES = {
    "NVIDIA A100-SXM4-80GB",
    "NVIDIA H100 NVL",
}

_REQUEUE_COUNT_DICT_NAME = "discord-bot-requeue-counts"
_requeue_counts = None


def _get_requeue_counts():
    global _requeue_counts
    if _requeue_counts is None:
        _requeue_counts = modal.Dict.from_name(_REQUEUE_COUNT_DICT_NAME, create_if_missing=True)
    return _requeue_counts


def _get_request_id(config: dict) -> Optional[str]:
    rid = config.get("_modal_request_id")
    if isinstance(rid, str) and rid.strip():
        return rid
    return None


def _increment_requeue_count(request_id: str) -> int:
    d = _get_requeue_counts()
    try:
        current = int(d[request_id])
    except KeyError:
        current = 0
    current += 1
    d[request_id] = current
    return current


def _pop_requeue_count(request_id: str) -> int:
    d = _get_requeue_counts()
    try:
        value = d.pop(request_id)
    except KeyError:
        value = 0
    return int(value or 0)


def _detect_nvidia_gpu_names() -> list[str]:
    """
    Best-effort GPU name detection using nvidia-smi.

    Returns:
        A list of GPU names (one per visible GPU). Empty list if unavailable.
    """
    try:
        proc = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            check=False,
            capture_output=True,
            text=True,
        )
    except Exception:
        return []

    if proc.returncode != 0:
        return []

    names = [line.strip() for line in proc.stdout.splitlines() if line.strip()]
    return names


@contextmanager
def timeout(seconds: int):
    """Context manager that raises TimeoutException after specified seconds"""

    def timeout_handler(signum, frame):
        raise TimeoutException(f"Script execution timed out after {seconds} seconds")

    # Set up the signal handler
    original_handler = signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(seconds)

    try:
        yield
    finally:
        # Restore the original handler and disable the alarm
        signal.alarm(0)
        signal.signal(signal.SIGALRM, original_handler)


def modal_run_config(  # noqa: C901
    config: dict,
    timeout_seconds: int = 1200,
) -> FullResult:
    """Modal version of run_pytorch_script, handling timeouts"""
    request_id = _get_request_id(config)
    try:
        gpu_names = _detect_nvidia_gpu_names()
        if any(name in _BANNED_GPU_NAMES for name in gpu_names):
            attempt = None
            if request_id is not None:
                attempt = _increment_requeue_count(request_id)
            # Use a built-in exception type so the local caller can always deserialize it.
            # (Custom exceptions require the same module to exist locally.)
            attempt_msg = ""
            if attempt is not None:
                # Modal's max_retries counts requeues, not total attempts.
                attempt_msg = f" (attempt={attempt}, requeues_so_far={max(0, attempt - 1)})"
            raise RuntimeError(
                f"{_REQUEUE_SENTINEL} Refusing to run on banned GPU(s) {gpu_names}{attempt_msg}; requeueing request."
            )

        with timeout(timeout_seconds):
            result = run_config(config)
        if request_id is not None:
            result.system.requeues = _pop_requeue_count(request_id)
        return result
    except RuntimeError as e:
        # Only propagate our sentinel runtime error to trigger Modal retries.
        # Any other RuntimeError from user code should *not* be requeued.
        if str(e).startswith(_REQUEUE_SENTINEL):
            raise
        requeues = _pop_requeue_count(request_id) if request_id is not None else 0
        exception = "".join(traceback.format_exception(e))
        return FullResult(
            success=False,
            error=f"Error executing script:\n{exception}",
            runs={},
            system=SystemInfo(requeues=requeues),
        )
    except TimeoutException as e:
        requeues = _pop_requeue_count(request_id) if request_id is not None else 0
        return FullResult(
            success=False,
            error=f"Timeout Error: {str(e)}",
            runs={},
            system=SystemInfo(requeues=requeues),
        )
    except BaseException as e:
        # Important: user submissions may raise SystemExit (e.g., via sys.exit or argparse)
        # which is not an Exception. If we let it propagate, Modal retries would requeue it,
        # which we do NOT want. Convert all other throwables into a normal failure result.
        requeues = _pop_requeue_count(request_id) if request_id is not None else 0
        exception = "".join(traceback.format_exception(e))
        return FullResult(
            success=False,
            error=f"Error executing script:\n{exception}",
            runs={},
            system=SystemInfo(requeues=requeues),
        )
