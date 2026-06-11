"""GPU coordination helpers for multi-agent experiments on shared servers.

Agents import this module to discover, acquire, and release GPUs. Lock files
coordinate between our agents; memory stats are collected via the driver API.

Public surface (see ``__all__``):
- ``get_gpu_info()`` — inspect GPU availability and lock state
- ``gpu_session()`` — context manager that locks GPUs, sets
  ``CUDA_VISIBLE_DEVICES`` for the duration of the block, then restores the
  previous value and releases the locks on exit
- ``GpuUnavailableError`` — raised when ``gpu_session`` cannot acquire

Other module-level names (``acquire_gpus``, ``release_gpus``,
``set_cuda_devices``, ``release_all_gpu_locks``, …) are implementation
details. They remain importable for controller-side code (e.g. host cleanup)
but are deliberately excluded from ``__all__`` so that ``from gpu_helpers
import *`` only exposes the supported workflow.
"""

from __future__ import annotations

import fcntl
import json
import os
import secrets
import subprocess
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

__all__ = ["get_gpu_info", "gpu_session", "GpuUnavailableError"]


class GpuUnavailableError(Exception):
    """Raised by gpu_session when no GPUs are available within timeout."""


def _workspace_root() -> Path:
    """Resolve workspace root: agent cwd when config exists, else Docker default."""
    cwd = Path.cwd()
    if (cwd / ".gpu_config.json").is_file():
        return cwd
    return Path("/workspace")


def _lock_dir() -> Path:
    return _workspace_root() / ".gpu_locks"


def _coord_file() -> Path:
    return _lock_dir() / ".coord"


def _config_file() -> Path:
    return _workspace_root() / ".gpu_config.json"


def _ensure_lock_dir() -> None:
    _lock_dir().mkdir(parents=True, exist_ok=True)


def _read_config() -> dict[str, Any]:
    path = _config_file()
    if path.is_file():
        return json.loads(path.read_text(encoding="utf-8"))
    return {"allowed_gpu_ids": [], "total_available": 0}


def _is_pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except (ProcessLookupError, PermissionError):
        return False


def _lock_path(gpu_id: int) -> Path:
    return _lock_dir() / f"gpu_{gpu_id}.lock"


def _read_lock(gpu_id: int) -> dict[str, Any] | None:
    path = _lock_path(gpu_id)
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def _write_lock(gpu_id: int, approach_id: str, gpu_lock_token: str) -> None:
    data = {
        "approach_id": approach_id,
        "acquired_at": datetime.now(tz=timezone.utc).isoformat(),
        "pid": os.getpid(),
        "gpu_lock_token": gpu_lock_token,
    }
    _lock_path(gpu_id).write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def _remove_lock(gpu_id: int) -> None:
    try:
        _lock_path(gpu_id).unlink()
    except FileNotFoundError:
        pass


def _cleanup_stale_locks() -> None:
    lock_dir = _lock_dir()
    if not lock_dir.is_dir():
        return
    for lock_path in lock_dir.glob("gpu_*.lock"):
        try:
            data = json.loads(lock_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            try:
                lock_path.unlink()
            except OSError:
                pass
            continue
        pid = data.get("pid", 0)
        if pid and not _is_pid_alive(pid):
            try:
                lock_path.unlink()
            except OSError:
                pass


def _query_driver_gpu_memory() -> list[dict[str, Any]]:
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,memory.used,memory.total",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode != 0:
            return []
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return []

    gpus = []
    for line in result.stdout.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) != 3:
            continue
        try:
            idx = int(parts[0])
            used = float(parts[1])
            total = float(parts[2])
            pct = (used / total * 100) if total > 0 else 0.0
            gpus.append({
                "id": idx,
                "memory_used_pct": round(pct, 1),
                "memory_total_mb": int(total),
            })
        except (ValueError, ZeroDivisionError):
            continue
    return gpus


def get_gpu_info() -> list[dict[str, Any]]:
    """Query GPU status from lock files and driver memory stats.

    Returns list of dicts, one per allowed GPU:
    - id: physical GPU index
    - locked_by: approach_id of locking agent, or None
    - memory_used_pct: GPU memory usage percentage
    - memory_total_mb: total GPU memory in MB
    """
    config = _read_config()
    allowed_ids = config.get("allowed_gpu_ids", [])
    if not allowed_ids:
        return []

    driver_data = {g["id"]: g for g in _query_driver_gpu_memory()}

    result = []
    for gpu_id in sorted(allowed_ids):
        lock = _read_lock(gpu_id)
        locked_by = lock["approach_id"] if lock else None
        driver = driver_data.get(gpu_id, {})
        result.append({
            "id": gpu_id,
            "locked_by": locked_by,
            "memory_used_pct": driver.get("memory_used_pct", 0.0),
            "memory_total_mb": driver.get("memory_total_mb", 0),
        })
    return result


def acquire_gpus(
    gpu_ids: list[int],
    approach_id: str = "",
    gpu_lock_token: str = "",
    timeout: int = 300,
) -> list[int] | None:
    """Acquire specific GPUs atomically using lock files.

    The agent MUST call get_gpu_info() first to check availability,
    then pass the desired GPU IDs here. This ensures the agent always
    makes an informed choice about which GPU(s) to use.

    - gpu_ids=[...]: acquire specific GPUs by physical ID (required)
    Returns list of physical GPU IDs, or None on timeout.
    Raises ValueError if gpu_ids is empty or contains IDs not in allowed list.
    """
    if not gpu_ids:
        raise ValueError("gpu_ids must be a non-empty list")

    config = _read_config()
    allowed_ids = set(config.get("allowed_gpu_ids", []))

    for gid in gpu_ids:
        if gid not in allowed_ids:
            raise ValueError(f"GPU {gid} is not in allowed list {sorted(allowed_ids)}")

    deadline = time.monotonic() + timeout
    lock_dir = _lock_dir()
    coord = _coord_file()
    while True:
        _ensure_lock_dir()
        with open(coord, "w") as coord_f:
            fcntl.flock(coord_f, fcntl.LOCK_EX)
            try:
                _cleanup_stale_locks()

                locked_gpus = set()
                for lock_path in lock_dir.glob("gpu_*.lock"):
                    try:
                        locked_gpus.add(int(lock_path.stem.split("_", 1)[1]))
                    except (json.JSONDecodeError, OSError, ValueError, IndexError):
                        pass

                target_ids = [gid for gid in gpu_ids if gid not in locked_gpus]
                if len(target_ids) == len(gpu_ids):
                    for gid in target_ids:
                        _write_lock(gid, approach_id, gpu_lock_token)
                    return target_ids
            finally:
                fcntl.flock(coord_f, fcntl.LOCK_UN)

        if time.monotonic() >= deadline:
            return None
        time.sleep(2.0)


def release_gpus(
    gpu_ids: list[int],
    approach_id: str = "",
    gpu_lock_token: str = "",
) -> None:
    """Remove lock files for the given GPUs. Idempotent."""
    _ensure_lock_dir()
    coord = _coord_file()
    with open(coord, "w") as coord_f:
        fcntl.flock(coord_f, fcntl.LOCK_EX)
        try:
            for gid in gpu_ids:
                lock = _read_lock(gid)
                if lock is not None:
                    if approach_id and lock.get("approach_id") != approach_id:
                        continue
                    existing_token = lock.get("gpu_lock_token", "")
                    if (existing_token or gpu_lock_token) and existing_token != gpu_lock_token:
                        continue
                    _remove_lock(gid)
                else:
                    _remove_lock(gid)
        finally:
            fcntl.flock(coord_f, fcntl.LOCK_UN)


def release_all_gpu_locks(lock_dir: Path | None = None) -> list[int]:
    """Remove all GPU lock files under *lock_dir* (or the current workspace).

    Returns the physical GPU ids that were released.
    """
    target = (lock_dir if lock_dir is not None else _lock_dir()).resolve()
    if not target.is_dir():
        return []

    target.mkdir(parents=True, exist_ok=True)
    coord = target / ".coord"
    released: list[int] = []
    with open(coord, "w") as coord_f:
        fcntl.flock(coord_f, fcntl.LOCK_EX)
        try:
            for lock_path in list(target.glob("gpu_*.lock")):
                try:
                    gpu_id = int(lock_path.stem.split("_", 1)[1])
                except (ValueError, IndexError):
                    gpu_id = -1
                try:
                    lock_path.unlink()
                    if gpu_id >= 0:
                        released.append(gpu_id)
                except OSError:
                    pass
        finally:
            fcntl.flock(coord_f, fcntl.LOCK_UN)
    return sorted(released)


def set_cuda_devices(gpu_ids: list[int]) -> None:
    """Set CUDA_VISIBLE_DEVICES for the current process.

    Deprecated for agent use — ``gpu_session()`` now sets this automatically
    inside the ``with`` block and restores the prior value on exit. Kept for
    backwards compatibility and controller-side utilities.
    """
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(g) for g in gpu_ids)


@contextmanager
def gpu_session(
    gpu_ids: list[int],
    approach_id: str = "",
    timeout: int = 300,
):
    """Context manager: acquire GPUs, expose them to CUDA, release on exit.

    Performs four steps as a single unit:

    1. Acquire ``gpu_ids`` via the lock-file protocol (atomic, fcntl-guarded).
    2. Set ``CUDA_VISIBLE_DEVICES`` so CUDA in this process sees the acquired
       GPUs (and only them).
    3. Yield the acquired physical GPU ids to the ``with`` block.
    4. On exit (including exceptions), restore the previous
       ``CUDA_VISIBLE_DEVICES`` value and release all locks.

    Raises ``GpuUnavailableError`` if the GPUs cannot be acquired within
    ``timeout`` seconds. Callers should call ``get_gpu_info()`` first to
    choose an available GPU.
    """
    if not approach_id.strip():
        raise ValueError("gpu_session requires a non-empty approach_id")
    gpu_lock_token = secrets.token_urlsafe(24)
    result = acquire_gpus(
        gpu_ids=gpu_ids,
        approach_id=approach_id,
        gpu_lock_token=gpu_lock_token,
        timeout=timeout,
    )
    if result is None:
        raise GpuUnavailableError(
            f"GPU(s) {gpu_ids} not available within {timeout}s"
        )
    # Snapshot prior CUDA_VISIBLE_DEVICES so we can restore on exit.
    _prev_cvd: str | None = os.environ.get("CUDA_VISIBLE_DEVICES")
    _prev_token: str | None = os.environ.get("EUREKA_GPU_LOCK_TOKEN")
    try:
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(g) for g in result)
        os.environ["EUREKA_GPU_LOCK_TOKEN"] = gpu_lock_token
        yield result
    finally:
        if _prev_cvd is None:
            os.environ.pop("CUDA_VISIBLE_DEVICES", None)
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = _prev_cvd
        if _prev_token is None:
            os.environ.pop("EUREKA_GPU_LOCK_TOKEN", None)
        else:
            os.environ["EUREKA_GPU_LOCK_TOKEN"] = _prev_token
        release_gpus(result, approach_id=approach_id, gpu_lock_token=gpu_lock_token)
