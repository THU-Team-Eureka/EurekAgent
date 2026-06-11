"""Controller-side GPU lock cleanup between pipeline stages."""

from __future__ import annotations

import logging
from pathlib import Path

from .eval_grader.gpu_helpers import release_all_gpu_locks

log = logging.getLogger(__name__)


def sweep_gpu_locks(workspace: Path, *, stage: str) -> list[int]:
    """Release unreleased GPU locks after a stage completes."""
    workspace = workspace.resolve()
    released: list[int] = []
    for lock_dir in _lock_directories(workspace):
        released.extend(release_all_gpu_locks(lock_dir))
    released = sorted(set(released))
    if released:
        log.warning(
            "Released %d GPU lock(s) at end of %s stage: %s",
            len(released),
            stage,
            released,
        )
    return released


def _lock_directories(workspace: Path) -> list[Path]:
    """Workspace locks plus legacy /workspace path used inside Docker."""
    dirs = [workspace / ".gpu_locks"]
    legacy = Path("/workspace/.gpu_locks")
    if legacy.resolve() != dirs[0].resolve():
        dirs.append(legacy)
    return dirs
