"""Runtime GPU policy resolution shared by Docker and agent prompts."""

from __future__ import annotations

import os
import re
import subprocess
from dataclasses import dataclass, field
from logging import Logger


@dataclass(frozen=True)
class GpuPolicy:
    requested: str
    allowed_gpu_ids: list[int]
    enable_docker_gpus: bool = False
    warnings: tuple[str, ...] = field(default_factory=tuple)


_EMITTED_WARNINGS: set[str] = set()


def log_gpu_policy_warnings(policy: GpuPolicy, logger: Logger) -> None:
    """Log GPU policy warnings once per process."""
    for warning in policy.warnings:
        if warning in _EMITTED_WARNINGS:
            continue
        _EMITTED_WARNINGS.add(warning)
        logger.warning(warning)


def validate_gpu_request(gpus: str | None) -> None:
    """Validate CLI syntax for --gpus without probing the host."""
    value = _normalize_gpu_value(gpus)
    if value in {"auto", "none"}:
        return
    _parse_gpu_ids(value)


def resolve_gpu_policy(gpus: str | None) -> GpuPolicy:
    """Resolve the effective GPU allowlist for the current host.

    `--gpus auto` adapts to the host. Explicit IDs are honored only when the
    local Docker/runtime can actually expose NVIDIA GPUs. On machines without
    Docker NVIDIA support (for example macOS Docker Desktop), explicit GPU
    requests degrade to CPU-only instead of letting `docker run --gpus ...`
    fail at startup.
    """
    requested = _normalize_gpu_value(gpus)
    warnings: list[str] = []

    if requested == "none":
        return GpuPolicy(requested=requested, allowed_gpu_ids=[])

    docker_has_gpus = _docker_has_nvidia_runtime()

    if requested == "auto":
        allowed = _cuda_visible_devices() or _host_nvidia_gpu_ids()
        if not docker_has_gpus:
            return GpuPolicy(requested=requested, allowed_gpu_ids=[])
        return GpuPolicy(
            requested=requested,
            allowed_gpu_ids=allowed,
            enable_docker_gpus=bool(allowed),
        )

    requested_ids = _parse_gpu_ids(requested)
    if not docker_has_gpus:
        warnings.append(
            f"--gpus {requested} was requested, but Docker has no NVIDIA runtime "
            "on this host; continuing with no GPUs."
        )
        return GpuPolicy(
            requested=requested,
            allowed_gpu_ids=[],
            warnings=tuple(warnings),
        )

    detected_ids = _host_nvidia_gpu_ids()
    if not detected_ids:
        warnings.append(
            f"--gpus {requested} was requested, but no local NVIDIA GPUs were "
            "detected; continuing with no GPUs."
        )
        return GpuPolicy(
            requested=requested,
            allowed_gpu_ids=[],
            warnings=tuple(warnings),
        )

    detected = set(detected_ids)
    allowed = [gpu_id for gpu_id in requested_ids if gpu_id in detected]
    missing = [gpu_id for gpu_id in requested_ids if gpu_id not in detected]
    if missing:
        if allowed:
            warnings.append(
                f"--gpus {requested} includes unavailable GPU id(s) {missing}; "
                f"using available requested id(s) {allowed}."
            )
        else:
            warnings.append(
                f"--gpus {requested} does not match detected GPU id(s) "
                f"{detected_ids}; continuing with no GPUs."
            )

    return GpuPolicy(
        requested=requested,
        allowed_gpu_ids=allowed,
        enable_docker_gpus=bool(allowed),
        warnings=tuple(warnings),
    )


def _normalize_gpu_value(gpus: str | None) -> str:
    if gpus is None:
        return "auto"
    value = gpus.strip()
    return value.lower() if value.lower() in {"auto", "none"} else value


def _parse_gpu_ids(value: str) -> list[int]:
    if not value or value.endswith(",") or value.startswith(","):
        raise ValueError('--gpus must be "auto", "none", or comma-separated GPU IDs like "0,1"')
    ids: list[int] = []
    seen: set[int] = set()
    for part in value.split(","):
        part = part.strip()
        if not re.fullmatch(r"\d+", part):
            raise ValueError(
                f'Invalid --gpus value "{value}". Use "auto", "none", '
                'or comma-separated GPU IDs like "0,1".'
            )
        gpu_id = int(part)
        if gpu_id not in seen:
            ids.append(gpu_id)
            seen.add(gpu_id)
    if not ids:
        raise ValueError('--gpus must include at least one GPU ID, or use "auto"/"none"')
    return ids


def _cuda_visible_devices() -> list[int]:
    value = os.environ.get("CUDA_VISIBLE_DEVICES")
    if value is None:
        return []
    value = value.strip()
    if not value or value.lower() in {"none", "void", "nodevfiles"}:
        return []
    ids: list[int] = []
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        if not part.isdigit():
            return []
        ids.append(int(part))
    return ids


def _host_nvidia_gpu_ids() -> list[int]:
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10, check=False,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return []
    if result.returncode != 0:
        return []
    return [
        int(line.strip())
        for line in result.stdout.splitlines()
        if line.strip().isdigit()
    ]


def _docker_has_nvidia_runtime() -> bool:
    try:
        result = subprocess.run(
            ["docker", "info", "--format", "{{json .Runtimes}}"],
            capture_output=True, text=True, check=False,
        )
    except FileNotFoundError:
        return False
    return result.returncode == 0 and '"nvidia"' in result.stdout
