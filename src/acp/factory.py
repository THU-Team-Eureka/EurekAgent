"""Adapter factory for Docker-backed Claude sessions."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from ..docker.adapter import DockerPtyAdapter, DockerStreamAdapter
from ..runtime import get_docker_container
from .stream_adapter import StreamAdapter

if TYPE_CHECKING:
    from ..config import Config
    from .pty_adapter import PtyAdapter


def make_adapter(config: "Config", workspace: Path) -> "StreamAdapter | PtyAdapter":
    if config.adapter_mode == "pty":
        return DockerPtyAdapter(
            config.claude_command, get_docker_container(), workspace_dir=workspace,
        )

    return DockerStreamAdapter(config.claude_command, get_docker_container())
