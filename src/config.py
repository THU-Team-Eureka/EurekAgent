"""Runtime configuration for the Eureka loop engine."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Config:
    max_loops: int = 5
    max_num_approaches: int = 5
    propose_time_limit_per_session: str = "20 minutes"
    implement_time_limit_per_session: str = "120 minutes"
    claude_command: str = "claude"
    runs_dir: str = "runs"
    # Cost tracking.
    cost_limit: float | None = None
    input_token_price: float | None = None
    cache_creation_token_price: float | None = None
    cache_read_token_price: float | None = None
    output_token_price: float | None = None
    model: str | None = None
    cost_currency: str = "USD"
    # Docker/runtime isolation.
    docker_image: str = "eureka-agent:node22-bookworm"
    docker_network: str = "host"
    # "auto", "none", or CUDA device IDs like "0,1".
    gpus: str = "auto"
    # Secure evaluation inputs.
    hidden_eval_dir: str = ""
    submission_format_path: str = ""
    # "stream" uses stream-json; "pty" uses interactive Claude.
    adapter_mode: str = "pty"
    # Skip prepare when the environment is already known-good.
    skip_prepare: bool = False
