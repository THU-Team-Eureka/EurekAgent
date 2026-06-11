"""LangGraph state definition for the Eureka loop engine."""

from __future__ import annotations

from operator import add
from typing import Annotated, Any, TypedDict


class LoopState(TypedDict, total=False):
    # Problem definition and stage budgets.
    problem: str
    propose_time_limit_per_session: str
    implement_time_limit_per_session: str

    # Run metadata.
    run_id: str
    run_dir: str
    workspace_dir: str

    # Loop control.
    loop_index: int
    max_loops: int
    max_num_approaches: int
    next_stage: str

    # Stage results.
    prepare_status: str
    propose_status: str
    implement_status: str
    prepare_abort_reason: str
    propose_abort_reason: str
    implement_abort_reason: str
    end_reason: str

    # Artifact paths.
    manifest_path: str
    ranked_history_path: str
    prepare_summary_path: str

    # Cost tracking.
    model: str
    cost_limit: float | None
    input_token_price: float | None
    cache_creation_token_price: float | None
    cache_read_token_price: float | None
    output_token_price: float | None
    cost_currency: str

    # Append-only event log.
    history: Annotated[list[dict[str, Any]], add]
