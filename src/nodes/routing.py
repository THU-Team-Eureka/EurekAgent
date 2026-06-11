"""Routing functions for the LangGraph state machine.

Each function inspects the current LoopState and returns
the name of the next node (or "end") for a conditional edge.
"""

from __future__ import annotations

from ..runtime import get_token_tracker, write_pipeline_state
from ..state import LoopState


def _cost_limit_exceeded(state: LoopState) -> bool:
    """Check if the total run cost has exceeded the cost_limit."""
    cost_limit = state.get("cost_limit")
    if cost_limit is None:
        return False
    tracker = get_token_tracker()
    current_cost = tracker.calculate_cost()
    if current_cost is None:
        return False
    if current_cost >= cost_limit:
        write_pipeline_state(pipeline_status="abort")
        return True
    return False


def route_from_entry(state: LoopState) -> str:
    """Route from the entry node to the first stage."""
    stage = state.get("next_stage", "prepare")
    if _cost_limit_exceeded(state):
        return "end"
    return stage if stage in {"prepare", "propose", "implement"} else "end"


def route_after_propose(state: LoopState) -> str:
    """Route after the propose stage completes."""
    if _cost_limit_exceeded(state):
        return "end"
    return "implement" if state.get("propose_status") == "ready" else "end"


def route_after_prepare(state: LoopState) -> str:
    """Route after the prepare stage completes."""
    return "propose" if state.get("prepare_status") == "ready" else "end"


def route_after_implement(state: LoopState) -> str:
    """Route after the implement stage completes.

    Routing is based on loop limit and cost limit. A round with zero valid
    results stops instead of proposing from unusable evidence.
    """
    if _cost_limit_exceeded(state):
        return "end"
    implement_status = state.get("implement_status", "")
    if implement_status == "abort":
        return "end"
    loop_index = state.get("loop_index", 0)
    max_loops = state.get("max_loops", 1)
    if loop_index == 0:
        return "end"
    if implement_status == "all_failed":
        return "end"
    if loop_index < max_loops:
        return "propose"
    return "end"
