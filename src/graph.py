"""LangGraph definition for the Eureka loop pipeline.

Graph topology:
    START -> entry -> prepare -> propose -> implement --+
                          ^                            |
                          +---------- (loop) ----------+
"""

from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from .nodes.prepare_node import prepare_node
from .nodes.propose_node import propose_node
from .nodes.implement_node import implement_node
from .nodes.routing import (
    route_after_implement,
    route_after_prepare,
    route_after_propose,
    route_from_entry,
)
from .state import LoopState


def build_graph() -> StateGraph:
    graph = StateGraph(LoopState)

    graph.add_node("entry", lambda s: s)
    graph.add_node("prepare", prepare_node)
    graph.add_node("propose", propose_node)
    graph.add_node("implement", implement_node)

    graph.add_edge(START, "entry")
    graph.add_conditional_edges(
        "entry",
        route_from_entry,
        {"prepare": "prepare", "propose": "propose", "implement": "implement", "end": END},
    )
    graph.add_conditional_edges(
        "prepare",
        route_after_prepare,
        {"propose": "propose", "end": END},
    )
    graph.add_conditional_edges(
        "propose",
        route_after_propose,
        {"implement": "implement", "end": END},
    )
    graph.add_conditional_edges(
        "implement",
        route_after_implement,
        {"propose": "propose", "end": END},
    )

    return graph


def compile_graph():
    return build_graph().compile()
