"""Graph builder."""

from __future__ import annotations

from typing import Any

from backend.app.core.graph_fallback import SequentialGraphRunner
from backend.app.graph.nodes import (
    GraphDependencies,
    decision_output_a,
    decision_output_b,
    fusion_node,
    load_context,
    perception_stub,
    reassessment_node,
    retrieval_node,
    safety_node,
    writeback_node,
)
from backend.app.graph.state import GraphState


def build_graph(deps: GraphDependencies) -> Any:
    try:
        from langgraph.graph import END, START, StateGraph
    except Exception:
        return SequentialGraphRunner(
            [
                lambda state: load_context(state, deps),
                lambda state: perception_stub(state, deps),
                lambda state: fusion_node(state, deps),
                lambda state: reassessment_node(state, deps),
                lambda state: retrieval_node(state, deps),
                lambda state: decision_output_a(state, deps),
                lambda state: decision_output_b(state, deps),
                lambda state: safety_node(state, deps),
                lambda state: writeback_node(state, deps),
            ]
        )

    workflow = StateGraph(GraphState)
    workflow.add_node("load_context", lambda state: load_context(state, deps))
    workflow.add_node("perception_stub", lambda state: perception_stub(state, deps))
    workflow.add_node("fusion_node", lambda state: fusion_node(state, deps))
    workflow.add_node("reassessment_node", lambda state: reassessment_node(state, deps))
    workflow.add_node("retrieval_node", lambda state: retrieval_node(state, deps))
    workflow.add_node("decision_output_a", lambda state: decision_output_a(state, deps))
    workflow.add_node("decision_output_b", lambda state: decision_output_b(state, deps))
    workflow.add_node("safety_node", lambda state: safety_node(state, deps))
    workflow.add_node("writeback_node", lambda state: writeback_node(state, deps))

    workflow.add_edge(START, "load_context")
    workflow.add_edge("load_context", "perception_stub")
    workflow.add_edge("perception_stub", "fusion_node")
    workflow.add_edge("fusion_node", "reassessment_node")
    workflow.add_edge("reassessment_node", "retrieval_node")
    workflow.add_edge("retrieval_node", "decision_output_a")
    workflow.add_edge("decision_output_a", "decision_output_b")
    workflow.add_edge("decision_output_b", "safety_node")
    workflow.add_edge("safety_node", "writeback_node")
    workflow.add_edge("writeback_node", END)
    return workflow.compile()
