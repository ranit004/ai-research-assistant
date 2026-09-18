"""Assemble the LangGraph StateGraph for multi-step research.

Graph topology:
    START
    → understand_query
    → rewrite_query
    → decompose_query
    → retrieve_evidence
    → evaluate_evidence
    → [conditional]
        insufficient & iterations < max  → refine_query → retrieve_evidence
        sufficient OR iterations >= max  → synthesize_answer
    → verify_answer
    → END
"""

from langgraph.graph import END, START, StateGraph

from research_assistant.agent.state import ResearchState
from research_assistant.config import settings


def _route_after_evaluate(state: ResearchState) -> str:
    """Decide whether to refine or synthesise.

    Hard ceiling on iterations prevents an infinite loop under any condition.
    """
    if state.get("evidence_sufficient"):
        return "synthesize_answer"
    if state.get("iteration_count", 0) >= settings.max_research_iterations:
        return "synthesize_answer"   # synthesise will return unsupported answer
    return "refine_query"


def build_graph() -> StateGraph:
    """Return a compiled LangGraph research workflow."""
    import research_assistant.agent.nodes as _n

    graph = StateGraph(ResearchState)

    # Each wrapper re-reads the module attribute at call-time so tests can
    # monkeypatch _n.<name> and have the patched version run through the graph.
    graph.add_node("understand_query",  lambda s: _n.understand_query(s))
    graph.add_node("rewrite_query",     lambda s: _n.rewrite_query(s))
    graph.add_node("decompose_query",   lambda s: _n.decompose_query(s))
    graph.add_node("retrieve_evidence", lambda s: _n.retrieve_evidence(s))
    graph.add_node("evaluate_evidence", lambda s: _n.evaluate_evidence(s))
    graph.add_node("refine_query",      lambda s: _n.refine_query(s))
    graph.add_node("synthesize_answer", lambda s: _n.synthesize_answer(s))
    graph.add_node("verify_answer",     lambda s: _n.verify_answer(s))

    graph.add_edge(START, "understand_query")
    graph.add_edge("understand_query", "rewrite_query")
    graph.add_edge("rewrite_query", "decompose_query")
    graph.add_edge("decompose_query", "retrieve_evidence")
    graph.add_edge("retrieve_evidence", "evaluate_evidence")

    graph.add_conditional_edges(
        "evaluate_evidence",
        _route_after_evaluate,
        {
            "refine_query": "refine_query",
            "synthesize_answer": "synthesize_answer",
        },
    )

    graph.add_edge("refine_query", "retrieve_evidence")
    graph.add_edge("synthesize_answer", "verify_answer")
    graph.add_edge("verify_answer", END)

    return graph.compile()


# Module-level singleton — compiled once on import.
research_graph = build_graph()
