"""
LangGraph definition for the Enterprise Knowledge Assistant.

Flow (mandatory, linear):
    START -> retriever -> responder -> evaluator -> END

Bonus conditional edge:
    evaluator -> retriever   (if RAGAS faithfulness is below threshold and
                               retries remain)
    evaluator -> mcp_logger -> END   (otherwise)
"""
from typing import List, Optional, TypedDict

from langgraph.graph import END, START, StateGraph

from .agents.evaluator_agent import evaluator_node, route_after_evaluation
from .agents.mcp_log_agent import mcp_log_node
from .agents.response_agent import response_node
from .agents.retriever_agent import retriever_node


class AssistantState(TypedDict, total=False):
    query: str
    contexts: List[str]
    sources: List[str]
    answer: str
    faithfulness: Optional[float]
    answer_relevancy: Optional[float]
    retrieval_attempts: int


def build_graph():
    workflow = StateGraph(AssistantState)

    workflow.add_node("retriever", retriever_node)
    workflow.add_node("responder", response_node)
    workflow.add_node("evaluator", evaluator_node)
    workflow.add_node("mcp_logger", mcp_log_node)

    workflow.add_edge(START, "retriever")
    workflow.add_edge("retriever", "responder")
    workflow.add_edge("responder", "evaluator")

    workflow.add_conditional_edges(
        "evaluator",
        route_after_evaluation,
        {
            "retry": "retriever",
            "finish": "mcp_logger",
        },
    )

    workflow.add_edge("mcp_logger", END)

    return workflow.compile()
