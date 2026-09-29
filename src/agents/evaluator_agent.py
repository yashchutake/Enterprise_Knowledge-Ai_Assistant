"""
Evaluator Agent (LangGraph node #3)

Responsibilities:
  1. Score the generated answer with RAGAS (Faithfulness, Answer Relevancy).
  2. (Bonus) Decide, via a conditional edge, whether faithfulness is too low
     and retrieval should be retried, or whether the pipeline should finish.
"""
from .. import config
from ..evaluation.ragas_eval import evaluate_response
from ..observability import log_node_event


async def evaluator_node(state: dict) -> dict:
    query = state["query"]
    answer = state.get("answer", "")
    contexts = state.get("contexts", [])

    log_node_event("EvaluatorAgent", "start", {"query": query})

    scores = await evaluate_response(query, answer, contexts)

    log_node_event("EvaluatorAgent", "end", {"scores": scores})

    return {
        "faithfulness": scores.get("faithfulness"),
        "answer_relevancy": scores.get("answer_relevancy"),
    }


def route_after_evaluation(state: dict) -> str:
    """Conditional edge (bonus): retry retrieval once if faithfulness is low."""
    faithfulness = state.get("faithfulness")
    faithfulness = faithfulness if faithfulness is not None else 1.0  # don't loop on scoring errors
    attempts = state.get("retrieval_attempts", 0)

    if faithfulness < config.FAITHFULNESS_THRESHOLD and attempts < config.MAX_RETRIEVAL_ATTEMPTS:
        log_node_event(
            "EvaluatorAgent",
            "route",
            {"decision": "retry_retrieval", "faithfulness": faithfulness, "attempts": attempts},
        )
        return "retry"

    log_node_event(
        "EvaluatorAgent",
        "route",
        {"decision": "finish", "faithfulness": faithfulness, "attempts": attempts},
    )
    return "finish"
