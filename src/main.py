"""
CLI entrypoint.

Usage:
    python -m src.main "How many casual leaves am I entitled to per year?"
    python -m src.main            # interactive prompt
"""
import asyncio
import sys

from .graph import build_graph
from .observability import logger


async def run_query(query: str) -> dict:
    graph = build_graph()
    logger.info(f"=== New run: {query} ===")
    initial_state = {"query": query, "retrieval_attempts": 0}
    final_state = await graph.ainvoke(initial_state)
    return final_state


def print_result(state: dict) -> None:
    print("\n" + "=" * 72)
    print("QUESTION:", state.get("query"))
    print("=" * 72)
    print("\nANSWER:\n" + str(state.get("answer")))
    print("\nSOURCES:")
    for source in state.get("sources", []):
        print(" -", source)
    print("\nRAGAS EVALUATION")
    print(f"  Faithfulness:       {state.get('faithfulness')}")
    print(f"  Answer Relevancy:   {state.get('answer_relevancy')}")
    print(f"  Retrieval attempts: {state.get('retrieval_attempts')}")
    print("=" * 72 + "\n")
    print("Full node-by-node execution trace: logs/execution_trace.log")
    print("Query log written via MCP: data/mcp_workspace/query_log.md\n")


def main() -> None:
    if len(sys.argv) > 1:
        query = " ".join(sys.argv[1:])
    else:
        query = input("Ask the Enterprise Knowledge Assistant: ")

    final_state = asyncio.run(run_query(query))
    print_result(final_state)


if __name__ == "__main__":
    main()
