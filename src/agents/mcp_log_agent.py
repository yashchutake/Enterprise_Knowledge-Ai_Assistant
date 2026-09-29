"""
MCP Log Agent (LangGraph node #4)

Responsibility: actively use the MCP Filesystem Server to persist every
question, answer, and RAGAS score to an external log file
(data/mcp_workspace/query_log.md), simulating writing back to an external
ticketing / audit system through MCP.
"""
from datetime import datetime, timezone

from ..mcp.mcp_client import mcp_read_file, mcp_write_file
from ..observability import log_node_event

LOG_FILENAME = "query_log.md"


async def mcp_log_node(state: dict) -> dict:
    log_node_event("MCPLogAgent", "start", {"query": state.get("query")})

    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "query": state.get("query"),
        "answer": state.get("answer"),
        "faithfulness": state.get("faithfulness"),
        "answer_relevancy": state.get("answer_relevancy"),
        "retrieval_attempts": state.get("retrieval_attempts"),
    }

    existing = await mcp_read_file(LOG_FILENAME)
    if not existing or existing.startswith("[MCP read_file error"):
        existing = "# Enterprise Knowledge Assistant - Query Log\n\n"

    new_block = (
        f"## {entry['timestamp']}\n"
        f"**Question:** {entry['query']}\n\n"
        f"**Answer:** {entry['answer']}\n\n"
        f"**Faithfulness:** {entry['faithfulness']} | "
        f"**Answer Relevancy:** {entry['answer_relevancy']} | "
        f"**Retrieval attempts:** {entry['retrieval_attempts']}\n\n---\n\n"
    )

    updated_content = existing + new_block
    success = await mcp_write_file(LOG_FILENAME, updated_content)

    log_node_event("MCPLogAgent", "end", {"mcp_write_success": success})

    return {}
