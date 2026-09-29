"""
Retriever Agent (LangGraph node #1)

Responsibilities:
  1. Perform semantic search over the Chroma vector store built from the
     enterprise knowledge base (RAG).
  2. Actively call the MCP Filesystem Server to pull in a supplementary
     document (faq_extra.md) that is NOT part of the vector store, simulating
     retrieval from a live external system.
"""
from ..mcp.mcp_client import mcp_read_file
from ..observability import log_node_event
from ..rag.vectorstore import get_retriever

_retriever = None


def _get_retriever():
    global _retriever
    if _retriever is None:
        _retriever = get_retriever()
    return _retriever


async def retriever_node(state: dict) -> dict:
    query = state["query"]
    attempts = state.get("retrieval_attempts", 0) + 1

    log_node_event("RetrieverAgent", "start", {"query": query, "attempt": attempts})

    retriever = _get_retriever()
    docs = await retriever.ainvoke(query)
    contexts = [doc.page_content for doc in docs]
    sources = [doc.metadata.get("source", "knowledge_base") for doc in docs]

    # Actively use the MCP Filesystem Server to fetch supplementary context.
    mcp_extra = await mcp_read_file("faq_extra.md")
    if mcp_extra:
        contexts.append(mcp_extra)
        sources.append("MCP:filesystem:faq_extra.md")

    log_node_event(
        "RetrieverAgent",
        "end",
        {"num_contexts": len(contexts), "sources": sources},
    )

    return {
        "contexts": contexts,
        "sources": sources,
        "retrieval_attempts": attempts,
    }
