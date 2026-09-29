"""
Response Agent (LangGraph node #2)

Responsibility: generate a grounded answer to the user's question using only
the context assembled by the Retriever Agent, via the Gemini chat model.
"""
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI

from .. import config
from ..observability import log_node_event

SYSTEM_PROMPT = (
    "You are an Enterprise Knowledge Assistant. Answer the employee's question "
    "using ONLY the information in the provided context. If the context does not "
    "contain enough information to answer confidently, say so explicitly instead "
    "of guessing. Be concise, and reference the relevant policy area when helpful."
)


def _get_llm() -> ChatGoogleGenerativeAI:
    # Not cached: ChatGoogleGenerativeAI's async client binds to whichever event
    # loop is running at first use, which breaks across Streamlit reruns.
    return ChatGoogleGenerativeAI(
        model=config.GEMINI_CHAT_MODEL,
        google_api_key=config.GOOGLE_API_KEY,
        temperature=0.2,
    )

async def response_node(state: dict) -> dict:
    query = state["query"]
    contexts = state.get("contexts", [])

    log_node_event("ResponseAgent", "start", {"query": query, "num_contexts": len(contexts)})

    context_block = "\n\n---\n\n".join(contexts) if contexts else "No context was retrieved."
    user_prompt = f"Context:\n{context_block}\n\nQuestion: {query}\n\nAnswer:"

    llm = _get_llm()
    result = await llm.ainvoke(
        [SystemMessage(content=SYSTEM_PROMPT), HumanMessage(content=user_prompt)]
    )
    answer = result.content

    log_node_event("ResponseAgent", "end", {"answer": answer})

    return {"answer": answer}
