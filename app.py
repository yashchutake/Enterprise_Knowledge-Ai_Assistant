"""
Streamlit UI for the Enterprise Knowledge Assistant.

Run with:
    streamlit run app.py

This gives a friendly surface to capture the required submission screenshots:
  - Application startup           (this page loading)
  - Graph execution trace          (sidebar "Execution Log")
  - RAGAS evaluation results       (metrics + interpretation panel)
  - Final output                   (Answer panel)
"""
import asyncio

import streamlit as st

from src import config
from src.graph import build_graph

st.set_page_config(page_title="Enterprise Knowledge Assistant", page_icon="🧠", layout="wide")

st.title("🧠 Enterprise Knowledge Assistant")
st.caption("LangGraph  •  RAG (Chroma + Gemini)  •  RAGAS  •  MCP Filesystem Server")

if not config.GOOGLE_API_KEY:
    st.error(
        "GOOGLE_API_KEY is not set. Copy `.env.example` to `.env`, add your Gemini API key, "
        "and restart the app."
    )

query = st.text_input(
    "Ask a question about company policy:",
    placeholder="How many casual leaves am I entitled to per year?",
)
run = st.button("Run Assistant", type="primary")

if run and query:
    with st.spinner("Running LangGraph pipeline: Retrieve → Respond → Evaluate → Log..."):
        graph = build_graph()
        final_state = asyncio.run(graph.ainvoke({"query": query, "retrieval_attempts": 0}))

    st.subheader("Answer")
    st.write(final_state.get("answer"))

    col1, col2, col3 = st.columns(3)
    col1.metric("Faithfulness", final_state.get("faithfulness"))
    col2.metric("Answer Relevancy", final_state.get("answer_relevancy"))
    col3.metric("Retrieval Attempts", final_state.get("retrieval_attempts"))

    with st.expander("Retrieved Context / Sources", expanded=False):
        sources = final_state.get("sources", [])
        contexts = final_state.get("contexts", [])
        for source, context in zip(sources, contexts):
            st.markdown(f"**Source:** `{source}`")
            st.text(context[:600])
            st.divider()

    with st.expander("RAGAS Interpretation", expanded=True):
        faithfulness = final_state.get("faithfulness") or 0
        relevancy = final_state.get("answer_relevancy") or 0

        if faithfulness >= 0.7:
            f_note = "High — the answer stays well grounded in the retrieved context."
        elif faithfulness >= 0.5:
            f_note = "Moderate — some claims may not be fully backed by the context."
        else:
            f_note = "Low — the answer may contain unsupported claims; a retry was likely triggered."

        if relevancy >= 0.7:
            r_note = "High — the answer directly addresses the question asked."
        elif relevancy >= 0.5:
            r_note = "Moderate — the answer only partially addresses the question."
        else:
            r_note = "Low — the answer may be off-topic relative to the question."

        st.markdown(f"- **Faithfulness ({faithfulness}):** {f_note}")
        st.markdown(f"- **Answer Relevancy ({relevancy}):** {r_note}")

    st.info(
        "This question, answer, and evaluation score were written via the MCP Filesystem "
        "Server to `data/mcp_workspace/query_log.md`."
    )
else:
    st.write(
        "Enter a question and click **Run Assistant** to run the full LangGraph pipeline: "
        "Retrieve → Respond → Evaluate → Log."
    )

st.sidebar.header("Execution Log (node-by-node trace)")
try:
    with open("logs/execution_trace.log", "r", encoding="utf-8") as f:
        lines = f.readlines()[-60:]
    st.sidebar.code("".join(lines) if lines else "No log entries yet.", language="json")
except FileNotFoundError:
    st.sidebar.write("No logs yet. Run a query first.")
