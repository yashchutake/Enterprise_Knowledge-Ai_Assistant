# Enterprise Knowledge Assistant

**LangGraph + RAG + RAGAS + MCP** — an AI-powered assistant that retrieves policy
information from an internal knowledge base, generates a grounded answer, evaluates
the quality of that answer with RAGAS, and integrates with an external system through
the Model Context Protocol (MCP).

Built for the **L3 Agentic AI Project Submission**. LLM + embeddings provider: **Google Gemini**.

---

## 1. Architecture Overview

```
                         ┌─────────────────────────────────────────────┐
                         │                LangGraph                    │
                         │                                             │
   User question         │   ┌───────────┐     ┌───────────┐          │
   ───────────────────▶ │   │ Retriever │────▶│ Response  │          │
                         │   │  Agent    │     │  Agent    │          │
                         │   └─────┬─────┘     └─────┬─────┘          │
                         │         │ ▲                │                │
                         │         │ │ retry (low     ▼                │
                         │         │ │ faithfulness)  ┌───────────┐    │
                         │         │ └────────────────│ Evaluator │    │
                         │         │                  │  Agent    │    │
                         │         │                  └─────┬─────┘    │
                         │         │                        │ finish   │
                         │         │                        ▼          │
                         │         │                  ┌───────────┐    │
                         │         │                  │ MCP Log   │    │
                         │         │                  │  Agent    │──▶ END
                         │         │                  └─────┬─────┘    │
                         └─────────┼────────────────────────┼──────────┘
                                   │                         │
                     ┌─────────────┴───────────┐   ┌─────────┴───────────┐
                     │   Chroma Vector Store    │   │  MCP Filesystem     │
                     │  (knowledge_base/*.md,   │   │  Server (stdio)     │
                     │   Gemini embeddings)     │   │  data/mcp_workspace │
                     └──────────────────────────┘   └──────────────────────┘
```

* **Retriever Agent** — semantic search over a Chroma vector store built from the
  markdown knowledge base, **plus** a live call to the MCP Filesystem Server to pull
  a supplementary FAQ document that lives outside the vector store.
* **Response Agent** — calls Gemini (`gemini-2.0-flash` by default) to generate an
  answer grounded only in the retrieved context.
* **Evaluator Agent** — scores the answer with RAGAS (**Faithfulness**,
  **Answer Relevancy**) and decides, via a conditional edge, whether to loop back to
  the Retriever Agent (bonus) if faithfulness is below a configurable threshold.
* **MCP Log Agent** — writes the question, answer, and scores back to
  `data/mcp_workspace/query_log.md` through the same MCP Filesystem Server, simulating
  an external ticket/audit log.

---

## 2. Setup Instructions

### Prerequisites
- Python 3.10+
- Node.js 18+ (needed to run the MCP Filesystem Server via `npx`)
- A Gemini API key from [Google AI Studio](https://aistudio.google.com/app/apikey)

### Steps
```bash
# 1. Unzip and enter the project
cd enterprise-knowledge-assistant

# 2. Create and activate a virtual environment
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure environment variables
cp .env.example .env
# Edit .env and set GOOGLE_API_KEY=your_gemini_api_key_here

# 5. (Optional but recommended) Pre-warm the MCP Filesystem Server package
npx -y @modelcontextprotocol/server-filesystem --help
```

The MCP Filesystem Server is launched automatically on demand via `npx`
(`config.MCP_FILESYSTEM_COMMAND` / `MCP_FILESYSTEM_ARGS` in `.env`) — no separate
process needs to be started manually.

The first run will automatically build the Chroma vector store from
`data/knowledge_base/*.md`. You can also build it explicitly:
```bash
python scripts/ingest.py
```

---

## 3. Execution Steps

### A. Command-line
```bash
python -m src.main "How many casual leaves am I entitled to per year?"
```
Sample output:
```
QUESTION: How many casual leaves am I entitled to per year?

ANSWER:
You are entitled to 12 casual leaves per calendar year, accrued at 1 day per
month. Unused casual leave lapses at the end of the calendar year.

SOURCES:
 - data/knowledge_base/leave_policy.md
 - MCP:filesystem:faq_extra.md

RAGAS EVALUATION
  Faithfulness:       0.92
  Answer Relevancy:   0.88
  Retrieval attempts: 1
```

### B. Streamlit UI (recommended for screenshots)
```bash
streamlit run app.py
```
Open the URL Streamlit prints (usually `http://localhost:8501`), type a question,
click **Run Assistant**, and see the answer, RAGAS scores/interpretation, retrieved
sources, and the node-by-node execution log (sidebar) all in one place.

### C. Smoke test (no API key required)
```bash
python tests/test_smoke.py
```
Verifies document loading, chunking, and that the LangGraph pipeline compiles.

### D. Bonus: offline Context Precision / Context Recall
```bash
python -m src.evaluation.offline_eval
```
Runs the retriever + response pipeline against the 5 labeled questions in
`data/eval_dataset.json` and reports Context Precision / Context Recall (requires
ground-truth answers, so it is run separately from the live per-query evaluation).

### Sample questions to try
- "How many casual leaves am I entitled to per year?"
- "What is the response time for a P1 critical IT ticket?"
- "How soon must I report a suspected security incident?"
- "What is the meal expense limit per person when meeting a client?"
- "Can I work remotely more than 2 days a week?" (answer will pull the MCP-only FAQ)

---

## 4. RAG Design

| Aspect | Detail |
|---|---|
| Document source | 4 markdown policy documents in `data/knowledge_base/` (leave, IT support, security, expense policies) |
| Chunking strategy | `RecursiveCharacterTextSplitter`, `chunk_size=800`, `chunk_overlap=120`, splitting on markdown headers first (`## `, `### `) then paragraphs/sentences |
| Embedding model | Gemini `models/text-embedding-004` via `langchain-google-genai` |
| Vector database | Chroma (local, persisted to `data/vector_store/`) |
| Retrieval | Top-`k=4` similarity search per query (configurable via `RETRIEVER_TOP_K`) |
| External augmentation | Retriever Agent also pulls `data/mcp_workspace/faq_extra.md` live through the MCP Filesystem Server, so answers can combine indexed + non-indexed context |

---

## 5. LangGraph Design

| Node | Responsibility |
|---|---|
| `retriever` | Chroma similarity search + MCP filesystem read (supplementary FAQ) |
| `responder` | Gemini chat completion, grounded strictly in retrieved context |
| `evaluator` | RAGAS scoring (Faithfulness, Answer Relevancy) + routing decision |
| `mcp_logger` | Writes Q&A + scores to the MCP Filesystem Server's `query_log.md` |

**Graph flow:**
```
START -> retriever -> responder -> evaluator --(finish)--> mcp_logger -> END
                                       |
                                       +--(retry, bonus)--> retriever
```
The conditional edge (`retry` vs `finish`) is implemented in
`src/agents/evaluator_agent.py::route_after_evaluation`: if `faithfulness` is below
`FAITHFULNESS_THRESHOLD` (default `0.5`) and fewer than `MAX_RETRIEVAL_ATTEMPTS`
(default `2`) have been made, the graph loops back to `retriever`; otherwise it
proceeds to `mcp_logger` and ends. State is a `TypedDict` (`AssistantState` in
`src/graph.py`) carrying `query`, `contexts`, `sources`, `answer`, `faithfulness`,
`answer_relevancy`, and `retrieval_attempts` across nodes.

---

## 6. MCP Integration

- **Server used:** official [`@modelcontextprotocol/server-filesystem`](https://github.com/modelcontextprotocol/servers/tree/main/src/filesystem), launched over stdio via `npx`, restricted to `data/mcp_workspace/` as its only allowed root.
- **Tools exposed/used:** `read_file`, `write_file`, `list_directory`.
- **Use case implemented:**
  1. **Retriever Agent** calls `read_file` on `faq_extra.md` — a document that
     intentionally lives outside the Chroma vector store — to demonstrate pulling
     live context from an external system on every query.
  2. **MCP Log Agent** calls `read_file` + `write_file` on `query_log.md` to append
     every question, generated answer, RAGAS scores, and retrieval-attempt count,
     simulating writing back to an external audit/ticketing system.
- Client implementation: `src/mcp/mcp_client.py`, using the official `mcp` Python SDK
  (`ClientSession`, `stdio_client`).

---

## 7. RAGAS Evaluation

| Metric | Type | Where computed |
|---|---|---|
| **Faithfulness** (mandatory) | Live, per query | `evaluator` node, every run |
| **Answer Relevancy** (mandatory) | Live, per query | `evaluator` node, every run |
| **Context Precision** (bonus) | Offline, needs ground truth | `python -m src.evaluation.offline_eval` against `data/eval_dataset.json` |
| **Context Recall** (bonus) | Offline, needs ground truth | same as above |

**Interpretation guide** (also shown live in the Streamlit app):
- **Faithfulness ≥ 0.7** — the answer is well grounded in retrieved context.
- **Faithfulness 0.5–0.7** — some claims may not be fully backed by context.
- **Faithfulness < 0.5** — likely unsupported claims; triggers the bonus retry edge.
- **Answer Relevancy ≥ 0.7** — the answer directly addresses the question.
- **Answer Relevancy < 0.5** — the answer may be off-topic.

Both mandatory metrics are computed reference-free (no ground truth needed) using
`ragas.metrics.Faithfulness` and `ragas.metrics.ResponseRelevancy`, backed by Gemini
as both the judge LLM and the embedding model (`src/evaluation/ragas_eval.py`).

---

## 8. Observability

Every node logs a structured JSON line (`start` / `end` / `route`) to both the
console and `logs/execution_trace.log`, so the full node-by-node execution path
(`retriever → responder → evaluator → [retry|mcp_logger]`) is visible for every run
even without LangSmith. The Streamlit sidebar also tails this log live.

To additionally enable **LangSmith** tracing, set in `.env`:
```
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=your_langsmith_key
LANGCHAIN_PROJECT=enterprise-knowledge-assistant
```

---

## 9. Screenshots to Capture for Submission

This zip contains the complete, runnable source code. Since screenshots must show
your own live run (with your own Gemini key), capture the following after following
the setup steps above:

1. **Application startup** — terminal output of `streamlit run app.py` or
   `python -m src.main "..."` starting successfully.
2. **Graph execution trace, node-by-node** — the Streamlit sidebar "Execution Log",
   or the contents of `logs/execution_trace.log` after a run (shows `retriever` →
   `responder` → `evaluator` → `mcp_logger` firing in order, each start/end event).
3. **RAGAS evaluation results** — the metrics panel in the Streamlit app, or the
   `RAGAS EVALUATION` block printed by the CLI.
4. **Final output** — the generated answer shown in the Streamlit app or terminal.
5. *(Optional, MCP evidence)* — contents of `data/mcp_workspace/query_log.md` after
   a few runs, showing entries written through the MCP Filesystem Server.

---

## 10. Project Structure

```
enterprise-knowledge-assistant/
├── app.py                        # Streamlit UI
├── requirements.txt
├── .env.example
├── data/
│   ├── knowledge_base/           # RAG source documents (4 policy .md files)
│   ├── mcp_workspace/            # Files served/written via MCP Filesystem Server
│   ├── eval_dataset.json         # Labeled Q&A for bonus RAGAS offline eval
│   └── vector_store/             # Persisted Chroma DB (built on first run)
├── src/
│   ├── config.py                 # Env/config loading
│   ├── observability.py          # Structured logging + LangSmith toggle
│   ├── graph.py                  # LangGraph wiring
│   ├── main.py                   # CLI entrypoint
│   ├── rag/
│   │   ├── loader.py              # Load + chunk documents
│   │   └── vectorstore.py         # Gemini embeddings + Chroma
│   ├── mcp/
│   │   └── mcp_client.py          # MCP Filesystem Server client
│   ├── agents/
│   │   ├── retriever_agent.py     # Node 1
│   │   ├── response_agent.py      # Node 2
│   │   ├── evaluator_agent.py     # Node 3 (+ conditional routing)
│   │   └── mcp_log_agent.py       # Node 4
│   └── evaluation/
│       ├── ragas_eval.py          # Faithfulness + Answer Relevancy (live)
│       └── offline_eval.py        # Context Precision + Context Recall (bonus)
├── scripts/
│   └── ingest.py                  # Rebuild the vector store
├── tests/
│   └── test_smoke.py              # No-API-key smoke test
└── logs/
    └── execution_trace.log        # Generated on first run
```

---

## 11. Requirements Coverage Checklist

- [x] LangGraph with ≥3 agent nodes, linear flow Retrieve → Respond → Evaluate → End
- [x] Bonus: conditional edge re-triggering retrieval on low RAGAS score
- [x] RAG: document loading, chunking, embeddings, vector DB, retrieval, generation
- [x] RAGAS: Faithfulness + Answer Relevancy (mandatory); Context Precision + Context Recall (bonus)
- [x] MCP Server integration (Filesystem), actively used by two nodes
- [x] Observability: structured per-node logs + optional LangSmith tracing
