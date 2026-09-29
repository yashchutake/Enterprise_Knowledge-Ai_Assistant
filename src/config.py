"""
Central configuration for the Enterprise Knowledge Assistant.
Loads settings from environment variables (via a .env file if present).
"""
import os

from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Gemini / Google Generative AI
# ---------------------------------------------------------------------------
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")
GEMINI_CHAT_MODEL = os.getenv("GEMINI_CHAT_MODEL", "gemini-2.0-flash")
GEMINI_EMBEDDING_MODEL = os.getenv("GEMINI_EMBEDDING_MODEL", "models/text-embedding-004")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
KNOWLEDGE_BASE_DIR = os.path.join(BASE_DIR, "data", "knowledge_base")
MCP_WORKSPACE_DIR = os.path.join(BASE_DIR, "data", "mcp_workspace")
VECTOR_STORE_DIR = os.path.join(BASE_DIR, "data", "vector_store")
LOG_DIR = os.path.join(BASE_DIR, "logs")

# ---------------------------------------------------------------------------
# RAG settings
# ---------------------------------------------------------------------------
RETRIEVER_TOP_K = int(os.getenv("RETRIEVER_TOP_K", "4"))
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "800"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "120"))

# ---------------------------------------------------------------------------
# Evaluation / routing
# ---------------------------------------------------------------------------
FAITHFULNESS_THRESHOLD = float(os.getenv("FAITHFULNESS_THRESHOLD", "0.5"))
MAX_RETRIEVAL_ATTEMPTS = int(os.getenv("MAX_RETRIEVAL_ATTEMPTS", "2"))

# ---------------------------------------------------------------------------
# Observability (LangSmith is optional)
# ---------------------------------------------------------------------------
LANGCHAIN_TRACING_V2 = os.getenv("LANGCHAIN_TRACING_V2", "false")
LANGCHAIN_API_KEY = os.getenv("LANGCHAIN_API_KEY", "")
LANGCHAIN_PROJECT = os.getenv("LANGCHAIN_PROJECT", "enterprise-knowledge-assistant")

# ---------------------------------------------------------------------------
# MCP Filesystem Server
# ---------------------------------------------------------------------------
MCP_FILESYSTEM_COMMAND = os.getenv("MCP_FILESYSTEM_COMMAND", "npx")
MCP_FILESYSTEM_ARGS = os.getenv(
    "MCP_FILESYSTEM_ARGS", "-y @modelcontextprotocol/server-filesystem"
).split()


def _warn_if_missing_key():
    if not GOOGLE_API_KEY:
        print(
            "[WARN] GOOGLE_API_KEY is not set. Create a .env file (see .env.example) "
            "and set your Gemini API key before running the application."
        )


_warn_if_missing_key()
