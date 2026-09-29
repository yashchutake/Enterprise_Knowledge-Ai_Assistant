"""
Observability for the Enterprise Knowledge Assistant.

Satisfies the "Observability" requirement two ways:
  1. If LANGCHAIN_TRACING_V2=true and LANGCHAIN_API_KEY is set, LangSmith
     tracing is enabled automatically -> every LangGraph node run is visible
     as a trace in the LangSmith UI.
  2. Regardless of LangSmith, every node logs a structured JSON line (start /
     end / routing decision) to both the console and logs/execution_trace.log,
     so the full node-by-node execution path is always visible even without
     LangSmith.
"""
import json
import logging
import os

from . import config

os.makedirs(config.LOG_DIR, exist_ok=True)


def _configure_langsmith() -> bool:
    if config.LANGCHAIN_TRACING_V2.lower() == "true" and config.LANGCHAIN_API_KEY:
        os.environ["LANGCHAIN_TRACING_V2"] = "true"
        os.environ["LANGCHAIN_API_KEY"] = config.LANGCHAIN_API_KEY
        os.environ["LANGCHAIN_PROJECT"] = config.LANGCHAIN_PROJECT
        return True
    os.environ["LANGCHAIN_TRACING_V2"] = "false"
    return False


def _setup_logging() -> logging.Logger:
    logger = logging.getLogger("eka")
    logger.setLevel(logging.INFO)
    logger.propagate = False

    if logger.handlers:
        return logger

    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    log_file = os.path.join(config.LOG_DIR, "execution_trace.log")
    file_handler = logging.FileHandler(log_file, mode="a", encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger


langsmith_enabled = _configure_langsmith()
logger = _setup_logging()
logger.info(f"Observability initialized | LangSmith tracing enabled: {langsmith_enabled}")


def log_node_event(node_name: str, event: str, data: dict | None = None) -> None:
    """Log a structured event for a LangGraph node (start/end/route/error)."""
    payload = {"node": node_name, "event": event}
    if data:
        safe_data = {}
        for key, value in data.items():
            if isinstance(value, str) and len(value) > 300:
                safe_data[key] = value[:300] + "...[truncated]"
            else:
                safe_data[key] = value
        payload["data"] = safe_data
    logger.info(json.dumps(payload, default=str, ensure_ascii=False))
