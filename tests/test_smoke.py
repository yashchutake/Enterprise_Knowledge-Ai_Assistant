"""
Smoke tests that do NOT require a Gemini API key or network access -
they only exercise document loading and chunking.

Run with:
    python tests/test_smoke.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.rag.loader import chunk_documents, load_documents  # noqa: E402


def test_load_documents():
    docs = load_documents()
    assert len(docs) >= 4, "Expected at least the 4 seeded knowledge base files"
    return docs


def test_chunk_documents(docs):
    chunks = chunk_documents(docs)
    assert len(chunks) >= len(docs), "Chunking should not reduce the number of segments"
    for chunk in chunks:
        assert chunk.page_content.strip(), "Chunks should not be empty"
    return chunks


def test_graph_builds():
    """Verify the LangGraph pipeline compiles without errors (no API calls made)."""
    from src.graph import build_graph

    graph = build_graph()
    assert graph is not None


if __name__ == "__main__":
    docs = test_load_documents()
    print(f"[PASS] Loaded {len(docs)} documents")

    chunks = test_chunk_documents(docs)
    print(f"[PASS] Produced {len(chunks)} chunks")

    test_graph_builds()
    print("[PASS] LangGraph pipeline compiles")

    print("\nAll smoke tests passed.")
