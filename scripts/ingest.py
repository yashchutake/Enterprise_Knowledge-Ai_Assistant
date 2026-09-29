"""
Build (or rebuild) the Chroma vector store from the knowledge base documents.

Run with:
    python scripts/ingest.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.rag.vectorstore import build_vector_store  # noqa: E402


def main():
    print("Loading, chunking, and embedding documents from data/knowledge_base ...")
    build_vector_store()
    print("Vector store built and persisted at data/vector_store/")


if __name__ == "__main__":
    main()
