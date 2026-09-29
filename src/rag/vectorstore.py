"""
Embedding generation and vector storage (Chroma) for the RAG pipeline.
"""
import os

from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings

from .. import config
from .loader import load_documents, chunk_documents

COLLECTION_NAME = "enterprise_knowledge_base"

_embeddings = None


def get_embeddings() -> GoogleGenerativeAIEmbeddings:
    global _embeddings
    if _embeddings is None:
        _embeddings = GoogleGenerativeAIEmbeddings(
            model=config.GEMINI_EMBEDDING_MODEL,
            google_api_key=config.GOOGLE_API_KEY,
        )
    return _embeddings


def build_vector_store(persist: bool = True) -> Chroma:
    """Load, chunk, embed and persist the knowledge base documents into Chroma."""
    documents = load_documents()
    chunks = chunk_documents(documents)
    embeddings = get_embeddings()

    persist_directory = config.VECTOR_STORE_DIR if persist else None
    if persist:
        os.makedirs(config.VECTOR_STORE_DIR, exist_ok=True)

    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=COLLECTION_NAME,
        persist_directory=persist_directory,
    )
    return vector_store


def load_vector_store() -> Chroma:
    """Load an existing persisted Chroma store, building it on first run."""
    embeddings = get_embeddings()
    store_is_empty = not os.path.isdir(config.VECTOR_STORE_DIR) or not os.listdir(
        config.VECTOR_STORE_DIR
    )
    if store_is_empty:
        return build_vector_store()

    return Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=config.VECTOR_STORE_DIR,
    )


def get_retriever(k: int | None = None):
    k = k or config.RETRIEVER_TOP_K
    store = load_vector_store()
    return store.as_retriever(search_kwargs={"k": k})
