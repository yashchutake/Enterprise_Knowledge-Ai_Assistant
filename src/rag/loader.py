"""
Load documents from the knowledge base directory and split them into chunks.
"""
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from .. import config


def load_documents(source_dir: str | None = None):
    """Load every markdown file from the knowledge base directory."""
    source_dir = source_dir or config.KNOWLEDGE_BASE_DIR
    loader = DirectoryLoader(
        source_dir,
        glob="**/*.md",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"},
        show_progress=False,
    )
    documents = loader.load()
    return documents


def chunk_documents(documents, chunk_size: int | None = None, chunk_overlap: int | None = None):
    """Split loaded documents into overlapping chunks for embedding."""
    chunk_size = chunk_size or config.CHUNK_SIZE
    chunk_overlap = chunk_overlap or config.CHUNK_OVERLAP

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n## ", "\n### ", "\n\n", "\n", " ", ""],
    )
    chunks = splitter.split_documents(documents)
    return chunks


def load_and_chunk(source_dir: str | None = None):
    documents = load_documents(source_dir)
    return chunk_documents(documents)
