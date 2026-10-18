"""Chroma vector-store construction."""
from __future__ import annotations

from langchain_chroma import Chroma
from langchain_core.embeddings import Embeddings


def load_or_create_vector_store(persist_dir: str, embedding: Embeddings) -> Chroma:
    """Open the configured persistent Chroma collection."""
    return Chroma(persist_directory=persist_dir, embedding_function=embedding)
