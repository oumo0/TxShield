from __future__ import annotations

from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.embeddings import Embeddings


def require_l2_collection(vector_store: Chroma) -> None:
    """Read old/new Chroma metric settings without relabeling the collection."""
    collection = getattr(vector_store, "_collection", None)
    if collection is None:
        raise ValueError("Cannot verify the Chroma collection's distance metric")
    metadata = collection.metadata or {}
    metric = metadata.get("hnsw:space")
    configuration = getattr(collection, "configuration", None)
    if isinstance(configuration, dict):
        for index_name in ("hnsw", "spann"):
            index = configuration.get(index_name)
            if not isinstance(index, dict) or not index.get("space"):
                continue
            configured_metric = index["space"]
            if metric is not None and metric != configured_metric:
                raise ValueError("Conflicting Chroma collection distance metrics")
            metric = configured_metric
    metric = metric or "l2"
    if metric != "l2":
        raise ValueError(
            f"The paper pipeline requires a Chroma l2 collection; found {metric!r}. "
            "Use an L2 index rather than changing or relabeling the metric."
        )


def load_or_create_vector_store(persist_dir: str, embedding: Embeddings) -> Chroma:
    """Open the existing nonempty L2 knowledge collection."""
    if not Path(persist_dir).is_dir():
        raise FileNotFoundError("Knowledge database directory does not exist")
    store = Chroma(
        persist_directory=persist_dir,
        embedding_function=embedding,
        create_collection_if_not_exists=False,
    )
    require_l2_collection(store)
    if store._collection.count() == 0:
        raise ValueError("Knowledge collection is empty")
    return store
