from __future__ import annotations

import os

from dotenv import load_dotenv

load_dotenv()

_paper_similarity_threshold = float(
    os.getenv("TXSHIELD_SIMILARITY_THRESHOLD", "0.85")
)
if not 0.0 <= _paper_similarity_threshold <= 1.0:
    raise ValueError("TXSHIELD_SIMILARITY_THRESHOLD must be between 0 and 1")

# Both forms refer to Euclidean L2 distance, NOT Chroma's squared L2 score.
# The legacy distance variable takes precedence when explicitly configured.
_legacy_distance_threshold = os.getenv("TXSHIELD_DISTANCE_THRESHOLD")
_retrieval_distance_cutoff = (
    float(_legacy_distance_threshold)
    if _legacy_distance_threshold is not None
    else 1.0 - _paper_similarity_threshold
)
if not 0.0 <= _retrieval_distance_cutoff <= 1.0:
    raise ValueError("TXSHIELD_DISTANCE_THRESHOLD must be between 0 and 1 (L2 distance)")

CONFIG = {
    "PERSIST_DIR": os.getenv("TXSHIELD_VECTOR_DIR", "artifacts/vector_store"),
    "CSV_PATH": os.getenv("TXSHIELD_CSV_PATH") or "artifacts/knowledge/knowledge.csv",
    "HTTP_PROXY": os.getenv("HTTP_PROXY", ""),
    "HTTPS_PROXY": os.getenv("HTTPS_PROXY", ""),
    "JINA_API_KEY": os.getenv("JINA_API_KEY", ""),
    "EMBEDDING_MODEL": os.getenv("TXSHIELD_EMBEDDING_MODEL", "jina-embeddings-v3"),
    "PAPER_SIMILARITY_THRESHOLD": _paper_similarity_threshold,
    "L2_DISTANCE_THRESHOLD": _retrieval_distance_cutoff,
    # Compatibility alias for code that previously read this internal key.
    "SIMILARITY_THRESHOLD": _retrieval_distance_cutoff,
}
