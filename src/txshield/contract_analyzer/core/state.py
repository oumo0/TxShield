"""Per-analysis RAG routing state."""
from __future__ import annotations

from contextvars import ContextVar


class SkipRAGManager:
    """Keep RAG routing state isolated across concurrent analyses."""

    def __init__(self) -> None:
        self._skip_rag: ContextVar[bool] = ContextVar("txshield_skip_rag", default=False)

    def set_skip_rag(self, value: bool) -> None:
        self._skip_rag.set(bool(value))

    def get_skip_rag(self) -> bool:
        return self._skip_rag.get()

    def reset(self) -> None:
        self._skip_rag.set(False)


skip_rag_manager = SkipRAGManager()
