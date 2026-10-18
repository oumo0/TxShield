"""LangChain callback hooks for the analyzer."""

from langchain_core.callbacks import BaseCallbackHandler


class SimplePromptMonitor(BaseCallbackHandler):
    """Compatibility callback reserved for non-sensitive runtime metrics."""
