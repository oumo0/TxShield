from __future__ import annotations

import os
from typing import Any, TypedDict

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.runnables import Runnable

from txshield.contract_analyzer.core.message_manager import MessageManager


class ChatMessage(TypedDict):
    role: str
    content: str


_API_PROVIDERS = {
    "openai": ("OPENAI_API_KEY", "https://api.openai.com/v1", "gpt-4.1-mini"),
    "deepseek": (
        "DEEPSEEK_API_KEY",
        "https://api.deepseek.com/v1",
        "deepseek-v4-pro",
    ),
}


def _messages(payload: Any) -> list[ChatMessage]:
    """Build paper-aligned RAG or zero-shot messages for one request."""
    if isinstance(payload, str):
        return [{"role": "user", "content": payload}]
    if not isinstance(payload, dict):
        raise TypeError("LLM payload must be a dictionary or a string")
    system_input = str(payload.get("system_input", ""))
    query = str(payload.get("query", payload.get("input", "Analyze the transaction.")))
    reference = str(payload.get("reference_text", ""))
    return MessageManager.build_messages_with_prompt_template(
        query=query,
        system_input=system_input,
        reference_text=reference,
        skip_rag=bool(payload.get("skip_rag", False)),
        prompt_type=str(payload.get("prompt_type", "rag")),
        xgb_prob=payload.get("xgb_prob"),
        xgb_context_threshold=payload.get("xgb_context_threshold"),
    )


def _render_prompt(messages: list[ChatMessage]) -> str:
    """Render provider-neutral messages for text-generation pipelines."""
    return "\n\n".join(
        f"{message['role'].upper()}:\n{message['content']}"
        for message in messages
    )


def _langchain_messages(payload: Any) -> list[SystemMessage | HumanMessage]:
    """Convert provider-neutral dictionaries to LangChain chat messages."""
    messages: list[SystemMessage | HumanMessage] = []
    for message in _messages(payload):
        if message["role"] == "system":
            messages.append(SystemMessage(content=message["content"]))
        else:
            messages.append(HumanMessage(content=message["content"]))
    return messages


def _required_environment(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise ValueError(f"Missing {name}. Set it in your local .env file.")
    return value


class ChatBackend(Runnable):
    """Stable adapter for the original Runnable-based analysis pipeline."""

    def __init__(self, client: Any):
        self.client = client

    def invoke(self, input: Any, config: Any = None, **kwargs: Any) -> AIMessage:
        response = self.client.invoke(_langchain_messages(input), config=config)
        content = response.content if hasattr(response, "content") else str(response)
        return AIMessage(content=content)


class TransformersBackend(Runnable):
    """Lazy local inference; requires torch, transformers, and model weights."""

    def __init__(self, model_id: str, max_new_tokens: int = 512):
        self.model_id = model_id
        self.max_new_tokens = max_new_tokens
        self._pipe = None

    def invoke(self, input: Any, config: Any = None, **kwargs: Any) -> AIMessage:
        if self._pipe is None:
            from transformers import pipeline

            self._pipe = pipeline(
                "text-generation",
                model=self.model_id,
                device_map="auto",
                trust_remote_code=False,
            )
        messages = _messages(input)
        prompt = _render_prompt(messages)
        tokenizer = self._pipe.tokenizer
        if getattr(tokenizer, "chat_template", None):
            prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        output = self._pipe(
            prompt,
            max_new_tokens=self.max_new_tokens,
            return_full_text=False,
        )
        return AIMessage(content=output[0]["generated_text"])


def create_llm() -> Runnable:
    """Select API, Ollama, or local Transformers via environment variables."""
    mode = os.getenv("TXSHIELD_LLM_MODE", "api").strip().lower()
    if mode == "api":
        provider = os.getenv("TXSHIELD_API_PROVIDER", "kimi").strip().lower()
        if provider == "kimi":
            from txshield.contract_analyzer.backends.kimi import KimiBackend

            return KimiBackend(
                api_key=_required_environment("MOONSHOT_API_KEY"),
                prompt_builder=_messages,
                base_url=(
                    os.getenv("TXSHIELD_API_BASE_URL")
                    or "https://api.moonshot.cn/v1"
                ),
                model=os.getenv("TXSHIELD_LLM_MODEL") or "kimi-k3",
                max_completion_tokens=int(
                    os.getenv("TXSHIELD_KIMI_MAX_COMPLETION_TOKENS", "2048")
                ),
                reasoning_effort=os.getenv("TXSHIELD_KIMI_REASONING_EFFORT", "high"),
                connect_timeout=float(os.getenv("TXSHIELD_KIMI_CONNECT_TIMEOUT", "10")),
                read_timeout=float(os.getenv("TXSHIELD_LLM_TIMEOUT", "180")),
                max_retries=int(os.getenv("TXSHIELD_LLM_RETRIES", "2")),
                retry_backoff=float(os.getenv("TXSHIELD_KIMI_RETRY_BACKOFF", "2")),
            )
        if provider not in _API_PROVIDERS:
            raise ValueError(f"Unsupported official API provider: {provider}")
        key_name, base_url, default_model = _API_PROVIDERS[provider]
        from langchain_openai import ChatOpenAI

        return ChatBackend(
            ChatOpenAI(
                api_key=_required_environment(key_name),
                base_url=os.getenv("TXSHIELD_API_BASE_URL") or base_url,
                model=os.getenv("TXSHIELD_LLM_MODEL") or default_model,
                temperature=0,
                timeout=float(os.getenv("TXSHIELD_LLM_TIMEOUT", "120")),
                max_retries=int(os.getenv("TXSHIELD_LLM_RETRIES", "2")),
            )
        )
    if mode == "ollama":
        from langchain_ollama import ChatOllama
        return ChatBackend(
            ChatOllama(
                model=os.getenv("TXSHIELD_OLLAMA_MODEL", "qwen2.5:3b"),
                base_url=os.getenv(
                    "TXSHIELD_OLLAMA_URL", "http://localhost:11434"
                ),
                temperature=0,
            )
        )
    if mode == "local":
        model_id = os.getenv("TXSHIELD_LOCAL_MODEL")
        if not model_id:
            raise ValueError("Set TXSHIELD_LOCAL_MODEL to a local path or model ID")
        return TransformersBackend(
            model_id,
            int(os.getenv("TXSHIELD_MAX_NEW_TOKENS", "512")),
        )
    raise ValueError("TXSHIELD_LLM_MODE must be api, ollama, or local")
