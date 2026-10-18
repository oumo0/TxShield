"""Official API, Ollama, and local Transformers chat backends.

The selected LLM is independent from the embedding model used by Chroma.
"""
from __future__ import annotations

import os
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.runnables import Runnable

from txshield.contract_analyzer.core.message_manager import MessageManager


def _messages(payload: Any) -> list[dict[str, str]]:
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
    )


def _prompt(payload: Any) -> str:
    """Return a text-only compatibility rendering of structured messages."""
    return "\n\n".join(
        f"{message['role'].upper()}:\n{message['content']}"
        for message in _messages(payload)
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


class ChatBackend(Runnable):
    """Stable adapter for the original Runnable-based analysis pipeline."""

    def __init__(self, client: Any):
        self.client = client

    def invoke(self, input: Any, config: Any = None, **kwargs: Any) -> AIMessage:
        response = self.client.invoke(_langchain_messages(input), config=config)
        return AIMessage(content=response.content if hasattr(response, "content") else str(response))


class TransformersBackend(Runnable):
    """Lazy local inference; requires torch, transformers, and model weights."""

    def __init__(self, model_id: str, max_new_tokens: int = 512):
        self.model_id = model_id
        self.max_new_tokens = max_new_tokens
        self._pipe = None

    def invoke(self, input: Any, config: Any = None, **kwargs: Any) -> AIMessage:
        if self._pipe is None:
            from transformers import pipeline
            # Set TXSHIELD_LOCAL_MODEL to a trusted model ID or local model directory.
            self._pipe = pipeline("text-generation", model=self.model_id,
                                  device_map="auto", trust_remote_code=False)
        messages = _messages(input)
        prompt = _prompt(input)
        tokenizer = self._pipe.tokenizer
        if getattr(tokenizer, "chat_template", None):
            prompt = tokenizer.apply_chat_template(
                messages, tokenize=False,
                add_generation_prompt=True)
        output = self._pipe(prompt, max_new_tokens=self.max_new_tokens,
                            return_full_text=False)
        return AIMessage(content=output[0]["generated_text"])


def create_llm() -> Runnable:
    """Select API, Ollama, or local Transformers via environment variables."""
    mode = os.getenv("TXSHIELD_LLM_MODE", "api").strip().lower()
    if mode == "api":
        provider = os.getenv("TXSHIELD_API_PROVIDER", "deepseek").strip().lower()
        if provider == "kimi":
            api_key = os.getenv("MOONSHOT_API_KEY")
            if not api_key:
                raise ValueError("Missing MOONSHOT_API_KEY. Set it in your local .env file.")
            from txshield.contract_analyzer.backends.kimi import KimiBackend

            return KimiBackend(
                api_key=api_key,
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
        settings = {
            "openai": ("OPENAI_API_KEY", "https://api.openai.com/v1", "gpt-4.1-mini"),
            "deepseek": ("DEEPSEEK_API_KEY", "https://api.deepseek.com/v1", "deepseek-v4-pro"),
        }
        if provider not in settings:
            raise ValueError(f"Unsupported official API provider: {provider}")
        key_name, base_url, default_model = settings[provider]
        api_key = os.getenv(key_name)
        if not api_key:
            raise ValueError(f"Missing {key_name}. Set it in your local .env file.")
        from langchain_openai import ChatOpenAI
        return ChatBackend(ChatOpenAI(
            api_key=api_key,
            base_url=os.getenv("TXSHIELD_API_BASE_URL") or base_url,
            model=os.getenv("TXSHIELD_LLM_MODEL") or default_model,
            temperature=0,
            timeout=float(os.getenv("TXSHIELD_LLM_TIMEOUT", "120")),
            max_retries=int(os.getenv("TXSHIELD_LLM_RETRIES", "2")),
        ))
    if mode == "ollama":
        from langchain_ollama import ChatOllama
        return ChatBackend(ChatOllama(
            model=os.getenv("TXSHIELD_OLLAMA_MODEL", "qwen2.5:3b"),
            base_url=os.getenv("TXSHIELD_OLLAMA_URL", "http://localhost:11434"),
            temperature=0,
        ))
    if mode == "local":
        model_id = os.getenv("TXSHIELD_LOCAL_MODEL")
        if not model_id:
            raise ValueError("Set TXSHIELD_LOCAL_MODEL to a local path or model ID")
        return TransformersBackend(model_id, int(os.getenv("TXSHIELD_MAX_NEW_TOKENS", "512")))
    raise ValueError("TXSHIELD_LLM_MODE must be api, ollama, or local")
