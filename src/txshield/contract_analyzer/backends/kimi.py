from __future__ import annotations

from typing import Any, Callable

import requests
from langchain_core.messages import AIMessage
from langchain_core.runnables import Runnable
from requests.adapters import HTTPAdapter, Retry


class KimiBackend(Runnable):
    """Invoke Kimi K3 without sending unsupported sampling parameters."""

    def __init__(
        self,
        api_key: str,
        prompt_builder: Callable[[Any], list[dict[str, str]]],
        *,
        base_url: str = "https://api.moonshot.cn/v1",
        model: str = "kimi-k3",
        max_completion_tokens: int = 2_048,
        reasoning_effort: str = "high",
        connect_timeout: float = 10.0,
        read_timeout: float = 180.0,
        max_retries: int = 2,
        retry_backoff: float = 2.0,
    ) -> None:
        if not api_key:
            raise ValueError("MOONSHOT_API_KEY is required for the Kimi provider")
        if not base_url.startswith(("https://", "http://")):
            raise ValueError("Kimi base URL must start with http:// or https://")
        if max_completion_tokens <= 0:
            raise ValueError("Kimi max_completion_tokens must be positive")
        if reasoning_effort not in {"low", "high", "max"}:
            raise ValueError("Kimi reasoning_effort must be low, high, or max")

        self.api_key = api_key
        self.api_url = f"{base_url.rstrip('/')}/chat/completions"
        self.model = model
        self.max_completion_tokens = max_completion_tokens
        self.reasoning_effort = reasoning_effort
        self.timeout = (connect_timeout, read_timeout)
        self.prompt_builder = prompt_builder
        self.session = requests.Session()
        retries = Retry(
            total=max_retries,
            backoff_factor=retry_backoff,
            status_forcelist=(408, 429, 500, 502, 503, 504),
            allowed_methods=("POST",),
            respect_retry_after_header=True,
        )
        self.session.mount("https://", HTTPAdapter(max_retries=retries))
        self.session.mount("http://", HTTPAdapter(max_retries=retries))

    def invoke(self, input: Any, config: Any = None, **kwargs: Any) -> AIMessage:
        payload = {
            "model": self.model,
            "messages": self.prompt_builder(input),
            "stream": False,
            "max_completion_tokens": self.max_completion_tokens,
            "reasoning_effort": self.reasoning_effort,
        }
        try:
            response = self.session.post(
                self.api_url,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                json=payload,
                timeout=self.timeout,
            )
            response.raise_for_status()
            data = response.json()
            choice = data["choices"][0]
            if choice.get("finish_reason") not in {None, "stop"}:
                raise ValueError("Kimi returned an incomplete final response")
            message = choice["message"]
            content = message.get("content")
            if not isinstance(content, str) or not content.strip():
                raise ValueError("Kimi returned an empty response")
            return AIMessage(content=content.strip())
        except (requests.RequestException, ValueError, KeyError, IndexError, TypeError) as exc:
            raise RuntimeError(f"Kimi API request failed: {exc}") from exc
