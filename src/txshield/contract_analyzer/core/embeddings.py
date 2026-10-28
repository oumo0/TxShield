from __future__ import annotations

import os

import requests
from langchain_core.embeddings import Embeddings
from requests.adapters import HTTPAdapter, Retry

from txshield.contract_analyzer.config import CONFIG


class JinaEmbedding(Embeddings):
    """Generate query and document embeddings through the official Jina API."""

    api_url = "https://api.jina.ai/v1/embeddings"

    def __init__(self) -> None:
        self.headers = {
            "Authorization": f"Bearer {CONFIG['JINA_API_KEY']}",
            "Content-Type": "application/json",
        }
        self.timeout = float(os.getenv("TXSHIELD_EMBEDDING_TIMEOUT", "15"))
        self.session = requests.Session()
        proxies = {
            protocol: url
            for protocol, url in (
                ("http", CONFIG.get("HTTP_PROXY")),
                ("https", CONFIG.get("HTTPS_PROXY")),
            )
            if url
        }
        if proxies:
            self.session.proxies.update(proxies)

        retries = Retry(
            total=int(os.getenv("TXSHIELD_EMBEDDING_RETRIES", "3")),
            backoff_factor=float(os.getenv("TXSHIELD_EMBEDDING_BACKOFF", "2")),
            status_forcelist=(429, 500, 502, 503, 504),
            allowed_methods=("POST",),
        )
        self.session.mount("https://", HTTPAdapter(max_retries=retries))

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._get_embedding(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._get_embedding(text)

    def _get_embedding(self, text: str) -> list[float]:
        try:
            response = self.session.post(
                self.api_url,
                headers=self.headers,
                json={
                    "model": CONFIG["EMBEDDING_MODEL"],
                    "input": [text[:8192]],
                },
                timeout=self.timeout,
            )
            response.raise_for_status()
            data = response.json().get("data")
            if not data:
                raise ValueError("API returned no embedding data")
            embedding = data[0].get("embedding")
            if not isinstance(embedding, list):
                raise ValueError("API returned an invalid embedding")
            return embedding
        except (requests.RequestException, ValueError, KeyError, TypeError) as exc:
            raise RuntimeError(f"Embedding API failed: {exc}") from exc
