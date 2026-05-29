"""Ollama API client for chat and embeddings."""

from __future__ import annotations

from collections.abc import Iterator
from ollama import Client

from fitness_chatbot.config import Settings


class OllamaClient:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._client = Client(host=settings.ollama_host)
 
    def chat_stream(self, messages: list[dict[str, str]]) -> Iterator[str]:
        stream = self._client.chat(
            model=self.settings.ollama_model,
            messages=messages,
            stream=True,
        )
        for part in stream:
            message = getattr(part, "message", None) or (
                part.get("message") if isinstance(part, dict) else None
            )
            if message is None:
                continue
            content = getattr(message, "content", None) or (
                message.get("content") if isinstance(message, dict) else None
            )
            if content:
                yield content

    def embed(self, text: str) -> list[float]:
        # RAG KORAK - Embedding jednog teksta:
        # Koristi se kod retrievala, kada korisnicko pitanje treba pretvoriti
        # u vektor kako bi se moglo usporediti s dokumentima u vektor bazi.
        response = self._client.embed(
            model=self.settings.embed_model,
            input=text,
        )
        embeddings = getattr(response, "embeddings", None) or response.get("embeddings") or []
        if not embeddings:
            raise RuntimeError(f"No embedding returned for model {self.settings.embed_model}")
        return embeddings[0]

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        # RAG KORAK - Batch embedding dokumenata:
        # Koristi se kod automatskog indexiranja knowledge basea. Umjesto da
        # svaki chunk saljemo Ollami posebno, saljemo listu tekstova i dobijemo
        # listu embedding vektora istim redom.
        response = self._client.embed(
            model=self.settings.embed_model,
            input=texts,
        )
        embeddings: list[list[float]] = (
            getattr(response, "embeddings", None) or response.get("embeddings") or []
        )
        if len(embeddings) != len(texts):
            raise RuntimeError(
                f"Expected {len(texts)} embeddings, got {len(embeddings)}"
            )
        return embeddings
