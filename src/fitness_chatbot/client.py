"""Ollama API client za chat i embeddinge."""

from __future__ import annotations

from collections.abc import Iterator

from ollama import Client

from fitness_chatbot.config import Settings


class OllamaClient:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._client = Client(host=settings.ollama_host)

    def streamajchat(self, messages: list[dict[str, str]]) -> Iterator[str]:
        """Streamaj odgovor modela chunk po chunk."""
        stream = self._client.chat(
            model=self.settings.ollama_model,
            messages=messages,
            stream=True,
        )
        for part in stream:
            content = part.message.content
            if content:
                yield content

    def embediraj(self, text: str) -> list[float]:
        """Generiraj embedding vektor za jedan tekst."""
        response = self._client.embed(model=self.settings.embed_model, input=text)
        return response.embeddings[0]

    def embedirajbatch(self, texts: list[str]) -> list[list[float]]:
        """Generiraj embedding vektore za listu tekstova."""
        if not texts:
            return []
        response = self._client.embed(model=self.settings.embed_model, input=texts)
        return response.embeddings
