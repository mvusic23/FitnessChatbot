"""Dohvacanje relevantnih chunkova za korisnicko pitanje."""

from __future__ import annotations

from fitness_chatbot.client import OllamaClient
from fitness_chatbot.config import Settings
from fitness_chatbot.rag.vector_store import VectorStore


def retrieve_context(
    query: str,
    client: OllamaClient,
    store: VectorStore,
    settings: Settings,
) -> list[dict]:
    """Embeddira upit i vraca top-k najblizih chunkova."""
    if store.count == 0:
        return []
    embedding = client.embed(query)
    return store.query(embedding, n=settings.rag_top_n)


def rag_available(store: VectorStore) -> bool:
    return store.count > 0
