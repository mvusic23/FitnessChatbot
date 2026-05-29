from __future__ import annotations

from fitness_chatbot.client import OllamaClient
from fitness_chatbot.config import Settings
from fitness_chatbot.rag.vector_store import VectorStore


def dohvatikontekst(
    query: str,
    client: OllamaClient,
    store: VectorStore,
    settings: Settings,
) -> list[dict]:
    if store.brojac == 0:
        return []
    embedding = client.embediraj(query)
    return store.pretrazi(embedding, n=settings.rag_top_n)


def ragdostupan(store: VectorStore) -> bool:
    return store.brojac > 0
