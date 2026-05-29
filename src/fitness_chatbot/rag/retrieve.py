"""Dohvacanje relevantnih chunkova za korisnicko pitanje."""

from __future__ import annotations

from fitness_chatbot.client import OllamaClient
from fitness_chatbot.config import Settings
from fitness_chatbot.rag.vector_store import VectorStore

MAX_CONTEXT_CHARS = 6000


def _format_context(hits: list[dict]) -> str:
    parts: list[str] = []
    total = 0
    for hit in hits:
        block = f"[source: {hit['source']}]\n{hit['text']}"
        if total + len(block) > MAX_CONTEXT_CHARS:
            break
        parts.append(block)
        total += len(block)
    return "\n\n".join(parts)


def retrieve_context(
    query: str,
    client: OllamaClient,
    store: VectorStore,
    settings: Settings,
) -> str:
    """Embeddira upit i vraca formatirani kontekst iz najblizih chunkova."""
    if store.count == 0:
        return ""
    embedding = client.embed(query)
    hits = store.query(embedding, n=settings.rag_top_n)
    return _format_context(hits)


def rag_available(store: VectorStore) -> bool:
    return store.count > 0
