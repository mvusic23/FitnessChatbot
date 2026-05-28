"""Retrieve relevant knowledge chunks for a user query."""

from __future__ import annotations

from fitness_chatbot.client import OllamaClient
from fitness_chatbot.config import Settings
from fitness_chatbot.rag.vector_store import VectorStore

MAX_CONTEXT_CHARS = 6000


def format_context(hits: list[dict]) -> str:
    if not hits:
        return ""
    parts: list[str] = []
    total = 0
    for hit in hits:
        source = hit.get("source", "unknown")
        text = (hit.get("text") or "").strip()
        block = f"[source: {source}]\n{text}"
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
    if store.count == 0:
        return ""
    embedding = client.embed(query)
    hits = store.query(embedding, k=settings.rag_top_k)
    return format_context(hits)


def rag_available(store: VectorStore, settings: Settings) -> bool:
    return settings.rag_enabled and store.count > 0
