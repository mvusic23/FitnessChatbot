"""Retrieve relevant knowledge chunks for a user query.

RAG retrieval:
Korisnicki upit se embeddira, usporedi s embeddingima u vektor bazi i pretvori
u tekstualni kontekst koji ce se dodati LLM promptu.
Koristi Top-N strategiju: rangira kandidat chunkove i uzima N najrelevantnijih.
"""

from __future__ import annotations

from fitness_chatbot.client import OllamaClient
from fitness_chatbot.config import Settings
from fitness_chatbot.rag.vector_store import VectorStore

MAX_CONTEXT_CHARS = 6000


def top_n_chunks(hits: list[dict], n: int) -> list[dict]:
    """Return the N most relevant chunks by vector distance."""
    ranked = sorted(
        hits,
        key=lambda hit: (
            float(hit.get("distance", 1.0)),
            str(hit.get("source", "")),
            int(hit.get("chunk_index", 0)),
        ),
    )
    return ranked[:n]


def format_context(hits: list[dict]) -> str:
    if not hits:
        return ""
    parts: list[str] = []
    total = 0
    for hit in hits:
        source = hit.get("source", "unknown")
        distance = hit.get("distance", None)
        score = f" | relevance: {1 - distance:.2f}" if distance is not None else ""
        text = (hit.get("text") or "").strip()
        block = f"\\[source: {source}{score}]\n{text}"
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
    candidates = store.query(embedding, n=settings.rag_top_n)
    hits = top_n_chunks(candidates, settings.rag_top_n)
    return format_context(hits)


def rag_available(store: VectorStore) -> bool:
    return store.count > 0
