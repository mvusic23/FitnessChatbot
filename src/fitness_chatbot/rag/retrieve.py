"""Retrieve relevant knowledge chunks for a user query.

RAG retrieval:
Ovaj modul odgovara demo funkciji dohvati(upit, top_n).
Korisnicki upit se embeddira, usporedi s embeddingima u vektor bazi i pretvori
u tekstualni kontekst koji ce se dodati LLM promptu.
"""

from __future__ import annotations

from fitness_chatbot.client import OllamaClient
from fitness_chatbot.config import Settings
from fitness_chatbot.rag.vector_store import VectorStore

MAX_CONTEXT_CHARS = 6000


def format_context(hits: list[dict]) -> str:
    # RAG KORAK 4 - Formatiranje pronadenih dokumenata u kontekst:
    # Vektor baza vraca strukturirane rezultate, a LLM treba obican tekst.
    # Zato svaki pronadeni chunk pretvaramo u blok s oznakom izvora.
    if not hits:
        return ""
    parts: list[str] = []
    total = 0
    for hit in hits:
        source = hit.get("source", "unknown")
        text = (hit.get("text") or "").strip()
        block = f"[source: {source}]\n{text}"
        # RAG zastita - ogranicenje velicine konteksta:
        # Ne saljemo previse teksta u prompt jer model ima ogranicen context window
        # i jer losiji/previse dug kontekst moze pogorsati odgovor.
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
    # RAG KORAK 1 - Provjera postoji li indeks:
    # Ako vektor baza nema zapisa, preskacemo retrieval i chatbot radi kao obican LLM.
    if store.count == 0:
        return ""

    # RAG KORAK 2 - Embedding korisnickog upita:
    # Isto kao dokumente kod automatskog indexiranja, i pitanje pretvaramo u
    # vektor brojeva.
    # Tako pitanje i dokumente mozemo usporediti u istom embedding prostoru.
    embedding = client.embed(query)

    # RAG KORAK 3 - Dohvat top-K najrelevantnijih chunkova:
    # store.query radi semanticku pretragu i vraca chunkove najslicnije upitu.
    hits = store.query(embedding, k=settings.rag_top_k)

    # RAG KORAK 4 - Pretvaranje rezultata retrievala u tekst za prompt.
    return format_context(hits)


def rag_available(store: VectorStore) -> bool:
    # RAG availability:
    # RAG je uvijek ukljucen; retrieval moze vratiti kontekst samo ako postoji
    # barem jedan indeksirani chunk u vektor bazi.
    return store.count > 0
