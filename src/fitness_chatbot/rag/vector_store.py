"""ChromaDB vector store for fitness knowledge chunks.

RAG vektor baza:
U demo primjeru VECTOR_DB je obicna Python lista parova (tekst, embedding).
Ovdje istu ulogu ima ChromaDB: trajno sprema tekstove, embeddinge i metapodatke
te zna brzo vratiti najrelevantnije zapise za embedding korisnickog upita.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Any

import chromadb
from chromadb.api.models.Collection import Collection

COLLECTION_PREFIX = "fitness_knowledge"


@dataclass
class ChunkRecord:
    id: str
    text: str
    source: str
    chunk_index: int
    embedding: list[float]


def collection_name_for_model(embed_model: str) -> str:
    """Build a stable Chroma collection name for the active embedding model."""
    # RAG tehnicki detalj - razdvajanje indeksa po embedding modelu:
    # Razliciti embedding modeli mogu vracati vektore razlicite dimenzije.
    # Zato za bge-m3 koristimo zasebnu Chroma kolekciju od npr. nomic-embed-text.
    normalized = re.sub(r"[^a-zA-Z0-9_-]+", "_", embed_model).strip("_-")
    if not normalized:
        normalized = "default"
    return f"{COLLECTION_PREFIX}_{normalized}"[:63]


class VectorStore:
    def __init__(self, persist_dir: Path, embed_model: str) -> None:
        # RAG KORAK - Otvaranje vektor baze:
        # persist_dir je direktorij na disku u kojem ChromaDB cuva indeks.
        # Collection je "tablica" dokumenata za aktivni embedding model.
        self._collection_name = collection_name_for_model(embed_model)
        persist_dir.mkdir(parents=True, exist_ok=True)
        self._client = chromadb.PersistentClient(path=str(persist_dir))
        self._collection: Collection = self._client.get_or_create_collection(
            name=self._collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    @property
    def count(self) -> int:
        # RAG provjera dostupnosti:
        # Ako je count 0, nema indeksiranih chunkova i retrieval nema sto vratiti.
        return self._collection.count()

    def reset(self) -> None:
        # RAG KORAK - Ponovno indexiranje:
        # Brisemo staru kolekciju prije /ingest kako rezultat pretrage ne bi
        # sadrzavao chunkove iz dokumenata koji vise ne postoje ili su promijenjeni.
        self._client.delete_collection(self._collection_name)
        self._collection = self._client.get_or_create_collection(
            name=self._collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def upsert(self, records: list[ChunkRecord]) -> None:
        if not records:
            return
        # RAG KORAK - Spremanje embeddinga:
        # ids su stabilni identifikatori chunkova.
        # documents su originalni tekstovi koji ce kasnije ici u prompt.
        # embeddings su vektori po kojima se radi semanticka pretraga.
        # metadatas cuvaju izvor i redni broj chunka radi objasnjenja/traceability.
        self._collection.upsert(
            ids=[r.id for r in records],
            documents=[r.text for r in records],
            embeddings=[r.embedding for r in records],
            metadatas=[
                {"source": r.source, "chunk_index": r.chunk_index} for r in records
            ],
        )

    def query(self, embedding: list[float], k: int = 4) -> list[dict[str, Any]]:
        if self.count == 0:
            return []
        # RAG KORAK - Retrieval iz vektor baze:
        # Ulaz je embedding korisnickog pitanja.
        # ChromaDB ga usporeduje s embeddingom svakog dokument chunk-a koristeci
        # cosine similarity i vraca k najblizih zapisa.
        result = self._collection.query(
            query_embeddings=[embedding],
            n_results=min(k, self.count),
            include=["documents", "metadatas", "distances"],
        )
        docs = (result.get("documents") or [[]])[0]
        metas = (result.get("metadatas") or [[]])[0]
        out: list[dict[str, Any]] = []
        # RAG KORAK - Normalizacija rezultata:
        # Iz ChromaDB odgovora uzimamo samo ono sto ostatku aplikacije treba:
        # tekst chunka, naziv izvora i redni broj chunka.
        for doc, meta in zip(docs, metas, strict=False):
            out.append(
                {
                    "text": doc or "",
                    "source": (meta or {}).get("source", "unknown"),
                    "chunk_index": (meta or {}).get("chunk_index", 0),
                }
            )
        return out
