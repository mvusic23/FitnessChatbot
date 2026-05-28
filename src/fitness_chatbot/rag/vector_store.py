"""ChromaDB vector store for fitness knowledge chunks."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import chromadb
from chromadb.api.models.Collection import Collection

COLLECTION_NAME = "fitness_knowledge"


@dataclass
class ChunkRecord:
    id: str
    text: str
    source: str
    chunk_index: int
    embedding: list[float]


class VectorStore:
    def __init__(self, persist_dir: Path) -> None:
        persist_dir.mkdir(parents=True, exist_ok=True)
        self._client = chromadb.PersistentClient(path=str(persist_dir))
        self._collection: Collection = self._client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )

    @property
    def count(self) -> int:
        return self._collection.count()

    def reset(self) -> None:
        self._client.delete_collection(COLLECTION_NAME)
        self._collection = self._client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )

    def upsert(self, records: list[ChunkRecord]) -> None:
        if not records:
            return
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
        result = self._collection.query(
            query_embeddings=[embedding],
            n_results=min(k, self.count),
            include=["documents", "metadatas", "distances"],
        )
        docs = (result.get("documents") or [[]])[0]
        metas = (result.get("metadatas") or [[]])[0]
        out: list[dict[str, Any]] = []
        for doc, meta in zip(docs, metas, strict=False):
            out.append(
                {
                    "text": doc or "",
                    "source": (meta or {}).get("source", "unknown"),
                    "chunk_index": (meta or {}).get("chunk_index", 0),
                }
            )
        return out
