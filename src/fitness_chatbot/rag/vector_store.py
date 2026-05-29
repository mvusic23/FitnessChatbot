from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import chromadb

COLLECTION_PREFIX = "fitness_knowledge"


@dataclass
class ChunkRecord:
    id: str
    text: str
    source: str
    chunk_index: int
    embedding: list[float]


def imekolekcije(embed_model: str) -> str:
    normalized = re.sub(r"[^a-zA-Z0-9_-]+", "_", embed_model).strip("_-") or "default"
    return f"{COLLECTION_PREFIX}_{normalized}"[:63]


class VectorStore:
    def __init__(self, persist_dir: Path, embed_model: str) -> None:
        self._name = imekolekcije(embed_model)
        persist_dir.mkdir(parents=True, exist_ok=True)
        self._client = chromadb.PersistentClient(path=str(persist_dir))
        self._collection = self._client.get_or_create_collection(
            name=self._name, metadata={"hnsw:space": "cosine"}
        )

    @property
    def brojac(self) -> int:
        return self._collection.count()

    def resetiraj(self) -> None:
        self._client.delete_collection(self._name)
        self._collection = self._client.get_or_create_collection(
            name=self._name, metadata={"hnsw:space": "cosine"}
        )

    def umetni(self, records: list[ChunkRecord]) -> None:
        if not records:
            return
        self._collection.upsert(
            ids=[r.id for r in records],
            documents=[r.text for r in records],
            embeddings=[r.embedding for r in records],
            metadatas=[{"source": r.source, "chunk_index": r.chunk_index} for r in records],
        )

    def pretrazi(self, embedding: list[float], n: int = 4) -> list[dict[str, Any]]:
        if self.brojac == 0:
            return []
        result = self._collection.query(
            query_embeddings=[embedding],
            n_results=min(n, self.brojac),
            include=["documents", "metadatas", "distances"],
        )
        docs = (result.get("documents") or [[]])[0]
        metas = (result.get("metadatas") or [[]])[0]
        distances = (result.get("distances") or [[]])[0]
        return [
            {"text": doc, "source": meta.get("source", ""), "chunk_index": meta.get("chunk_index", 0), "distance": dist}
            for doc, meta, dist in zip(docs, metas, distances)
        ]
