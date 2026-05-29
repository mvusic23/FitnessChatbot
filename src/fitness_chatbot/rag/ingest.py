"""Indexiranje .txt dokumenata iz data/knowledge/ u vektor bazu."""

from __future__ import annotations

from pathlib import Path

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

from fitness_chatbot.client import OllamaClient
from fitness_chatbot.config import Settings
from fitness_chatbot.rag.vector_store import ChunkRecord, VectorStore

BATCH_SIZE = 16


def pronadidatoteke(knowledge_dir: Path) -> list[Path]:
    if not knowledge_dir.is_dir():
        return []
    return sorted(p for p in knowledge_dir.rglob("*.txt") if p.is_file())


def podijelijtekst(text: str) -> list[str]:
    """Svaka neprazna linija je jedan chunk."""
    return [line.strip() for line in text.splitlines() if line.strip()]


def indeksiraj(
    settings: Settings,
    client: OllamaClient,
    store: VectorStore,
    console: Console | None = None,
) -> int:
    """Ponovno indeksira knowledge base. Vraca broj indeksiranih chunkova."""
    out = console or Console()
    files = pronadidatoteke(settings.knowledge_dir)

    store.resetiraj()

    if not files:
        return 0

    all_records: list[tuple[str, str, int, str]] = []
    for path in files:
        source = str(path.relative_to(settings.knowledge_dir))
        chunks = podijelijtekst(path.read_text(encoding="utf-8"))
        for idx, chunk in enumerate(chunks):
            all_records.append((f"{source}::{idx}", chunk, idx, source))

    if not all_records:
        return 0

    indexed = 0
    with Progress(SpinnerColumn(), TextColumn("{task.description}"), console=out) as progress:
        task = progress.add_task("Embeddiranje chunkova...", total=len(all_records))
        for i in range(0, len(all_records), BATCH_SIZE):
            batch = all_records[i : i + BATCH_SIZE]
            embeddings = client.embedirajbatch([b[1] for b in batch])
            records = [
                ChunkRecord(id=doc_id, text=text, source=source, chunk_index=idx, embedding=emb)
                for (doc_id, text, idx, source), emb in zip(batch, embeddings, strict=True)
            ]
            store.umetni(records)
            indexed += len(records)
            progress.advance(task, len(batch))

    out.print(f"[green]Indeksirano {indexed} chunkova iz {len(files)} datoteke.[/green]")
    return indexed
