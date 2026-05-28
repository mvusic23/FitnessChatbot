"""Ingest documents from data/knowledge into the vector store."""

from __future__ import annotations

from pathlib import Path

from pypdf import PdfReader
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

from fitness_chatbot.client import OllamaClient
from fitness_chatbot.config import Settings
from fitness_chatbot.rag.vector_store import ChunkRecord, VectorStore

CHUNK_SIZE = 1500
CHUNK_OVERLAP = 200
SUPPORTED_SUFFIXES = {".md", ".txt", ".pdf"}
BATCH_SIZE = 16


def _read_file(path: Path) -> str:
    if path.suffix.lower() == ".pdf":
        reader = PdfReader(str(path))
        parts = []
        for page in reader.pages:
            text = page.extract_text()
            if text:
                parts.append(text)
        return "\n".join(parts)
    return path.read_text(encoding="utf-8", errors="replace")


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    text = text.strip()
    if not text:
        return []
    if len(text) <= chunk_size:
        return [text]

    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        if end >= len(text):
            break
        start = end - overlap
    return chunks


def _discover_files(knowledge_dir: Path) -> list[Path]:
    if not knowledge_dir.is_dir():
        return []
    files: list[Path] = []
    for path in sorted(knowledge_dir.rglob("*")):
        if path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES:
            files.append(path)
    return files


def ingest_knowledge_base(
    settings: Settings,
    client: OllamaClient,
    store: VectorStore,
    console: Console | None = None,
) -> int:
    """Rebuild the index from knowledge_dir. Returns number of chunks indexed."""
    out = console or Console()
    files = _discover_files(settings.knowledge_dir)
    if not files:
        out.print("[yellow]No documents found in data/knowledge/ (.md, .txt, .pdf)[/yellow]")
        store.reset()
        return 0

    store.reset()
    all_records: list[tuple[str, str, int, str]] = []

    for path in files:
        rel = path.relative_to(settings.knowledge_dir)
        source = str(rel)
        try:
            raw = _read_file(path)
        except Exception as exc:
            out.print(f"[red]Failed to read {source}: {exc}[/red]")
            continue
        for idx, chunk in enumerate(chunk_text(raw)):
            doc_id = f"{source}::{idx}"
            all_records.append((doc_id, chunk, idx, source))

    if not all_records:
        out.print("[yellow]No text content extracted from documents.[/yellow]")
        return 0

    indexed = 0
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=out,
    ) as progress:
        task = progress.add_task("Embedding chunks...", total=len(all_records))

        for i in range(0, len(all_records), BATCH_SIZE):
            batch = all_records[i : i + BATCH_SIZE]
            texts = [b[1] for b in batch]
            embeddings = client.embed_batch(texts)
            records = [
                ChunkRecord(
                    id=doc_id,
                    text=text,
                    source=source,
                    chunk_index=idx,
                    embedding=emb,
                )
                for (doc_id, text, idx, source), emb in zip(batch, embeddings, strict=True)
            ]
            store.upsert(records)
            indexed += len(records)
            progress.advance(task, len(batch))

    out.print(f"[green]Indexed {indexed} chunks from {len(files)} file(s).[/green]")
    return indexed
