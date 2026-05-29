"""Ingest documents from data/knowledge into the vector store.

RAG KORAK 1 - Indexiranje dokumenata:
Ovaj modul cita lokalne dokumente, dijeli ih na chunkove, embeddira svaki chunk
i sprema rezultat u vektor bazu (dokumenti -> embeddinzi -> VECTOR_DB).
"""

from __future__ import annotations

from pathlib import Path

from pypdf import PdfReader
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

from fitness_chatbot.client import OllamaClient
from fitness_chatbot.config import Settings
from fitness_chatbot.rag.vector_store import ChunkRecord, VectorStore

SUPPORTED_SUFFIXES = {".md", ".txt", ".pdf"}
BATCH_SIZE = 16


def _read_file(path: Path) -> str:
    # RAG KORAK 1.1 - Ucitavanje dokumenta:
    # PDF citamo po stranicama, .md/.txt kao obican UTF-8 tekst.
    if path.suffix.lower() == ".pdf":
        reader = PdfReader(str(path))
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    return path.read_text(encoding="utf-8", errors="replace")


def chunk_text(text: str) -> list[str]:
    # RAG KORAK 1.2 - Chunkiranje po linijama:
    # Svaka neprazna linija je jedan atomarni chunk. Time svaki zapis (npr. jedna
    # vjezba u katalogu) ostaje cjelovit, sto top-n retrieval cini preciznijim.
    return [line.strip() for line in text.splitlines() if line.strip()]


def _discover_files(knowledge_dir: Path) -> list[Path]:
    # RAG KORAK 1.0 - Pronalazak knowledge base dokumenata:
    # Pretrazujemo data/knowledge/ i uzimamo samo formate koje znamo obraditi.
    if not knowledge_dir.is_dir():
        return []
    return [
        path
        for path in sorted(knowledge_dir.rglob("*"))
        if path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES
    ]


def ingest_knowledge_base(
    settings: Settings,
    client: OllamaClient,
    store: VectorStore,
    console: Console | None = None,
) -> int:
    """Rebuild the index from knowledge_dir. Returns number of chunks indexed."""
    out = console or Console()

    # RAG KORAK 1 - Ucitavanje knowledge basea.
    files = _discover_files(settings.knowledge_dir)
    if not files:
        out.print("[yellow]No documents found in data/knowledge/ (.md, .txt, .pdf)[/yellow]")
        store.reset()
        return 0

    # RAG KORAK 2 - Svako indexiranje gradi indeks od nule da baza odgovara
    # trenutnom sadrzaju direktorija data/knowledge/.
    store.reset()
    all_records: list[tuple[str, str, int, str]] = []

    for path in files:
        source = str(path.relative_to(settings.knowledge_dir))
        try:
            chunks = chunk_text(_read_file(path))
        except Exception as exc:
            out.print(f"[red]Failed to read {source}: {exc}[/red]")
            continue
        # doc_id mora biti stabilan kako bi ChromaDB znala koji zapis azurira.
        for idx, chunk in enumerate(chunks):
            all_records.append((f"{source}::{idx}", chunk, idx, source))

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

            # RAG KORAK 3 - Embedding chunkova (batchirano).
            embeddings = client.embed_batch([b[1] for b in batch])

            # RAG KORAK 4+5 - Spajanje teksta, metapodataka i embeddinga te
            # spremanje u vektor bazu radi kasnije semanticke pretrage.
            records = [
                ChunkRecord(id=doc_id, text=text, source=source, chunk_index=idx, embedding=emb)
                for (doc_id, text, idx, source), emb in zip(batch, embeddings, strict=True)
            ]
            store.upsert(records)
            indexed += len(records)
            progress.advance(task, len(batch))

    out.print(f"[green]Indexed {indexed} chunks from {len(files)} file(s).[/green]")
    return indexed
