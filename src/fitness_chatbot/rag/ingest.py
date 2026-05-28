"""Ingest documents from data/knowledge into the vector store.

RAG KORAK 1 - Indexiranje dokumenata:
Ovaj modul cita lokalne dokumente, dijeli ih na manje dijelove, embeddira svaki
dio i sprema rezultat u vektor bazu. To odgovara dijelu demo koda:
"dokumenti -> embeddinzi -> VECTOR_DB".
"""

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
    # RAG KORAK 1.1 - Ucitavanje dokumenta:
    # Ako je dokument PDF, izvlacimo tekst sa svake stranice.
    # Ako je .md ili .txt, citamo ga kao obican UTF-8 tekst.
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
    # RAG KORAK 1.2 - Dijeljenje teksta na chunkove:
    # LLM i embedding modeli imaju ogranicen kontekst, pa veliki dokument ne
    # spremamo kao jedan ogroman tekst. Dijelimo ga na manje dijelove.
    # Overlap cuva dio prethodnog teksta kako se vazne recenice ne bi izgubile
    # na granici izmedu dva chunka.
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
    # RAG KORAK 1.0 - Pronalazak knowledge base dokumenata:
    # Pretrazujemo data/knowledge/ i uzimamo samo formate koje znamo obraditi.
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

    # RAG KORAK 1 - Ucitavanje knowledge basea:
    # Pronalazimo sve dokumente koje korisnik zeli koristiti kao izvor znanja.
    files = _discover_files(settings.knowledge_dir)
    if not files:
        out.print("[yellow]No documents found in data/knowledge/ (.md, .txt, .pdf)[/yellow]")
        # Ako nema dokumenata, brisemo stari indeks da chatbot ne koristi
        # zastarjele informacije iz prethodnog indexiranja.
        store.reset()
        return 0

    # RAG KORAK 2 - Priprema vektor baze:
    # Svaki /ingest ponovno gradi indeks od nule, tako da baza tocno odgovara
    # trenutnom sadrzaju direktorija data/knowledge/.
    store.reset()
    all_records: list[tuple[str, str, int, str]] = []

    for path in files:
        rel = path.relative_to(settings.knowledge_dir)
        source = str(rel)
        try:
            # RAG KORAK 2.1 - Citanje izvornog teksta iz dokumenta.
            raw = _read_file(path)
        except Exception as exc:
            out.print(f"[red]Failed to read {source}: {exc}[/red]")
            continue
        # RAG KORAK 2.2 - Svaki dokument pretvaramo u jedan ili vise chunkova.
        # doc_id mora biti stabilan kako bi ChromaDB znala koji zapis azurira.
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

            # RAG KORAK 3 - Embedding dokumenata:
            # Svaki tekstualni chunk saljemo embedding modelu (npr. bge-m3).
            # Model vraca vektor brojeva koji predstavlja znacenje teksta.
            # To je ekvivalent demo funkciji dodaj_u_bazu(), samo batchirano.
            embeddings = client.embed_batch(texts)

            # RAG KORAK 4 - Spajanje teksta, metapodataka i embeddinga:
            # Za svaki chunk cuvamo originalni tekst, izvor, redni broj chunka
            # i embedding vektor. To je zapis koji ide u vektor bazu.
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

            # RAG KORAK 5 - Spremanje u vektor bazu:
            # ChromaDB sprema embeddinge i kasnije omogucuje pretragu po
            # semantickoj slicnosti, slicno kao VECTOR_DB iz demo primjera.
            store.upsert(records)
            indexed += len(records)
            progress.advance(task, len(batch))

    out.print(f"[green]Indexed {indexed} chunks from {len(files)} file(s).[/green]")
    return indexed
