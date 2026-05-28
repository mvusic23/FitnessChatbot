"""Ingest documents from data/knowledge into the vector store.

RAG KORAK 1 - Indexiranje dokumenata:
Ovaj modul cita lokalne dokumente, dijeli ih na manje dijelove, embeddira svaki
dio i sprema rezultat u vektor bazu. To odgovara dijelu demo koda:
"dokumenti -> embeddinzi -> VECTOR_DB".
"""

from __future__ import annotations

from pathlib import Path
import re

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
EXERCISE_FIELD_LABELS = (
    "Instructions",
    "Category",
    "Body Part",
    "Body Park",
    "Equipment",
    "Muscle Group",
    "Secondary Muscles",
    "Target",
)
EXERCISE_FIELD_RE = re.compile(
    r"(?P<label>" + "|".join(re.escape(label) for label in EXERCISE_FIELD_LABELS) + r")\s*:"
)
EXERCISE_LINE_RE = re.compile(r"^\s*(?P<number>\d+)\s*:\s*(?P<body>.+)$")


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
    # RAG KORAK 1.2 - Rekurzivno dijeljenje teksta na chunkove:
    # Umjesto naivnog rezanja po fiksnom broju znakova, tekst se dijeli
    # rekurzivno koristeci hijerarhiju separatora:
    #   1. Dvostruki newline (paragrafi)
    #   2. Jednostruki newline (redovi)
    #   3. Tocka + razmak (recenice)
    #   4. Razmak (rijeci)
    #   5. Prazan string (znakovi) - fallback
    # Na svakoj razini pokusavamo podijeliti tekst na smislene cjeline.
    # Ako je chunk prevelik, rekurzivno ga dijelimo sljedecim separatorom.
    text = text.strip()
    if not text:
        return []

    separators = ["\n\n", "\n", ". ", " ", ""]
    return _recursive_split(text, separators, chunk_size, overlap)


def chunk_knowledge_text(
    text: str,
    source: str,
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
) -> list[str]:
    """Choose the best chunking strategy for a knowledge document."""
    if source == "vzb.txt":
        chunks = chunk_exercise_catalog(text)
        if chunks:
            return chunks
    return chunk_text(text, chunk_size=chunk_size, overlap=overlap)


def chunk_exercise_catalog(text: str) -> list[str]:
    """Chunk vzb.txt as one exercise record per chunk.

    The file is a structured exercise catalog, not prose. Keeping each exercise
    atomic prevents overlap from mixing instructions and metadata across
    unrelated exercises, while the normalized field layout improves retrieval by
    name, muscle group, equipment, category, and target.
    """
    chunks: list[str] = []
    for line in text.splitlines():
        record = _parse_exercise_record(line)
        if record:
            chunks.append(_format_exercise_chunk(record))
    return chunks


def _parse_exercise_record(line: str) -> dict[str, str] | None:
    line = line.strip()
    if not line:
        return None

    line_match = EXERCISE_LINE_RE.match(line)
    if not line_match:
        return None

    body = line_match.group("body")
    name, separator, field_text = body.partition(", Instructions:")
    if not separator:
        return None

    fields = _parse_exercise_fields("Instructions:" + field_text)
    if not fields:
        return None

    fields["Number"] = line_match.group("number")
    fields["Name"] = name.strip()
    return fields


def _parse_exercise_fields(text: str) -> dict[str, str]:
    matches = list(EXERCISE_FIELD_RE.finditer(text))
    fields: dict[str, str] = {}
    for idx, match in enumerate(matches):
        label = match.group("label")
        next_start = matches[idx + 1].start() if idx + 1 < len(matches) else len(text)
        value = text[match.end() : next_start].strip(" ,")
        if label == "Body Park":
            label = "Body Part"
        fields[label] = value
    return fields


def _format_exercise_chunk(record: dict[str, str]) -> str:
    name = record.get("Name", "")
    readable_name = _split_camel_case(name)
    search_terms = _exercise_search_terms(record, readable_name)

    lines = [
        f"Exercise #{record.get('Number', '')}: {readable_name}",
        f"Canonical name: {name}",
        f"Instructions: {record.get('Instructions', '')}",
        f"Category: {record.get('Category', '')}",
        f"Body part: {record.get('Body Part', '')}",
        f"Equipment: {record.get('Equipment', '')}",
        f"Primary muscle group: {record.get('Muscle Group', '')}",
        f"Secondary muscles: {record.get('Secondary Muscles', '')}",
        f"Target: {record.get('Target', '')}",
        f"Search terms: {', '.join(search_terms)}",
    ]
    return "\n".join(line for line in lines if not line.endswith(": "))


def _split_camel_case(value: str) -> str:
    spaced = re.sub(r"(?<=[a-z])(?=[A-Z])", " ", value)
    spaced = re.sub(r"(?<=[A-Z])(?=[A-Z][a-z])", " ", spaced)
    return spaced.strip() or value


def _exercise_search_terms(record: dict[str, str], readable_name: str) -> list[str]:
    raw_terms: list[str] = [
        record.get("Name", ""),
        readable_name,
        readable_name.replace(" ", ""),
        record.get("Category", ""),
        record.get("Body Part", ""),
        record.get("Equipment", ""),
        record.get("Muscle Group", ""),
        record.get("Secondary Muscles", ""),
        record.get("Target", ""),
    ]

    terms: list[str] = []
    seen: set[str] = set()
    for raw in raw_terms:
        for term in re.split(r"[,/&]+", raw):
            normalized = term.strip()
            key = normalized.lower()
            if normalized and key not in seen:
                seen.add(key)
                terms.append(normalized)
    return terms


def _recursive_split(
    text: str,
    separators: list[str],
    chunk_size: int,
    overlap: int,
) -> list[str]:
    """Rekurzivno dijeli tekst koristeci hijerarhiju separatora."""
    if len(text) <= chunk_size:
        return [text]

    # Uzmi trenutni separator
    sep = separators[0]
    remaining_seps = separators[1:] if len(separators) > 1 else [""]

    # Podijeli tekst po separatoru
    if sep == "":
        # Fallback: rezi po znakovima s overlapom
        chunks: list[str] = []
        start = 0
        while start < len(text):
            end = start + chunk_size
            chunks.append(text[start:end])
            if end >= len(text):
                break
            start = end - overlap
        return chunks

    splits = text.split(sep)

    # Spajaj splitove u chunkove koji stanu u chunk_size
    chunks = []
    current = ""
    for piece in splits:
        candidate = current + sep + piece if current else piece
        if len(candidate) <= chunk_size:
            current = candidate
        else:
            # Spremi trenutni chunk ako postoji
            if current:
                chunks.append(current)
            # Ako je piece sam prevelik, rekurzivno ga dijeli dubljim separatorom
            if len(piece) > chunk_size:
                chunks.extend(_recursive_split(piece, remaining_seps, chunk_size, overlap))
                current = ""
            else:
                current = piece

    if current:
        chunks.append(current)

    # Dodaj overlap izmedu chunkova
    if overlap > 0 and len(chunks) > 1:
        chunks = _add_overlap(chunks, overlap)

    return chunks


def _add_overlap(chunks: list[str], overlap: int) -> list[str]:
    """Dodaje overlap tako da svaki chunk (osim prvog) pocinje krajem prethodnog."""
    result = [chunks[0]]
    for i in range(1, len(chunks)):
        prev_tail = chunks[i - 1][-overlap:]
        combined = prev_tail + chunks[i]
        result.append(combined)
    return result


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
    # Svako automatsko indexiranje ponovno gradi indeks od nule, tako da baza
    # tocno odgovara trenutnom sadrzaju direktorija data/knowledge/.
    store.reset()
    all_records: list[tuple[str, str, int, str]] = []

    for path in files:
        rel = path.relative_to(settings.knowledge_dir)
        source = str(rel)
        try:
            # RAG KORAK 2.1 - Citanje izvornog teksta iz dokumenta.
            raw = _read_file(path)
            chunks = chunk_knowledge_text(raw, source)
        except Exception as exc:
            out.print(f"[red]Failed to read {source}: {exc}[/red]")
            continue
        # RAG KORAK 2.2 - Svaki dokument pretvaramo u jedan ili vise chunkova.
        # doc_id mora biti stabilan kako bi ChromaDB znala koji zapis azurira.
        for idx, chunk in enumerate(chunks):
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
