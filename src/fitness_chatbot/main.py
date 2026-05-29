from __future__ import annotations

import pyfiglet
from prompt_toolkit import prompt as pt_prompt
from prompt_toolkit.formatted_text import HTML
from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.live import Live
from rich.spinner import Spinner

from fitness_chatbot.client import OllamaClient
from fitness_chatbot.prompt import izradiporuku
from fitness_chatbot.rag.ingest import indeksiraj
from fitness_chatbot.rag.retrieve import dohvatikontekst
from fitness_chatbot.rag.vector_store import VectorStore
from fitness_chatbot.config import Settings

EXIT_COMMANDS = ("/quit", "/exit")
BLOCK_CHARS = "█▀▄▌▐"
BORDER_CHARS = "╔═╗║╚╝╠╣╬╦╩"


def dohvatipostavke() -> Settings:
    return Settings()


def stvoribanner(raw: str) -> str:
    parts = []
    for ch in raw:
        if ch in BLOCK_CHARS:
            parts.append(f"[rgb(137,207,240)]{ch}[/rgb(137,207,240)]")
        elif ch in BORDER_CHARS:
            parts.append(f"[white]{ch}[/white]")
        else:
            parts.append(ch)
    return "".join(parts)


def pocetniekran(settings: Settings) -> Panel:
    banner = stvoribanner(
        pyfiglet.figlet_format("Fitness Coach", font="ansi_shadow")
    )
    body = (
        "[bold bright_white]Dobrodošli u[/bold bright_white]\n\n" + banner
        + f"\n▪ [bold]Fitness Coach[/bold] (Ollama: {settings.ollama_model})"
        + "\n▪ /quit za izlaz iz Chatbot-a. Ctrl+C za otkazivanje odgovora."
    )
    return Panel(body, border_style="white", box=box.ROUNDED)


def streamajodgovor(
    client: OllamaClient, messages: list[dict[str, str]], console: Console
) -> str:
    chunks: list[str] = []
    try:
        with Live(Spinner("dots", text="Generiram odgovor..."), console=console, transient=True):
            stream = client.streamajchat(messages)
            first = next(stream, None)
        if first is None:
            return ""
        console.print("[bold rgb(137,207,240)]Coach:[/bold rgb(137,207,240)] ", end="")
        console.print(first, end="")
        chunks.append(first)
        for chunk in stream:
            console.print(chunk, end="")
            chunks.append(chunk)
    except KeyboardInterrupt:
        pass
    console.print()
    return "".join(chunks)


def odgovori(user_input: str, history: list[dict[str, str]], client: OllamaClient, store: VectorStore, settings: Settings, console: Console) -> None:
    hits = dohvatikontekst(user_input, client, store, settings)
    rag_context: str | None = None
    if hits:
        panel_lines = []
        for hit in hits:
            score = 1 - hit["distance"]
            panel_lines.append(f"[bold]{hit['source']}[/bold] (score: {score:.2f})\n  {hit['text']}")
        console.print(
            Panel(
                "\n\n".join(panel_lines),
                title="[bold magenta]RAG Context[/bold magenta]",
                border_style="magenta",
            )
        )
        rag_context = "\n\n".join(f"[source: {h['source']}]\n{h['text']}" for h in hits)

    messages = izradiporuku(history, user_input, rag_context=rag_context)
    history.append({"role": "user", "content": user_input})
    reply = streamajodgovor(client, messages, console)
    if reply:
        history.append({"role": "assistant", "content": reply})


def pokreni() -> None:
    console = Console()
    settings = dohvatipostavke()
    client = OllamaClient(settings)
    store = VectorStore(settings.chroma_dir, settings.embed_model)
    history: list[dict[str, str]] = []

    indeksiraj(settings, client, store, console)
    console.print(pocetniekran(settings))

    while True:
        try:
            user_input = pt_prompt(
                HTML("<b>&gt;</b> "),
                placeholder=HTML("<style color='gray'>pitaj me pitanje.. ↵</style>"),
            ).strip()
        except (EOFError, KeyboardInterrupt):
            break

        if not user_input:
            continue
        if user_input.lower() in EXIT_COMMANDS:
            break

        odgovori(user_input, history, client, store, settings, console)


def main():
    pokreni()
