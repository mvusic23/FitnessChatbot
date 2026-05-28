"""CLI REPL for the fitness chatbot."""

from __future__ import annotations

import sys

from rich.console import Console
from rich.panel import Panel

from fitness_chatbot.client import OllamaClient
from fitness_chatbot.config import get_settings
from fitness_chatbot.conversation import Conversation
from fitness_chatbot.prompt import build_messages
from fitness_chatbot.rag.ingest import ingest_knowledge_base
from fitness_chatbot.rag.retrieve import rag_available, retrieve_context
from fitness_chatbot.rag.vector_store import VectorStore

HELP_TEXT = """
[bold]Commands[/bold]
  /help    Show this help
  /clear   Clear conversation history
  /ingest  Rebuild knowledge index from data/knowledge/
  /quit    Exit (also /exit)
"""


def run() -> None:
    console = Console()
    settings = get_settings()
    client = OllamaClient(settings)
    store = VectorStore(settings.chroma_dir)
    conversation = Conversation(max_turns=settings.max_history_turns)

    ok, err = client.check_connection()
    if not ok:
        console.print(f"[red]{err}[/red]")
        sys.exit(1)

    rag_on = rag_available(store, settings)
    rag_hint = ""
    if settings.rag_enabled and not rag_on:
        rag_hint = "\n[dim]RAG: no index yet — add files to data/knowledge/ and run /ingest[/dim]"

    console.print(
        Panel(
            f"[bold]Fitness Coach[/bold] (Ollama: {settings.ollama_model})"
            f"{rag_hint}\nType /help for commands. Ctrl+C to cancel a reply.",
            border_style="green",
        )
    )

    while True:
        try:
            user_input = console.input("[bold cyan]You:[/bold cyan] ").strip()
        except (EOFError, KeyboardInterrupt):
            console.print("\n[dim]Goodbye![/dim]")
            break

        if not user_input:
            continue

        lower = user_input.lower()
        if lower in ("/quit", "/exit"):
            console.print("[dim]Goodbye![/dim]")
            break
        if lower == "/help":
            console.print(HELP_TEXT)
            continue
        if lower == "/clear":
            conversation.clear()
            console.print("[dim]Conversation cleared.[/dim]")
            continue
        if lower == "/ingest":
            ingest_knowledge_base(settings, client, store, console)
            rag_on = rag_available(store, settings)
            continue

        rag_context: str | None = None
        if rag_available(store, settings):
            try:
                rag_context = retrieve_context(user_input, client, store, settings)
            except Exception as exc:
                console.print(f"[yellow]RAG retrieval failed: {exc}[/yellow]")

        messages = build_messages(
            conversation.history,
            user_input,
            rag_context=rag_context,
        )

        conversation.add_user(user_input)
        full_reply: list[str] = []

        console.print("[bold green]Coach:[/bold green] ", end="")
        try:
            for chunk in client.chat_stream(messages):
                console.print(chunk, end="")
                full_reply.append(chunk)
        except KeyboardInterrupt:
            console.print("\n[dim](reply cancelled)[/dim]")
            conversation.pop_last_user()
            continue
        except Exception as exc:
            console.print(f"\n[red]Error: {exc}[/red]")
            conversation.pop_last_user()
            continue

        console.print()
        assistant_text = "".join(full_reply)
        if assistant_text:
            conversation.add_assistant(assistant_text)


if __name__ == "__main__":
    run()
