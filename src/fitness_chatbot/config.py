"""Application configuration from environment and .env."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

# Project root (FitnessChatbot/)
PROJECT_ROOT = Path(__file__).resolve().parents[2]
KNOWLEDGE_DIR = PROJECT_ROOT / "data" / "knowledge"
CHROMA_DIR = PROJECT_ROOT / "chroma_db"

load_dotenv(PROJECT_ROOT / ".env")


def _env_bool(key: str, default: bool) -> bool:
    raw = os.getenv(key)
    if raw is None:
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


@dataclass(frozen=True)
class Settings:
    ollama_host: str
    ollama_model: str
    embed_model: str
    rag_enabled: bool
    rag_top_k: int
    max_history_turns: int
    knowledge_dir: Path
    chroma_dir: Path


def get_settings() -> Settings:
    return Settings(
        ollama_host=os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434"),
        ollama_model=os.getenv("OLLAMA_MODEL", "llama3.2"),
        embed_model=os.getenv("EMBED_MODEL", "nomic-embed-text"),
        rag_enabled=_env_bool("RAG_ENABLED", True),
        rag_top_k=int(os.getenv("RAG_TOP_K", "4")),
        max_history_turns=int(os.getenv("MAX_HISTORY_TURNS", "20")),
        knowledge_dir=KNOWLEDGE_DIR,
        chroma_dir=CHROMA_DIR,
    )
