from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class Settings:
    ollama_host: str = "http://127.0.0.1:11434"
    ollama_model: str = "jobautomation/OpenEuroLLM-Croatian:latest"
    embed_model: str = "bge-m3"
    rag_top_n: int = 4
    knowledge_dir: Path = PROJECT_ROOT / "data" / "knowledge"
    chroma_dir: Path = PROJECT_ROOT / "chroma_db"
