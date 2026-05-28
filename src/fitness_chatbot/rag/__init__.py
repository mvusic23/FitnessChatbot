"""RAG: ingest documents, store embeddings, retrieve context."""

from fitness_chatbot.rag.ingest import ingest_knowledge_base
from fitness_chatbot.rag.retrieve import retrieve_context

__all__ = ["ingest_knowledge_base", "retrieve_context"]
