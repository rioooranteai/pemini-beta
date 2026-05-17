from dataclasses import dataclass
from app.services.retrieval.models import RetrievedChunk

@dataclass
class ChatRequest:
    query: str
    top_k: int = 5
    filters: dict | None = None
    system_prompt: str | None = None


@dataclass
class ChatResponse:
    answer: str
    sources: list[RetrievedChunk]