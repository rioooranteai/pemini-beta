from dataclasses import dataclass

@dataclass
class RetrievalRequest:
    query: str
    top_k: int = 5
    filters: dict | None = None


@dataclass
class RetrievedChunk:
    chunk_id: str
    content: str
    metadata: dict
    score: float