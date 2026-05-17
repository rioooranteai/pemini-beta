from typing import Any

from pydantic import BaseModel


# Ingest
class IngestDocumentRequest(BaseModel):
    doc_id: str
    content: str
    metadata: dict[str, Any] = {}


class IngestBatchRequest(BaseModel):
    documents: list[IngestDocumentRequest]


class IngestResponse(BaseModel):
    doc_id: str
    chunks_stored: int
    success: bool


# Retrieval
class RetrieveRequest(BaseModel):
    query: str
    top_k: int = 5
    filters: dict[str, Any] | None = None


class RetrievedChunkResponse(BaseModel):
    chunk_id: str
    content: str
    metadata: dict[str, Any]
    score: float


class RetrieveResponse(BaseModel):
    query: str
    results: list[RetrievedChunkResponse]


# Chat
class ChatRequest(BaseModel):
    query: str
    top_k: int = 5
    filters: dict[str, Any] | None = None
    system_prompt: str | None = None


class ChatResponse(BaseModel):
    answer: str
    sources: list[RetrievedChunkResponse]
