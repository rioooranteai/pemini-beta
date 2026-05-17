from dataclasses import dataclass

@dataclass
class IngestRequest:
    doc_id: str
    content: str
    metadata: dict


@dataclass
class IngestResult:
    doc_id: str
    chunks_stored: int
    success: bool


@dataclass
class Chunk:
    chunk_id: str
    content: str
    metadata: dict