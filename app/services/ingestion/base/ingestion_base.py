from abc import ABC, abstractmethod
from app.services.ingestion.models import IngestResult, IngestRequest

class IngestionBase(ABC):

    @abstractmethod
    async def ingest(self, request: IngestRequest) -> IngestResult:
        """Process and store a single document."""
        ...

    @abstractmethod
    async def ingest_batch(self, requests: list[IngestRequest]) -> list[IngestResult]:
        """Process and store multiple documents."""
        ...

    @abstractmethod
    async def delete(self, doc_id: str) -> None:
        """Delete all chunks of a document by doc_id."""
        ...