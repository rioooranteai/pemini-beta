from abc import ABC, abstractmethod
from app.services.retrieval.models import RetrievalRequest, RetrievedChunk


class RetrievalBase(ABC):

    @abstractmethod
    async def retrieve(self, request: RetrievalRequest) -> list[RetrievedChunk]:
        """Retrieve relevant chunks for a query."""
        ...