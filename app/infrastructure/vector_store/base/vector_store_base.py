from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class Document:
    id: str
    content: str
    embedding: list[float]
    metadata: dict


@dataclass
class SearchResult:
    id: str
    content: str
    metadata: dict
    score: float


class VectorStoreBase(ABC):

    @abstractmethod
    async def upsert(self, documents: list[Document]) -> None:
        """Insert or update documents in the vector store."""
        ...

    @abstractmethod
    async def search(
        self,
        query_embedding: list[float],
        top_k: int = 5,
        filters: dict | None = None
    ) -> list[SearchResult]:
        """Search for similar documents by embedding."""
        ...

    @abstractmethod
    async def delete(self, ids: list[str]) -> None:
        """Delete documents by ID."""
        ...

    @abstractmethod
    async def get_all_documents(self) -> list[Document]:
        """Retrieve all documents (used for BM25 re-ranking)."""
        ...