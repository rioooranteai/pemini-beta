import chromadb
from chromadb.config import Settings as ChromaSettings

from app.infrastructure.vector_store.base.vector_store_base import (
    VectorStoreBase, Document, SearchResult
)


class ChromaVectorStoreProvider(VectorStoreBase):

    def __init__(self, path: str, collection_name: str) -> None:
        self._client = chromadb.PersistentClient(
            path=path,
            settings=ChromaSettings(anonymized_telemetry=False)
        )
        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )

    async def upsert(self, documents: list[Document]) -> None:
        self._collection.upsert(
            ids=[doc.id for doc in documents],
            embeddings=[doc.embedding for doc in documents],
            documents=[doc.content for doc in documents],
            metadatas=[doc.metadata for doc in documents],
        )

    async def search(
        self,
        query_embedding: list[float],
        top_k: int = 5,
        filters: dict | None = None
    ) -> list[SearchResult]:
        results = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=filters,
            include=["documents", "metadatas", "distances"]
        )

        search_results = []
        for i, doc_id in enumerate(results["ids"][0]):
            search_results.append(SearchResult(
                id=doc_id,
                content=results["documents"][0][i],
                metadata=results["metadatas"][0][i],
                score=1 - results["distances"][0][i]  # cosine similarity
            ))
        return search_results

    async def delete(self, ids: list[str]) -> None:
        self._collection.delete(ids=ids)

    async def get_all_documents(self) -> list[Document]:
        results = self._collection.get(include=["documents", "metadatas", "embeddings"])
        documents = []
        for i, doc_id in enumerate(results["ids"]):
            documents.append(Document(
                id=doc_id,
                content=results["documents"][i],
                embedding=results["embeddings"][i],
                metadata=results["metadatas"][i]
            ))
        return documents