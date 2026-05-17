from app.infrastructure.embedder.base.embedder_base import EmbedderBase
from app.infrastructure.vector_store.base.vector_store_base import VectorStoreBase
from app.services.retrieval.base.retrieval_base import RetrievalBase, RetrievalRequest, RetrievedChunk
from app.services.retrieval.reranker import BM25Reranker


class HybridRetrievalProvider(RetrievalBase):

    def __init__(
        self,
        embedder: EmbedderBase,
        vector_store: VectorStoreBase,
        bm25_weight: float = 0.4,
        vector_weight: float = 0.6,
        vector_fetch_multiplier: int = 3
    ) -> None:
        self._embedder = embedder
        self._vector_store = vector_store
        self._reranker = BM25Reranker(bm25_weight, vector_weight)
        self._fetch_multiplier = vector_fetch_multiplier

    async def retrieve(self, request: RetrievalRequest) -> list[RetrievedChunk]:
        # Step 1: Embed query
        query_embedding = await self._embedder.embed(request.query)

        # Step 2: Fetch more candidates than needed for reranking
        fetch_k = request.top_k * self._fetch_multiplier
        vector_results_raw = await self._vector_store.search(
            query_embedding=query_embedding,
            top_k=fetch_k,
            filters=request.filters
        )

        vector_chunks = [
            RetrievedChunk(
                chunk_id=r.id,
                content=r.content,
                metadata=r.metadata,
                score=r.score
            )
            for r in vector_results_raw
        ]

        # Step 3: Get all docs for BM25 corpus
        all_docs = await self._vector_store.get_all_documents()
        all_chunks = [
            RetrievedChunk(
                chunk_id=doc.id,
                content=doc.content,
                metadata=doc.metadata,
                score=0.0
            )
            for doc in all_docs
        ]

        # Step 4: Hybrid rerank
        return self._reranker.rerank(
            query=request.query,
            vector_results=vector_chunks,
            all_chunks=all_chunks,
            top_k=request.top_k
        )