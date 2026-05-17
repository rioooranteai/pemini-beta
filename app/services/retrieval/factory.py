from app.infrastructure.embedder.base.embedder_base import EmbedderBase
from app.infrastructure.vector_store.base.vector_store_base import VectorStoreBase
from app.services.retrieval.base.retrieval_base import RetrievalBase
from app.services.retrieval.providers.hybrid_provider import HybridRetrievalProvider

class RetrievalFactory:
    @staticmethod
    def create(
        embedder: EmbedderBase,
        vector_store: VectorStoreBase,
        bm25_weight: float,
        vector_weight: float
    ) -> RetrievalBase:

        return HybridRetrievalProvider(
            embedder=embedder,
            vector_store=vector_store,
            bm25_weight=bm25_weight,
            vector_weight=vector_weight
        )