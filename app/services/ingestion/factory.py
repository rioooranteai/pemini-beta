from app.infrastructure.embedder.base.embedder_base import EmbedderBase
from app.infrastructure.vector_store.base.vector_store_base import VectorStoreBase
from app.services.ingestion.base.ingestion_base import IngestionBase
from app.services.ingestion.providers.default_provider import DefaultIngestionProvider


class IngestionFactory:

    @staticmethod
    def create(embedder: EmbedderBase, vector_store: VectorStoreBase) -> IngestionBase:
        return DefaultIngestionProvider(
                embedder=embedder,
                vector_store=vector_store,
                chunk_size=512,
                chunk_overlap=64
            )