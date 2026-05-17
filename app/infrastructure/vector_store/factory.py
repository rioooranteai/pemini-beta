from app.infrastructure.vector_store.base.vector_store_base import VectorStoreBase
from app.infrastructure.vector_store.providers.chroma_provider import ChromaVectorStoreProvider


class VectorStoreFactory:

    @staticmethod
    def create(path: str, collection_name: str) -> VectorStoreBase:
        return ChromaVectorStoreProvider(
                path=path,
                collection_name=collection_name
            )