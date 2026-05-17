from app.infrastructure.embedder.base.embedder_base import EmbedderBase
from app.infrastructure.vector_store.base.vector_store_base import VectorStoreBase, Document
from app.services.ingestion.base.ingestion_base import IngestionBase, IngestRequest, IngestResult
from app.services.ingestion.chunker import TextChunker
from app.services.ingestion.models import Chunk


class DefaultIngestionProvider(IngestionBase):

    def __init__(
            self,
            embedder: EmbedderBase,
            vector_store: VectorStoreBase,
            chunk_size: int = 512,
            chunk_overlap: int = 64
    ) -> None:
        self._embedder = embedder
        self._vector_store = vector_store
        self._chunker = TextChunker(chunk_size, chunk_overlap)

    async def ingest(self, request: IngestRequest) -> IngestResult:
        try:
            if getattr(request, 'bypass_chunking', False):
                chunks = [
                    Chunk(
                        chunk_id=f"{request.doc_id}_full",
                        content=request.content,
                        metadata={**request.metadata, "doc_id": request.doc_id, "chunk_index": 0}
                    )
                ]
            else:

                chunks = self._chunker.chunk(
                    doc_id=request.doc_id,
                    content=request.content,
                    metadata=request.metadata
                )
            embeddings = await self._embedder.embed_batch(
                [chunk.content for chunk in chunks]
            )

            documents = [
                Document(
                    id=chunk.chunk_id,
                    content=chunk.content,
                    embedding=embeddings[i],
                    metadata=chunk.metadata
                )
                for i, chunk in enumerate(chunks)
            ]

            await self._vector_store.upsert(documents)

            return IngestResult(
                doc_id=request.doc_id,
                chunks_stored=len(documents),
                success=True
            )

        except Exception as e:
            return IngestResult(
                doc_id=request.doc_id,
                chunks_stored=0,
                success=False
            )

    async def ingest_batch(self, requests: list[IngestRequest]) -> list[IngestResult]:
        results = []
        for request in requests:
            result = await self.ingest(request)
            results.append(result)
        return results

    async def delete(self, doc_id: str) -> None:
        all_docs = await self._vector_store.get_all_documents()
        ids_to_delete = [
            doc.id for doc in all_docs
            if doc.metadata.get("doc_id") == doc_id
        ]
        if ids_to_delete:
            await self._vector_store.delete(ids_to_delete)
