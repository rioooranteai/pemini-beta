import logging
from pathlib import Path
from typing import BinaryIO, Dict, Any, List

from app.engines.models import ChatRequest, ChatResponse
from app.infrastructure.llm.base.llm_base import LLMBase
from app.services.ingestion.base.ingestion_base import IngestionBase, IngestRequest, IngestResult
from app.services.loaders.base.loader_base import DocumentLoaderBase
from app.services.retrieval.base.retrieval_base import RetrievalBase, RetrievalRequest, RetrievedChunk

logger = logging.getLogger(__name__)

_PROMPT_DIR = Path(__file__).resolve().parents[2] / "prompts"


def _load_prompt(filename: str) -> str:
    path = _PROMPT_DIR / filename
    try:
        return path.read_text(encoding="utf-8")
    except FileNotFoundError:
        raise FileNotFoundError(f"Prompt file tidak ditemukan: {path}")


DEFAULT_SYSTEM_PROMPT = _load_prompt("default_prompt.md")


class RAGEngine:

    def __init__(
            self,
            llm: LLMBase,
            ingestion: IngestionBase,
            retrieval: RetrievalBase,
            loader: DocumentLoaderBase
    ) -> None:
        self._llm = llm
        self._ingestion = ingestion
        self._retrieval = retrieval
        self._loader = loader

    async def process_uploaded_file(
            self,
            filename: str,
            file_stream: BinaryIO,
            metadata: Dict[str, Any]
    ) -> dict:
        """
        Orkestrator utama untuk memproses file unggahan.
        Mengekstrak file via Loader, lalu menyuntikkan hasilnya ke Vector DB (RAG)
        dan mengekspor DataFrame ke Relational DB (NL2SQL).
        """
        if not self._loader:
            raise ValueError("Loader service belum dikonfigurasi pada RAGEngine.")

        logger.info(f"Memulai pemrosesan dokumen: {filename}")

        # Ekstrak data menggunakan Loader
        try:
            load_result = self._loader.load(
                file=file_stream,
                metadata=metadata
            )
        except Exception as e:
            logger.error(f"Gagal mengekstrak dokumen {filename}: {str(e)}")
            raise

        ingest_requests = []
        sql_injection_status = []

        # Proses hasil teks untuk Vector Database (RAG)
        for sheet_name, markdown_text in load_result.text_content.items():
            doc_id = f"{filename}_{sheet_name}"

            # Gabungkan metadata pengguna dengan metadata teknis dokumen
            chunk_metadata = {
                **metadata,
                "source_file": filename,
                "sheet_name": sheet_name,
                "document_type": "parsed_sheet"
            }

            # Pastikan IngestRequest di ingestion_base.py mendukung atribut ini!
            request = IngestRequest(
                doc_id=doc_id,
                content=markdown_text,
                metadata=chunk_metadata,
            )

            # Flag khusus agar TextChunker di DefaultIngestionProvider tidak memotong teks ini.
            setattr(request, "bypass_chunking", True)

            ingest_requests.append(request)

        # Eksekusi Batch Ingestion ke Vector DB
        if ingest_requests:
            ingest_results = await self._ingestion.ingest_batch(ingest_requests)
            vdb_success_count = sum(res.chunks_stored for res in ingest_results if res.success)
            logger.info(f"Berhasil menyimpan {vdb_success_count} dokumen ke Vector DB.")
        else:
            vdb_success_count = 0

        # Proses DataFrame untuk Relational DB (NL2SQL)
        if hasattr(load_result, 'dataframes') and load_result.dataframes:
            sql_injection_status = await self._inject_dataframes_to_sql(
                dataframes=load_result.dataframes,
                metadata=metadata
            )

        return {
            "status": "success",
            "filename": filename,
            "vdb_chunks_stored": vdb_success_count,
            "sql_tables_created": sql_injection_status
        }

    async def _inject_dataframes_to_sql(self, dataframes: dict, metadata: dict) -> List[str]:
        """
        Helper method untuk mengekspor Pandas DataFrame ke PostgreSQL.
        """
        status = []

        for sheet_name, df in dataframes.items():
            try:
                # Membuat nama tabel unik per sesi/user agar aman untuk MVP
                user_id = metadata.get("user_id", "default_user")

                import re
                clean_sheet_name = re.sub(r'[^a-zA-Z0-9_]', '', sheet_name.replace(" ", "_"))
                table_name = f"nl2sql_{user_id}_{clean_sheet_name}".lower()

                logger.info(f"Mengekspor {len(df)} baris ke tabel SQL: {table_name}")

                # Uncomment baris di bawah ini jika engine SQL sudah siap:
                # df.to_sql(name=table_name, con=engine, if_exists='append', index=False)

                status.append(table_name)
            except Exception as e:
                logger.error(f"Gagal menginjeksi DataFrame {sheet_name} ke SQL: {str(e)}")

        return status

    async def ingest(self, doc_id: str, content: str, metadata: dict) -> IngestResult:
        return await self._ingestion.ingest(
            IngestRequest(doc_id=doc_id, content=content, metadata=metadata)
        )

    async def ingest_batch(self, documents: list[dict]) -> list[IngestResult]:
        requests = [
            IngestRequest(
                doc_id=doc["doc_id"],
                content=doc["content"],
                metadata=doc.get("metadata", {})
            )
            for doc in documents
        ]
        return await self._ingestion.ingest_batch(requests)

    async def delete_document(self, doc_id: str) -> None:
        await self._ingestion.delete(doc_id)

    async def retrieve(self, query: str, top_k: int = 5, filters: dict | None = None) -> list[RetrievedChunk]:
        return await self._retrieval.retrieve(
            RetrievalRequest(query=query, top_k=top_k, filters=filters)
        )

    async def chat(self, request: ChatRequest) -> ChatResponse:
        # Retrieve relevant chunks
        sources = await self._retrieval.retrieve(
            RetrievalRequest(
                query=request.query,
                top_k=request.top_k,
                filters=request.filters
            )
        )

        # Build context from chunks
        context = self._build_context(sources)

        # Build prompt
        system_prompt = request.system_prompt or DEFAULT_SYSTEM_PROMPT
        prompt = self._build_prompt(
            system_prompt=system_prompt,
            context=context,
            query=request.query
        )

        # Generate answer
        answer = await self._llm.generate(prompt)

        return ChatResponse(answer=answer, sources=sources)

    async def stream_chat(self, request: ChatRequest):
        # Retrieve
        sources = await self._retrieval.retrieve(
            RetrievalRequest(
                query=request.query,
                top_k=request.top_k,
                filters=request.filters
            )
        )

        # Build prompt
        context = self._build_context(sources)
        system_prompt = request.system_prompt or DEFAULT_SYSTEM_PROMPT
        prompt = self._build_prompt(system_prompt, context, request.query)

        # Stream answer + yield sources at the end
        async for token in self._llm.stream(prompt):
            yield token, None

        yield None, sources

    def _build_context(self, chunks: list[RetrievedChunk]) -> str:
        if not chunks:
            return "No relevant context found."

        parts = []
        for i, chunk in enumerate(chunks, 1):
            parts.append(f"[{i}] {chunk.content}")
        return "\n\n".join(parts)

    def _build_prompt(self, system_prompt: str, context: str, query: str) -> str:
        prompt = f"""{system_prompt}

        Context:
        {context}

        Question: {query}

        Answer:"""

        return prompt
