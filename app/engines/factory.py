from app.engines.rag_engine import RAGEngine
from app.infrastructure.llm.base.llm_base import LLMBase
from app.services.ingestion.base.ingestion_base import IngestionBase
from app.services.retrieval.base.retrieval_base import RetrievalBase


class RAGEngineFactory:

    @staticmethod
    def create(llm: LLMBase,
               ingestion: IngestionBase,
               retrieval: RetrievalBase) -> RAGEngine:
        return RAGEngine(
                llm=llm,
                ingestion=ingestion,
                retrieval=retrieval
            )