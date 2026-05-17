import logging
import os
from contextlib import asynccontextmanager

from app.core.config import get_settings
from app.engines.factory import RAGEngineFactory
from app.infrastructure.embedder.factory import EmbedderFactory
from app.infrastructure.llm.factory import LLMFactory
from app.infrastructure.vector_store.factory import VectorStoreFactory
from app.services.ingestion.factory import IngestionFactory
from app.services.retrieval.factory import RetrievalFactory
from app.services.loaders.factory import LoaderFactory
from fastapi import FastAPI

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # --- STARTUP PHASE ---
    settings = get_settings()

    # Setup Environment
    os.makedirs(settings.chroma_path, exist_ok=True)
    os.makedirs(settings.images_path, exist_ok=True)

    # Build Infrastructure
    llm = LLMFactory.create(
        base_url=settings.ollama_base_url,
        model=settings.ollama_llm_model
    )
    embedder = EmbedderFactory.create(
        base_url=settings.ollama_base_url,
        model=settings.ollama_embed_model
    )
    vector_store = VectorStoreFactory.create(
        path=settings.chroma_path,
        collection_name=settings.chroma_collection_name
    )

    # Build Services
    ingestion = IngestionFactory.create(
        embedder=embedder,
        vector_store=vector_store
    )

    retrieval = RetrievalFactory.create(
        embedder=embedder,
        vector_store=vector_store,
        bm25_weight=settings.bm25_weight,
        vector_weight=settings.vector_weight
    )

    loader = LoaderFactory()

    # Build Engine & BIND TO APP.STATE
    app.state.rag_engine = RAGEngineFactory.create(
        llm=llm,
        ingestion=ingestion,
        retrieval=retrieval,
        loader=loader
    )

    yield

    # --- SHUTDOWN PHASE ---
    logger.info("Shutting down...")