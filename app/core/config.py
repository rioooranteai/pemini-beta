from pydantic_settings import BaseSettings
from pydantic import ConfigDict
from functools import lru_cache


class Settings(BaseSettings):
    model_config = ConfigDict(
        env_file=".env",
        env_file_encoding="utf-8"
    )

    # Ollama
    ollama_base_url: str
    ollama_llm_model: str
    ollama_embed_model: str

    # ChromaDB
    chroma_path: str = "./data/chroma_db"
    chroma_collection_name: str = "rag_documents"

    # Storage
    images_path: str = "./data/images"

    # RAG
    top_k: int = 5
    bm25_weight: float = 0.4
    vector_weight: float = 0.6


@lru_cache
def get_settings() -> Settings:
    return Settings()