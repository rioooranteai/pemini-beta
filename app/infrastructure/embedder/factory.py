from app.infrastructure.embedder.base.embedder_base import EmbedderBase
from app.infrastructure.embedder.providers.ollama_provider import OllamaEmbedderProvider


class EmbedderFactory:

    @staticmethod
    def create(base_url: str, model: str) -> EmbedderBase:
        return OllamaEmbedderProvider(
                base_url=base_url,
                model=model
            )