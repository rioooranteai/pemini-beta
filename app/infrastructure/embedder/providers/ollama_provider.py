from ollama import AsyncClient
from app.infrastructure.embedder.base.embedder_base import EmbedderBase


class OllamaEmbedderProvider(EmbedderBase):

    def __init__(self, base_url: str, model: str) -> None:
        self._client = AsyncClient(host=base_url)
        self._model = model

    async def embed(self, text: str) -> list[float]:
        response = await self._client.embed(
            model=self._model,
            input=text
        )
        return response.embeddings[0]

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        response = await self._client.embed(
            model=self._model,
            input=texts
        )
        return response.embeddings