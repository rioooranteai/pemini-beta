from ollama import AsyncClient
from app.infrastructure.llm.base.llm_base import LLMBase


class OllamaLLMProvider(LLMBase):

    def __init__(self, base_url: str, model: str) -> None:
        self._client = AsyncClient(host=base_url)
        self._model = model

    async def generate(self, prompt: str) -> str:
        response = await self._client.chat(
            model=self._model,
            messages=[{"role": "user", "content": prompt}]
        )
        return response.message.content

    async def stream(self, prompt: str):
        async for chunk in await self._client.chat(
            model=self._model,
            messages=[{"role": "user", "content": prompt}],
            stream=True
        ):
            yield chunk.message.content