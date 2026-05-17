from app.infrastructure.llm.base.llm_base import LLMBase
from app.infrastructure.llm.providers.ollama_provider import OllamaLLMProvider


class LLMFactory:

    @staticmethod
    def create(base_url: str,
               model: str) -> LLMBase:
        return OllamaLLMProvider(
            base_url=base_url,
            model=model
        )
