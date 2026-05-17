from abc import ABC, abstractmethod


class LLMBase(ABC):

    @abstractmethod
    async def generate(self, prompt: str) -> str:
        """Generate text from a prompt."""
        ...

    @abstractmethod
    async def stream(self, prompt: str):
        """Stream generated text token by token."""
        ...