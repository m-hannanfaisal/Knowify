from abc import ABC, abstractmethod
import hashlib
from openai import AsyncOpenAI

from app.core.config import settings


class BaseEmbeddingProvider(ABC):
    """Interface for pluggable embedding providers."""

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Returns the dimension of the embedding vectors."""
        pass

    @abstractmethod
    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embeds a list of text documents.

        Args:
            texts (list[str]): The list of text chunks.

        Returns:
            list[list[float]]: A list of embedding vectors.
        """
        pass


class MockEmbeddingProvider(BaseEmbeddingProvider):
    """Generates deterministic mock embeddings for offline testing and CI."""

    def __init__(self, dimension: int = 1536) -> None:
        self._dimension = dimension

    @property
    def dimension(self) -> int:
        return self._dimension

    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        embeddings: list[list[float]] = []
        for text in texts:
            hash_val = hashlib.sha256(text.encode("utf-8")).digest()
            vector: list[float] = []
            for i in range(self._dimension):
                byte_val = hash_val[i % len(hash_val)]
                val = (byte_val / 255.0) - 0.5
                vector.append(val)
            embeddings.append(vector)
        return embeddings


class OpenAIEmbeddingProvider(BaseEmbeddingProvider):
    """Provides async embeddings via the OpenAI API client."""

    def __init__(self, api_key: str | None = None, model: str = "text-embedding-3-small") -> None:
        self._api_key = api_key or settings.LLM_API_KEY
        self._model = model
        self.client = AsyncOpenAI(api_key=self._api_key)

    @property
    def dimension(self) -> int:
        if "3-small" in self._model:
            return 1536
        return 1536

    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        response = await self.client.embeddings.create(
            input=texts,
            model=self._model
        )
        return [data.embedding for data in response.data]
