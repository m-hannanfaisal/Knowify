from abc import ABC, abstractmethod
import hashlib
import asyncio
from openai import AsyncOpenAI
from sentence_transformers import SentenceTransformer

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
    """Provides embeddings via local SentenceTransformer or OpenAI client depending on API keys."""

    def __init__(self, api_key: str | None = None, model: str | None = None) -> None:
        self._api_key = api_key or settings.LLM_API_KEY
        
        # Use local sentence-transformers if using Groq (which lacks embedding endpoint) or placeholder key
        is_local = (not self._api_key) or self._api_key == "placeholder_key" or self._api_key.startswith("gsk_")
        
        if is_local:
            self._model = model or "all-MiniLM-L6-v2"
            self._local_model = SentenceTransformer(self._model)
            self._is_local = True
            self.client = None
        else:
            self._model = model or "text-embedding-3-small"
            self._is_local = False
            self.client = AsyncOpenAI(api_key=self._api_key)

        import structlog
        logger = structlog.get_logger(__name__)
        logger.info("embedding_model_initialized", provider="OpenAIEmbeddingProvider", model=self._model, is_local=self._is_local)

    @property
    def dimension(self) -> int:
        if self._is_local:
            return 384
        return 1536

    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        if self._is_local:
            # Offload blocking CPU-bound local model inference to background threads
            embeddings = await asyncio.to_thread(self._local_model.encode, texts)
            return [e.tolist() for e in embeddings]
        else:
            response = await self.client.embeddings.create(
                input=texts,
                model=self._model
            )
            return [data.embedding for data in response.data]
