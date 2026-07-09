from abc import ABC, abstractmethod
import os
import cohere

from app.retrieval.models import RetrievedChunk


class BaseReranker(ABC):
    """Abstract interface for pluggable cross-encoder rerankers."""

    @abstractmethod
    async def rerank(
        self, query: str, chunks: list[RetrievedChunk], top_k: int
    ) -> list[RetrievedChunk]:
        """Reranks the retrieved chunks based on their relevance to the query.

        Args:
            query (str): The search query.
            chunks (list[RetrievedChunk]): Chunks to rerank.
            top_k (int): Number of top results to return.

        Returns:
            list[RetrievedChunk]: Re-scored and sorted chunks.
        """
        pass


class MockReranker(BaseReranker):
    """Deterministic mock cross-encoder reranker for testing and CI."""

    async def rerank(
        self, query: str, chunks: list[RetrievedChunk], top_k: int
    ) -> list[RetrievedChunk]:
        if not chunks:
            return []

        scored_chunks: list[RetrievedChunk] = []
        query_words = set(query.lower().split())

        for chunk in chunks:
            # Score based on overlap of words to simulate keyword relevance
            chunk_words = set(chunk.text.lower().split())
            overlap = len(query_words.intersection(chunk_words))
            
            # Tie breaker: text length
            score = float(overlap) + (len(chunk.text) / 10000.0)

            scored_chunks.append(
                RetrievedChunk(
                    text=chunk.text,
                    source_filename=chunk.source_filename,
                    file_type=chunk.file_type,
                    page_number=chunk.page_number,
                    chunk_index=chunk.chunk_index,
                    score=score,
                )
            )

        # Sort descending by score
        scored_chunks.sort(key=lambda c: c.score, reverse=True)
        return scored_chunks[:top_k]


class CohereReranker(BaseReranker):
    """Reranker implementation using Cohere's async Rerank API."""

    def __init__(self, api_key: str | None = None, model: str = "rerank-english-v3.0") -> None:
        self.api_key = api_key or os.environ.get("COHERE_API_KEY")
        self.model = model
        self.client = cohere.AsyncClient(api_key=self.api_key) if self.api_key else None

    async def rerank(
        self, query: str, chunks: list[RetrievedChunk], top_k: int
    ) -> list[RetrievedChunk]:
        if not chunks:
            return []
        if not self.client:
            # Fallback to mock scoring if no API key is set
            mock = MockReranker()
            return await mock.rerank(query, chunks, top_k)

        documents = [chunk.text for chunk in chunks]
        response = await self.client.rerank(
            model=self.model,
            query=query,
            documents=documents,
            top_n=top_k,
        )

        reranked_chunks: list[RetrievedChunk] = []
        for result in response.results:
            original_chunk = chunks[result.index]
            reranked_chunks.append(
                RetrievedChunk(
                    text=original_chunk.text,
                    source_filename=original_chunk.source_filename,
                    file_type=original_chunk.file_type,
                    page_number=original_chunk.page_number,
                    chunk_index=original_chunk.chunk_index,
                    score=float(result.relevance_score),
                )
            )

        return reranked_chunks
